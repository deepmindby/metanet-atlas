"""MetaNet-aTLAS模型训练脚本

此脚本用于训练MetaNet-aTLAS模型，该模型使用Meta-Net为每个样本动态生成
任务向量组合系数，实现样本级别的知识组合。
"""

import os
import time
import json
import torch
import torchvision

from torch.cuda.amp import GradScaler
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.metanet_composition import MetaNetImageEncoder, MetaNetLinearizedModel

from src.utils import cosine_lr
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp


def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    """L1或L2正则化项"""
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()


def main(rank, args):
    """主函数，训练MetaNet-aTLAS模型"""

    # 加载任务向量池
    # pool = [
    #     "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
    #     "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
    #     "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
    # ]
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
    ]
    task_vectors = {}
    for dataset in pool:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors[dataset] = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    args.rank = rank
    for dataset in args.datasets:
        args.target_dataset = dataset + "Val"
        print("=" * 100)
        print(f"Learning MetaNet-aTLAS for {dataset} with {args.model}")
        print("=" * 100)

        train(task_vectors, args)


def train(task_vectors, args):
    """训练函数

    参数:
    ----------
    task_vectors: Dict[str, TaskVector]
        任务向量字典
    args: argparse.Namespace
        命令行参数
    """
    setup_ddp(args.rank, args.world_size, port=args.port)
    target_dataset = args.target_dataset
    ckpdir = os.path.join(args.save, target_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "只支持linear和standard微调模式。"

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("使用线性化微调。")

    # 获取任务向量，排除目标数据集的任务向量
    orig_dataset = target_dataset.replace("Val", "")
    task_vectors_list = [v for k, v in task_vectors.items() if orig_dataset != k]

    # 初始化模型
    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = MetaNetLinearizedModel(
            image_encoder.model, task_vectors_list, blockwise=args.blockwise_coef
        )
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = MetaNetImageEncoder(
            image_encoder, task_vectors_list, blockwise=args.blockwise_coef
        )

    # 获取分类头
    classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    # 冻结分类头
    model.freeze_head()
    model = model.cuda()

    # 使用更激进的随机裁剪和水平翻转预处理
    preprocess_fn = torchvision.transforms.Compose([
                                                       torchvision.transforms.RandomResizedCrop(
                                                           size=224, scale=(0.5, 1),
                                                           interpolation=torchvision.transforms.InterpolationMode.BICUBIC
                                                       ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                   ] + model.train_preprocess.transforms[-3:])

    # 获取数据集和数据加载器
    dataset = get_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(data_loader)

    # 打印损失的频率设置
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    # 分布式训练设置
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 只训练需要梯度的参数
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # 不使用预热阶段的学习率调度器
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # 保存路径设置
    if linearized_finetuning:
        head_path = os.path.join(ckpdir, "learned_linear_metanet.pt")
        log_path = os.path.join(args.save, "learned_linear_metanet.json")
        meta_net = ddp_model.module.image_encoder.model.meta_net
    else:
        head_path = os.path.join(ckpdir, "learned_metanet.pt")
        log_path = os.path.join(args.save, "learned_metanet.json")
        meta_net = ddp_model.module.image_encoder.meta_net

    # 混合精度训练
    scaler = GradScaler()

    # 记录零样本准确率
    if is_main_process():
        print(f"=> 在 {target_dataset} 上的零样本准确率：\t{100 * args.zs_acc[target_dataset]:.2f}%.")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                comp_acc = json.load(f)
        else:
            comp_acc = {}

    # 记录最佳状态
    best_meta_net = None
    best_acc = args.zs_acc[target_dataset]

    # 开始训练
    for epoch in range(args.epochs):
        ddp_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                    i // args.num_grad_accumulation
                    + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs)
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
                # 应用正则化
                meta_net_params = torch.cat([p.flatten() for p in meta_net.parameters()])
                reg = lp_reg(meta_net_params, args.lp_reg)
                loss = loss + reg
                # 缩放损失
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            # 打印训练信息
            if (
                    step % print_every == 0
                    and ((i + 1) % args.num_grad_accumulation == 0)
                    and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"训练周期: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"
                    f"损失: {loss.item():.6f}\t数据时间 {data_time:.3f}\t批次时间 {batch_time:.3f}",
                    flush=True,
                )

        # 每个epoch后评估
        if is_main_process():
            image_encoder = ddp_model.module.image_encoder
            acc = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
            if acc > best_acc:
                best_acc = acc
                best_meta_net = {k: v.data.clone() for k, v in meta_net.state_dict().items()}
                # 保存最佳模型
                torch.save(best_meta_net, head_path)

    # 保存结果
    if is_main_process():
        comp_acc[target_dataset] = best_acc
        target_dataset_no_val = target_dataset.replace("Val", "")

        # 加载最佳模型进行测试
        if best_meta_net is not None:
            # 找到更好的模型，加载它
            if linearized_finetuning:
                ddp_model.module.image_encoder.model.meta_net.load_state_dict(best_meta_net)
            else:
                ddp_model.module.image_encoder.meta_net.load_state_dict(best_meta_net)
            print(f"加载精度为 {best_acc * 100:.2f}% 的最佳模型")
        else:
            # 没有找到更好的模型，使用当前模型
            print(f"警告：训练未能超过零样本精度 {args.zs_acc[target_dataset] * 100:.2f}%")
            print("使用最后一个epoch的模型进行测试")

        # 在测试集上评估
        image_encoder = ddp_model.module.image_encoder
        comp_acc[target_dataset_no_val] = eval_single_dataset(image_encoder, target_dataset_no_val, args)["top1"]

        # 保存结果
        with open(log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)

    cleanup_ddp()


if __name__ == "__main__":
    # 目标数据集和训练周期
    target_datasets = {
        "Cars": 5,  # 35
        "DTD": 5,  # 76
        "EuroSAT": 5,  # 13
        "GTSRB": 5,  # 11
        "MNIST": 5,  # 5
        "RESISC45": 5,  # 15
        "SUN397": 5,  # 14
        "SVHN": 5,  # 4
        # "CIFAR10": 5,
        # "CIFAR100": 6,
        # "ImageNet": 10,
        # "STL10": 4,
        # "Food101": 15,
        # "Caltech101": 10,
        # "Caltech256": 8,
        # "FGVCAircraft": 60,
        # "Flowers102": 40,
        # "OxfordIIITPet": 5,
        # "CUB200": 20,
        # "PascalVOC": 10,
        # "Country211": 15,
        # "UCF101": 20,
    }

    # 解析命令行参数
    args = parse_arguments()
    args.datasets = target_datasets
    # 设置默认参数
    args.lr = 1e-2  # MetaNet参数需要较高的学习率
    args.epochs = 5
    # 使用梯度累积模拟更大的批次大小
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"


    # 加载零样本准确率
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)

    save_dir = args.save
    if args.subsample is not None:
        save_dir += f"_{args.subsample * 100:.0f}perc"
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)

    # 启动分布式训练
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)