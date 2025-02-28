"""MetaNet-aTLAS模型评估脚本

此脚本用于评估MetaNet-aTLAS模型在不同数据集上的性能。
"""

import os
import json
import torch

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.metanet_composition import MetaNetImageEncoder, MetaNetLinearizedModel


def main(args):
    """主函数，评估MetaNet-aTLAS模型

    参数:
    ----------
    args: argparse.Namespace
        命令行参数
    """
    print("*" * 100)
    print(f"评估MetaNet-aTLAS模型（{args.finetuning_mode}模式）")
    print("*" * 100)

    # 加载任务向量池
    # pool = [
    #     "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
    #     "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
    #     "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
    # ]
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
    ]

    # 记录结果
    all_results = {}

    for dataset in args.eval_datasets:
        print("-" * 100)
        print(f"评估 {dataset}")

        # 获取任务向量，排除当前数据集
        task_vectors = []
        for ds in pool:
            if ds == dataset:
                continue

            if args.finetuning_mode == "linear":
                pretrained_checkpoint = f"{args.save}/{ds}Val/linear_zeroshot.pt"
                finetuned_checkpoint = f"{args.save}/{ds}Val/linear_finetuned.pt"
                task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))
            else:
                pretrained_checkpoint = f"{args.save}/{ds}Val/zeroshot.pt"
                finetuned_checkpoint = f"{args.save}/{ds}Val/finetuned.pt"
                task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

        # 加载模型和Meta-Net
        if args.finetuning_mode == "linear":
            image_encoder = LinearizedImageEncoder(args, keep_lang=False)
            image_encoder.model = MetaNetLinearizedModel(
                image_encoder.model, task_vectors, blockwise=args.blockwise_coef
            )

            # 加载训练好的Meta-Net
            meta_net_path = os.path.join(args.save, f"{dataset}Val", "learned_linear_metanet.pt")
            if os.path.exists(meta_net_path):
                image_encoder.model.meta_net.load_state_dict(torch.load(meta_net_path))
            else:
                print(f"警告：找不到{meta_net_path}，使用未训练的Meta-Net")

        else:
            image_encoder = ImageEncoder(args)
            image_encoder = MetaNetImageEncoder(
                image_encoder, task_vectors, blockwise=args.blockwise_coef
            )

            # 加载训练好的Meta-Net
            meta_net_path = os.path.join(args.save, f"{dataset}Val", "learned_metanet.pt")
            if os.path.exists(meta_net_path):
                image_encoder.meta_net.load_state_dict(torch.load(meta_net_path))
            else:
                print(f"警告：找不到{meta_net_path}，使用未训练的Meta-Net")

        # 评估
        result = eval_single_dataset(image_encoder, dataset, args)
        all_results[dataset] = result["top1"]
        print(f"{dataset} 测试准确率: {100 * result['top1']:.2f}%")

    # 计算平均准确率
    avg_acc = sum(all_results.values()) / len(all_results)
    all_results["average"] = avg_acc
    print("-" * 100)
    print(f"平均准确率: {100 * avg_acc:.2f}%")

    # 保存结果
    if args.finetuning_mode == "linear":
        results_path = os.path.join(args.save, "metanet_linear_results.json")
    else:
        results_path = os.path.join(args.save, "metanet_results.json")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"结果已保存到 {results_path}")


if __name__ == "__main__":
    args = parse_arguments()

    # 设置评估数据集
    if args.eval_datasets is None:
        args.eval_datasets = [
            "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"
        ]

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

    main(args)