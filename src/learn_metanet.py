"""MetaNet-aTLAS Model Training Script

This script is used to train the MetaNet-aTLAS model, which uses Meta-Net to dynamically
generate task vector combination coefficients for each sample, achieving sample-level
knowledge composition.
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
    """L1 or L2 regularization term"""
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()


def main(rank, args):
    """Main function for training MetaNet-aTLAS model"""

    # Load task vector pool
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
    setup_ddp(args.rank, args.world_size, port=args.port)

    for dataset in args.datasets:
        args.target_dataset = dataset + "Val"
        print("=" * 100)
        print(f"Learning MetaNet-aTLAS for {dataset} with {args.model}")
        print("=" * 100)

        train(task_vectors, args)

    cleanup_ddp()


def train(task_vectors, args):
    """Training function

    Parameters:
    ----------
    task_vectors: Dict[str, TaskVector]
        Dictionary of task vectors
    args: argparse.Namespace
        Command line arguments
    """
    # setup_ddp(args.rank, args.world_size, port=args.port)
    target_dataset = args.target_dataset
    ckpdir = os.path.join(args.save, target_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning modes are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # Get task vectors, excluding the target dataset's task vector
    orig_dataset = target_dataset.replace("Val", "")
    task_vectors_list = [v for k, v in task_vectors.items() if orig_dataset != k]

    # Initialize model
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

    # Get classification head
    classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    # Freeze classification head
    model.freeze_head()
    model = model.cuda()

    # Use more aggressive random crop with horizontal flip preprocessing
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + model.train_preprocess.transforms[-3:])

    # Get dataset and data loader
    dataset = get_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(data_loader)

    # Set print frequency
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    # Distributed training setup
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Only train parameters that require gradients
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Learning rate scheduler without warmup
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # Setup save paths
    if linearized_finetuning:
        head_path = os.path.join(ckpdir, "learned_linear_metanet.pt")
        log_path = os.path.join(args.save, "learned_linear_metanet.json")
        meta_net = ddp_model.module.image_encoder.model.meta_net
    else:
        head_path = os.path.join(ckpdir, "learned_metanet.pt")
        log_path = os.path.join(args.save, "learned_metanet.json")
        meta_net = ddp_model.module.image_encoder.meta_net

    # Mixed precision training
    scaler = GradScaler()

    # Record zero-shot accuracy
    if is_main_process():
        print(f"=> Zero-shot accuracy on {target_dataset}: {100 * args.zs_acc[target_dataset]:.2f}%.")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                comp_acc = json.load(f)
        else:
            comp_acc = {}

    # Record best state
    best_meta_net = None
    best_acc = args.zs_acc[target_dataset]

    # Start training
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
                # Apply regularization
                meta_net_params = torch.cat([p.flatten() for p in meta_net.parameters()])
                reg = lp_reg(meta_net_params, args.lp_reg)
                loss = loss + reg
                # Scale loss
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            # Print training info
            if (
                    step % print_every == 0
                    and ((i + 1) % args.num_grad_accumulation == 0)
                    and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Training epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"
                    f"Loss: {loss.item():.6f}\tData time: {data_time:.3f}\tBatch time: {batch_time:.3f}",
                    flush=True,
                )

        # Evaluate after each epoch
        if is_main_process():
            image_encoder = ddp_model.module.image_encoder
            acc = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
            if acc > best_acc:
                best_acc = acc
                best_meta_net = {k: v.data.clone() for k, v in meta_net.state_dict().items()}
                # Save best model
                torch.save(best_meta_net, head_path)

    # Save results
    if is_main_process():
        comp_acc[target_dataset] = best_acc
        target_dataset_no_val = target_dataset.replace("Val", "")

        # Load best model for testing
        if best_meta_net is not None:
            # Found a better model, load it
            if linearized_finetuning:
                ddp_model.module.image_encoder.model.meta_net.load_state_dict(best_meta_net)
            else:
                ddp_model.module.image_encoder.meta_net.load_state_dict(best_meta_net)
            print(f"Loaded best model with accuracy {best_acc * 100:.2f}%")
        else:
            # No better model found, use current model
            print(f"Warning: Training did not exceed zero-shot accuracy {args.zs_acc[target_dataset] * 100:.2f}%")
            print("Using the model from the last epoch for testing")

        # Evaluate on test set
        image_encoder = ddp_model.module.image_encoder
        comp_acc[target_dataset_no_val] = eval_single_dataset(image_encoder, target_dataset_no_val, args)["top1"]

        # Save results
        with open(log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)

    # cleanup_ddp()


if __name__ == "__main__":
    # Target datasets and training epochs
    target_datasets = {
        "Cars": 35,  # 35
        "DTD": 76,  # 76
        "EuroSAT": 13,  # 13
        "GTSRB": 11,  # 11
        "MNIST": 5,  # 5
        "RESISC45": 15,  # 15
        "SUN397": 14,  # 14
        "SVHN": 4,  # 4
    }

    # Parse command line arguments
    args = parse_arguments()
    args.datasets = target_datasets
    # Set default parameters
    args.lr = 1e-2  # MetaNet parameters need higher learning rate
    args.epochs = 5
    # Use gradient accumulation to simulate larger batch sizes
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"


    # Load zero-shot accuracy
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)

    save_dir = args.save + f"-meta"
    os.makedirs(save_dir, exist_ok=True)
    # if args.subsample is not None:
    #     save_dir += f"_{args.subsample * 100:.0f}perc"
    #     # Create directory
    #     os.makedirs(save_dir, exist_ok=True)

    # Launch distributed training
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)