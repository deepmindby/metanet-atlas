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
import matplotlib.pyplot as plt
import numpy as np

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


def plot_accuracy_curve(accuracies, dataset_name, save_dir):
    """Plot and save accuracy curve

    Args:
        accuracies: list of accuracy values for each epoch
        dataset_name: string, name of the dataset
        save_dir: string, directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(accuracies) + 1))

    plt.plot(epochs, [acc*100 for acc in accuracies], 'b-o', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(f'Validation Accuracy vs. Epochs for {dataset_name}', fontsize=16)

    # Set y-axis range slightly wider than the data range
    min_acc = min(accuracies) * 100 - 1
    max_acc = max(accuracies) * 100 + 1
    plt.ylim(min_acc, max_acc)

    # Annotate final accuracy
    final_acc = accuracies[-1] * 100
    plt.annotate(f'Final: {final_acc:.2f}%',
                 xy=(len(accuracies), final_acc),
                 xytext=(len(accuracies)-0.5, final_acc+0.5),
                 fontsize=12,
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'{dataset_name}_accuracy_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Accuracy curve saved to: {plot_path}")

    # Save data as JSON
    data_path = os.path.join(save_dir, f'{dataset_name}_accuracy_data.json')
    with open(data_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'epochs': epochs,
            'accuracies': [float(acc) for acc in accuracies]
        }, f, indent=4)

    print(f"Accuracy data saved to: {data_path}")


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

    for dataset, num_epochs in args.datasets.items():
        args.target_dataset = dataset + "Val"
        args.epochs = num_epochs  # Use dataset-specific epochs
        print("=" * 100)
        print(f"Learning MetaNet-aTLAS for {dataset} with {args.model} for {args.epochs} epochs")
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
    # Setup for training
    target_dataset = args.target_dataset
    ckpdir = os.path.join(args.save, target_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    # Add monitoring variables
    train_losses = []
    val_accuracies = []
    old_params = None
    param_changes = []
    grad_norms = []
    coefficient_stats = []

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

            # Check Meta-Net coefficients periodically
            if i % (print_every * 5) == 0 and is_main_process():
                with torch.no_grad():
                    # Get a small batch of samples for analysis
                    sample_batch = inputs[:min(4, inputs.shape[0])]

                    # Get base features and calculate coefficients
                    if linearized_finetuning:
                        base_features = ddp_model.module.image_encoder.model.func0(
                            ddp_model.module.image_encoder.model.params0,
                            ddp_model.module.image_encoder.model.buffers0,
                            sample_batch
                        )
                        coefficients = ddp_model.module.image_encoder.model.meta_net(base_features)
                    else:
                        # Get the features directly
                        base_features = ddp_model.module.image_encoder.func(
                            ddp_model.module.image_encoder.params,
                            ddp_model.module.image_encoder.buffer,
                            sample_batch
                        )
                        coefficients = ddp_model.module.image_encoder.meta_net(base_features)

                    # Calculate and print coefficient statistics
                    coef_mean = coefficients.mean().item()
                    coef_std = coefficients.std().item()
                    coef_min = coefficients.min().item()
                    coef_max = coefficients.max().item()

                    coefficient_stats.append({
                        'mean': coef_mean,
                        'std': coef_std,
                        'min': coef_min,
                        'max': coef_max
                    })

                    print(f"Meta-Net coefficients - Mean: {coef_mean:.4f}, Std: {coef_std:.4f}, Min: {coef_min:.4f}, Max: {coef_max:.4f}")

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs)
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
                # Apply regularization
                meta_net_params = torch.cat([p.flatten() for p in meta_net.parameters()])
                reg = lp_reg(meta_net_params, args.lp_reg, gamma=0.1)
                loss = loss + reg
                # Scale loss
                loss = loss / args.num_grad_accumulation

            # Record loss
            if is_main_process():
                train_losses.append(loss.item())

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                # Calculate gradient norm before clipping
                grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
                if is_main_process():
                    grad_norms.append(grad_norm.item())

                # Track parameter changes
                if old_params is None and is_main_process():
                    old_params = {name: param.clone().detach() for name, param in meta_net.named_parameters()}

                # Execute optimizer step
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scheduler(step)
                scaler.step(optimizer)
                scaler.update()

                # Calculate parameter changes after update
                if old_params is not None and is_main_process():
                    current_changes = []
                    for name, param in meta_net.named_parameters():
                        if name in old_params:
                            change = (param - old_params[name]).abs().mean().item()
                            current_changes.append(change)
                            old_params[name] = param.clone().detach()

                    avg_change = sum(current_changes) / len(current_changes) if current_changes else 0
                    param_changes.append(avg_change)

                    if step % print_every == 0:
                        print(f"Step {step} - Gradient norm: {grad_norm:.6f}, Avg param change: {avg_change:.8f}")

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
            val_accuracies.append(acc)
            print(f"Epoch {epoch+1}/{args.epochs} validation accuracy: {acc*100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_meta_net = {k: v.data.clone() for k, v in meta_net.state_dict().items()}
                # Save best model
                torch.save(best_meta_net, head_path)

    # Save results and generate accuracy plot
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

        # Plot accuracy curve
        if val_accuracies:
            plot_dir = os.path.join(args.save, "accuracy_plots")
            plot_accuracy_curve(val_accuracies, target_dataset_no_val, plot_dir)

        # Plot training curves
        if train_losses and len(train_losses) > 1:
            # Create plots directory
            plot_dir = os.path.join(args.save, "monitoring_plots")
            os.makedirs(plot_dir, exist_ok=True)

            # Plot training loss
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, 'r-')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title(f'Training Loss for {target_dataset.replace("Val", "")}')
            plt.savefig(os.path.join(plot_dir, f"{target_dataset.replace('Val', '')}_loss_curve.png"))
            plt.close()

            # Plot gradient norms
            if grad_norms:
                plt.figure(figsize=(10, 6))
                plt.plot(grad_norms, 'g-')
                plt.xlabel('Optimization Steps')
                plt.ylabel('Gradient Norm')
                plt.title(f'Gradient Norms for {target_dataset.replace("Val", "")}')
                plt.savefig(os.path.join(plot_dir, f"{target_dataset.replace('Val', '')}_grad_norms.png"))
                plt.close()

            # Plot parameter changes
            if param_changes:
                plt.figure(figsize=(10, 6))
                plt.plot(param_changes, 'b-')
                plt.xlabel('Optimization Steps')
                plt.ylabel('Average Parameter Change')
                plt.title(f'Parameter Changes for {target_dataset.replace("Val", "")}')
                plt.savefig(os.path.join(plot_dir, f"{target_dataset.replace('Val', '')}_param_changes.png"))
                plt.close()

            # Plot coefficient statistics
            if coefficient_stats:
                means = [stats['mean'] for stats in coefficient_stats]
                stds = [stats['std'] for stats in coefficient_stats]
                plt.figure(figsize=(10, 6))
                plt.plot(means, 'b-', label='Mean')
                plt.fill_between(range(len(means)),
                                [m-s for m,s in zip(means, stds)],
                                [m+s for m,s in zip(means, stds)],
                                alpha=0.2, color='b')
                plt.xlabel('Checkpoints')
                plt.ylabel('Coefficient Value')
                plt.title(f'Meta-Net Coefficient Statistics for {target_dataset.replace("Val", "")}')
                plt.legend()
                plt.savefig(os.path.join(plot_dir, f"{target_dataset.replace('Val', '')}_coef_stats.png"))
                plt.close()

            # Save monitoring data as JSON
            monitoring_data = {
                'train_losses': train_losses,
                'grad_norms': grad_norms,
                'param_changes': param_changes,
                'coefficient_stats': coefficient_stats
            }
            with open(os.path.join(plot_dir, f"{target_dataset.replace('Val', '')}_monitoring_data.json"), 'w') as f:
                json.dump(monitoring_data, f, indent=4)

        # Save results
        with open(log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)


if __name__ == "__main__":
    # Target datasets and training epochs
    target_datasets = {
        "Cars": 10,  # 35           # Fine-grained car classification
        "DTD": 10,  # 76           # Texture description
        "EuroSAT": 10,  # 12      # Satellite imagery
        "GTSRB": 10,  # 11          # Traffic signs
        "MNIST": 10,  # 5         # Handwritten digits
        "RESISC45": 10,  # 15       # Remote sensing imagery
        "SUN397": 10,  # 14      # Scene classification
        "SVHN": 10,  # 4            # Street view house numbers
    }

    # Parse command line arguments
    args = parse_arguments()
    args.datasets = target_datasets
    # Set default parameters
    args.lr = 1e-2  # MetaNet parameters need higher learning rate

    # Use gradient accumulation to simulate larger batch sizes
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    # if args.seed is not None:
    #     args.save = f"checkpoints_{args.seed}/{args.model}"
    # else:
    #     args.save = f"checkpoints/{args.model}"
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