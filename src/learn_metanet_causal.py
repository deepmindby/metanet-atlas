"""MetaNet-aTLAS Model Training with Causal Intervention

This script implements the training of MetaNet-aTLAS models with causal intervention techniques.
The causal intervention helps understand the contribution of different parameter blocks
and makes the model more robust to variations in any single parameter block.
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


def plot_accuracy_and_variance(accuracies, variances, dataset_name, save_dir):
    """Plot and save accuracy and variance curves

    Args:
        accuracies: list of accuracy values for each epoch
        variances: list of intervention variances for each epoch
        dataset_name: string, name of the dataset
        save_dir: string, directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    epochs = list(range(1, len(accuracies) + 1))

    # Plot accuracy curve
    ax1.plot(epochs, [acc * 100 for acc in accuracies], 'b-o', linewidth=2, markersize=8)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_title(f'Validation Accuracy vs. Epochs for {dataset_name}', fontsize=16)

    # Set y-axis range slightly wider than the data range for accuracy
    min_acc = min(accuracies) * 100 - 1
    max_acc = max(accuracies) * 100 + 1
    ax1.set_ylim(min_acc, max_acc)

    # Annotate final accuracy
    final_acc = accuracies[-1] * 100
    ax1.annotate(f'Final: {final_acc:.2f}%',
                 xy=(len(accuracies), final_acc),
                 xytext=(len(accuracies) - 0.5, final_acc + 0.5),
                 fontsize=12,
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Plot variance curve
    ax2.plot(epochs, variances, 'r-o', linewidth=2, markersize=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Intervention Variance', fontsize=14)
    ax2.set_title(f'Intervention Variance vs. Epochs for {dataset_name}', fontsize=16)

    # Annotate final variance
    final_var = variances[-1]
    ax2.annotate(f'Final: {final_var:.4f}',
                 xy=(len(variances), final_var),
                 xytext=(len(variances) - 0.5, final_var + 0.5),
                 fontsize=12,
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'{dataset_name}_causal_training.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Accuracy and variance curves saved to: {plot_path}")

    # Save data as JSON
    data_path = os.path.join(save_dir, f'{dataset_name}_causal_training_data.json')
    with open(data_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'epochs': epochs,
            'accuracies': [float(acc) for acc in accuracies],
            'variances': [float(var) for var in variances]
        }, f, indent=4)

    print(f"Training data saved to: {data_path}")


def main(rank, args):
    """Main function for training MetaNet-aTLAS model with causal intervention"""

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
        causal_str = "with causal intervention" if args.causal_intervention else "without causal intervention"
        print(f"Learning MetaNet-aTLAS for {dataset} {causal_str} with {args.model} for {args.epochs} epochs")
        print("=" * 100)

        train(task_vectors, args)

    cleanup_ddp()


def train(task_vectors, args):
    """Training function with causal intervention

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
    var_losses = []  # Variance intervention losses
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

    # Initialize model with causal intervention support
    if args.finetuning_mode == "linear":
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = MetaNetLinearizedModel(
            image_encoder.model, task_vectors_list,
            blockwise=args.blockwise_coef,
            enable_causal=args.causal_intervention,
            top_k_ratio=args.top_k_ratio
        )
    else:
        image_encoder = ImageEncoder(args)
        image_encoder = MetaNetImageEncoder(
            image_encoder, task_vectors_list,
            blockwise=args.blockwise_coef,
            enable_causal=args.causal_intervention,
            top_k_ratio=args.top_k_ratio
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
        ] + model.train_preprocess.transforms[-3:]
    )

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
        suffix = "_causal" if args.causal_intervention else ""
        head_path = os.path.join(ckpdir, f"learned_linear_metanet{suffix}.pt")
        log_path = os.path.join(args.save, f"learned_linear_metanet{suffix}.json")
        meta_net = ddp_model.module.image_encoder.model.meta_net
    else:
        suffix = "_causal" if args.causal_intervention else ""
        head_path = os.path.join(ckpdir, f"learned_metanet{suffix}.pt")
        log_path = os.path.join(args.save, f"learned_metanet{suffix}.json")
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
    epoch_var_losses = []  # Track variance losses per epoch

    # Start training
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_var_loss = 0.0
        total_batches = 0

        ddp_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                    i // args.num_grad_accumulation
                    + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass with main model
                logits = ddp_model(inputs)

                # Calculate main task loss
                task_loss = loss_fn(logits, labels)

                # Calculate variance penalty loss if causal intervention is enabled
                if args.causal_intervention:
                    if linearized_finetuning:
                        var_loss = ddp_model.module.image_encoder.model.compute_intervention_loss(inputs)
                    else:
                        var_loss = ddp_model.module.image_encoder.compute_intervention_loss(inputs)
                else:
                    var_loss = torch.tensor(0.0, device=inputs.device)

                # Apply regularization for meta-net parameters if needed
                meta_net_params = torch.cat([p.flatten() for p in meta_net.parameters()])
                reg_loss = lp_reg(meta_net_params, args.lp_reg, gamma=0.1)

                # Combine losses with variance penalty coefficient
                total_loss = task_loss + args.var_penalty_coef * var_loss + reg_loss

                # Scale loss for gradient accumulation
                scaled_loss = total_loss / args.num_grad_accumulation

            # Record losses
            epoch_loss += task_loss.item()
            epoch_var_loss += var_loss.item()
            total_batches += 1

            if is_main_process():
                train_losses.append(task_loss.item())
                var_losses.append(var_loss.item())

            # Backward pass
            scaler.scale(scaled_loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                # Calculate gradient norm before clipping
                grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
                if is_main_process():
                    grad_norms.append(grad_norm.item())

                # Track parameter changes
                if old_params is None and is_main_process():
                    old_params = {name: param.clone().detach() for name, param in meta_net.named_parameters()}

                # Execute optimizer step
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
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
                var_str = f", Var Loss: {var_loss.item():.6f}" if args.causal_intervention else ""
                print(
                    f"Training epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"
                    f"Task Loss: {task_loss.item():.6f}{var_str}\t"
                    f"Data time: {data_time:.3f}\tBatch time: {batch_time:.3f}",
                    flush=True,
                )

        # Record average losses for the epoch
        if total_batches > 0:
            avg_epoch_loss = epoch_loss / total_batches
            avg_epoch_var_loss = epoch_var_loss / total_batches
            epoch_var_losses.append(avg_epoch_var_loss)
            if is_main_process():
                print(
                    f"Epoch {epoch + 1} average - Task Loss: {avg_epoch_loss:.6f}, Var Loss: {avg_epoch_var_loss:.6f}")

        # Evaluate after each epoch
        if is_main_process():
            image_encoder = ddp_model.module.image_encoder
            acc = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
            val_accuracies.append(acc)
            print(f"Epoch {epoch + 1}/{args.epochs} validation accuracy: {acc * 100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_meta_net = {k: v.data.clone() for k, v in meta_net.state_dict().items()}
                # Save best model
                torch.save(best_meta_net, head_path)

    # Save results and generate plots
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

        # Save the causal intervention statistics in the model log
        if args.causal_intervention:
            comp_acc[f"{target_dataset}_var_losses"] = epoch_var_losses

        # Plot accuracy and variance curves if causal intervention was used
        if args.causal_intervention and val_accuracies and len(epoch_var_losses) > 0:
            plot_dir = os.path.join(args.save, "causal_plots")
            plot_accuracy_and_variance(val_accuracies, epoch_var_losses, target_dataset_no_val, plot_dir)

        # Save results
        with open(log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)

        # Save monitoring data
        if train_losses:
            plot_dir = os.path.join(args.save, "monitoring_plots")
            os.makedirs(plot_dir, exist_ok=True)

            monitoring_data = {
                'dataset': target_dataset_no_val,
                'causal_intervention': args.causal_intervention,
                'train_losses': train_losses,
                'var_losses': var_losses,
                'grad_norms': grad_norms,
                'param_changes': param_changes,
                'val_accuracies': val_accuracies,
                'epoch_var_losses': epoch_var_losses
            }

            with open(os.path.join(plot_dir, f"{target_dataset_no_val}_causal_monitoring.json"), 'w') as f:
                json.dump(monitoring_data, f, indent=4)


if __name__ == "__main__":
    # Target datasets and training epochs
    target_datasets = {
        # "Cars": 10,
        # "DTD": 10,
        # "EuroSAT": 10,
        # "GTSRB": 10,
        "MNIST": 10,
        # "RESISC45": 10,
        "SUN397": 10,
        # "SVHN": 10,
    }

    # Parse command line arguments
    args = parse_arguments()
    args.datasets = target_datasets
    # Set default parameters
    args.lr = 1e-2  # MetaNet parameters need higher learning rate

    # Use gradient accumulation to simulate larger batch sizes
    if args.batch_size is None:
        args.batch_size = 128
    if args.num_grad_accumulation is None:
        args.num_grad_accumulation = 1
    args.print_every = 10

    args.save = f"checkpoints/{args.model}"

    # Load zero-shot accuracy
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)

    # Create directory for causal models
    if args.causal_intervention:
        save_dir = args.save + f"-causal"
    else:
        save_dir = args.save + f"-meta"
    os.makedirs(save_dir, exist_ok=True)

    # Launch distributed training
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)