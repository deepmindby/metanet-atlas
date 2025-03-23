"""Training Script Using Pre-computed Features

This script trains MetaNet models using pre-computed CLIP features,
which significantly accelerates training by avoiding the forward pass
through the CLIP image encoder.
"""

import os
import time
import json
import torch
import random
import socket
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler

from src.metanet_precomputed import PrecomputedMetaNet
from src.utils import cosine_lr
from src.args import parse_arguments
from src.datasets.registry import get_dataset
from src.datasets.common import maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp


def find_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_ddp_robust(rank, world_size, port=None, max_retries=5):
    """Setup distributed training with robust port handling

    Args:
        rank: process rank
        world_size: number of processes
        port: optional specific port to try first
        max_retries: maximum number of port retries

    Returns:
        success: whether setup succeeded
    """
    # Generate random port if not specified
    if port is None:
        port = random.randint(40000, 65000)
        print(f"Process {rank}: Generated random port {port}")

    for retry in range(max_retries):
        try:
            # Use different port for each retry
            current_port = port + retry

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(current_port)

            print(f"Process {rank}: Attempting to initialize with port {current_port}")

            torch.distributed.init_process_group(
                "nccl",
                rank=rank,
                world_size=world_size,
            )

            torch.cuda.set_device(rank)
            torch.distributed.barrier()

            print(f"Process {rank}: Successfully initialized distributed setup with port {current_port}")
            return True

        except Exception as e:
            print(f"Process {rank}: Failed on port {current_port}: {e}")
            # Wait before retrying
            time.sleep(random.uniform(1, 3))

    print(f"Process {rank}: Failed to initialize distributed setup after {max_retries} attempts")
    return False


def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    """L1 or L2 regularization term"""
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()


def plot_training_curves(train_losses, val_accuracies, dataset_name, save_dir):
    """Plot and save training curves

    Args:
        train_losses: list of loss values
        val_accuracies: list of accuracy values
        dataset_name: name of the dataset
        save_dir: directory to save plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot losses
    ax1.plot(train_losses, 'r-')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss for {dataset_name}')
    ax1.grid(True)

    # Plot accuracies
    epochs = list(range(1, len(val_accuracies) + 1))
    ax2.plot(epochs, [acc * 100 for acc in val_accuracies], 'b-o')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Validation Accuracy for {dataset_name}')
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_training_curves.png'))
    plt.close()


def evaluate_model(model, dataset, device):
    """Evaluate model on dataset

    Args:
        model: trained model
        dataset: evaluation dataset
        device: computation device

    Returns:
        accuracy: evaluation accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataset.test_loader:
            batch = maybe_dictionarize(batch)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(features)

            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


def main(rank, args):
    """Main training function

    Args:
        rank: process rank for distributed training
        args: command line arguments
    """
    args.rank = rank

    # Initialize distributed setup with robust port handling
    if not setup_ddp_robust(rank, args.world_size, args.port):
        print(f"Process {rank}: Failed to initialize distributed setup. Exiting.")
        return

    # Process all datasets specified or use defaults
    if hasattr(args, 'datasets') and args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = ["MNIST", "SUN397", "SVHN", "EuroSAT", "GTSRB", "DTD", "Cars"]

    # Ensure save directory exists
    if not hasattr(args, 'save') or args.save is None:
        args.save = "checkpoints_precomputed"
        print(f"Process {rank}: No save directory specified, using default: {args.save}")

    # Create save directory
    os.makedirs(args.save, exist_ok=True)

    # Set default feature directory if not specified
    if not hasattr(args, 'precomputed_dir') or args.precomputed_dir is None:
        args.precomputed_dir = os.path.join(args.data_location, "precomputed_features")

    # Ensure number of task vectors is set
    if not hasattr(args, 'num_task_vectors'):
        args.num_task_vectors = 8

    for dataset_name in datasets_to_process:
        target_dataset = f"precomputed_{dataset_name}Val"  # Use precomputed features
        print(f"=== Training on {dataset_name} with pre-computed features ===")

        # Setup save directory for this dataset
        save_dir = os.path.join(args.save, dataset_name + "Val")
        os.makedirs(save_dir, exist_ok=True)

        try:
            # Setup model for precomputed features
            model_name_safe = args.model.replace("-", "_").replace("/", "_")
            feature_dir = os.path.join(args.precomputed_dir, model_name_safe, dataset_name + "Val")

            # Load a sample feature to get dimensions
            sample_feature_path = os.path.join(feature_dir, "train_features.pt")
            if not os.path.exists(sample_feature_path):
                print(f"Error: Features not found at {sample_feature_path}")
                continue

            sample_features = torch.load(sample_feature_path)
            feature_dim = sample_features[0].shape[0]
            print(f"Feature dimension: {feature_dim}")

            # Create model
            model = PrecomputedMetaNet(
                feature_dim=feature_dim,
                task_vectors=args.num_task_vectors,  # Number of task vectors to simulate
                blockwise=args.blockwise_coef,
                enable_causal=args.causal_intervention if hasattr(args, 'causal_intervention') else False,
                top_k_ratio=args.top_k_ratio if hasattr(args, 'top_k_ratio') else 0.1
            )
            model = model.cuda()

            # Get data loader with precomputed features
            dataset = get_dataset(
                target_dataset,  # This should be handled by the registry to return precomputed features
                preprocess=None,  # Not needed for precomputed features
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=4
            )

            data_loader = dataset.train_loader
            num_batches = len(data_loader)

            # Set print frequency
            print_every = max(num_batches // 10, 1)

            # Distributed training setup
            ddp_loader = distribute_loader(data_loader)
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.rank],
                find_unused_parameters=True
            )

            # Setup classifier layer
            num_classes = len(dataset.classnames)
            classifier = torch.nn.Linear(feature_dim, num_classes).cuda()

            # Setup optimizer
            params = list(ddp_model.parameters()) + list(classifier.parameters())
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

            # Learning rate scheduler
            scheduler = cosine_lr(
                optimizer, args.lr, 0,
                args.epochs * num_batches
            )

            # Loss function
            loss_fn = torch.nn.CrossEntropyLoss()

            # Mixed precision training
            scaler = GradScaler()

            # Training monitoring
            train_losses = []
            epoch_losses = []
            val_accuracies = []
            best_acc = 0.0
            best_model_state = None

            # Training loop
            for epoch in range(args.epochs):
                ddp_model.train()
                classifier.train()

                epoch_loss = 0.0
                batch_count = 0

                for i, batch in enumerate(ddp_loader):
                    start_time = time.time()

                    batch = maybe_dictionarize(batch)
                    features = batch["features"].to(rank)
                    labels = batch["labels"].to(rank)

                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # Forward pass
                        transformed_features = ddp_model(features)
                        logits = classifier(transformed_features)

                        # Task loss
                        task_loss = loss_fn(logits, labels)

                        # Variance loss if causal intervention is enabled
                        if hasattr(args, 'causal_intervention') and args.causal_intervention:
                            var_loss = ddp_model.module.compute_intervention_loss(features)
                            var_penalty = args.var_penalty_coef if hasattr(args, 'var_penalty_coef') else 0.1
                            total_loss = task_loss + var_penalty * var_loss
                        else:
                            var_loss = torch.tensor(0.0, device=features.device)
                            total_loss = task_loss

                    # Backward pass
                    scaler.scale(total_loss).backward()

                    # Step optimizer
                    scheduler(i + epoch * num_batches)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Record stats
                    batch_count += 1
                    epoch_loss += task_loss.item()
                    train_losses.append(task_loss.item())

                    # Print progress
                    if i % print_every == 0 and is_main_process():
                        var_str = f", Var Loss: {var_loss.item():.6f}" if hasattr(args, 'causal_intervention') and args.causal_intervention else ""
                        print(f"Epoch {epoch+1}/{args.epochs}, Batch {i}/{num_batches}, "
                              f"Loss: {task_loss.item():.6f}{var_str}, "
                              f"Time: {time.time() - start_time:.3f}s")

                # Record epoch stats
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                epoch_losses.append(avg_epoch_loss)

                # Evaluate on validation set
                if is_main_process():
                    model.eval()
                    classifier.eval()

                    val_acc = evaluate_model(
                        model=lambda x: classifier(ddp_model.module(x)),
                        dataset=dataset,
                        device=rank
                    )
                    val_accuracies.append(val_acc)

                    print(f"Epoch {epoch+1}/{args.epochs}, "
                          f"Avg Loss: {avg_epoch_loss:.6f}, "
                          f"Val Acc: {val_acc*100:.2f}%")

                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model_state = {
                            'meta_net': ddp_model.module.state_dict(),
                            'classifier': classifier.state_dict(),
                            'epoch': epoch,
                            'acc': val_acc
                        }

            # Save results
            if is_main_process():
                # Save best model
                if best_model_state:
                    torch.save(best_model_state, os.path.join(save_dir, "best_precomputed_model.pt"))

                # Save training history
                history = {
                    'train_losses': train_losses,
                    'epoch_losses': epoch_losses,
                    'val_accuracies': val_accuracies,
                    'best_acc': best_acc
                }
                with open(os.path.join(save_dir, "precomputed_training_history.json"), 'w') as f:
                    json.dump(history, f, indent=4)

                # Plot training curves
                plot_dir = os.path.join(args.save, "precomputed_plots")
                plot_training_curves(epoch_losses, val_accuracies, dataset_name, plot_dir)

                print(f"Training completed for {dataset_name}. Best validation accuracy: {best_acc*100:.2f}%")

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    cleanup_ddp()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Set default save directory if not provided
    if args.save is None:
        args.save = "checkpoints_precomputed"
        print(f"No save directory specified, using default: {args.save}")

    # Add additional arguments for precomputed features
    args.precomputed_dir = os.path.join(args.data_location, "precomputed_features")
    args.num_task_vectors = 8  # Default number of task vectors to simulate

    # Set default training parameters if not specified
    if not hasattr(args, 'epochs') or not args.epochs:
        args.epochs = 10
    if not hasattr(args, 'batch_size') or not args.batch_size:
        args.batch_size = 256  # Can use larger batch size with precomputed features
    if not hasattr(args, 'lr') or not args.lr:
        args.lr = 1e-3
    if not hasattr(args, 'wd') or not args.wd:
        args.wd = 0.01

    # Set random port if not specified to avoid conflicts
    if not hasattr(args, 'port') or not args.port:
        args.port = random.randint(40000, 65000)
        print(f"Using randomly selected port: {args.port}")

    # Launch training
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)