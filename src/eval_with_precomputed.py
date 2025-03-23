"""Evaluation Script Using Pre-computed Features

This script evaluates MetaNet models using pre-computed CLIP features,
which significantly accelerates evaluation by avoiding the forward pass
through the CLIP image encoder.
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from src.metanet_precomputed import PrecomputedMetaNet
from src.datasets.registry import get_dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate models with pre-computed features")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model used for feature extraction")
    parser.add_argument("--data-location", type=str,
                        default=os.path.expanduser("/home/haichao/zby/atlas/data"),
                        help="Root directory for datasets")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--model-dir", type=str, default="checkpoints",
                        help="Directory with trained models")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads for data loading")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["MNIST", "SUN397", "SVHN", "EuroSAT", "GTSRB", "DTD", "Cars"],
                        help="Datasets to evaluate")
    return parser.parse_args()


def evaluate_model(model_path, dataset, device):
    """Evaluate model on dataset

    Args:
        model_path: path to saved model
        dataset: evaluation dataset
        device: computation device

    Returns:
        dict: evaluation metrics
    """
    # Load model state
    state_dict = torch.load(model_path, map_location=device)

    # Get feature dimension
    sample_batch = next(iter(dataset.test_loader))
    sample_batch = sample_batch if not isinstance(sample_batch, dict) else sample_batch
    feature_dim = sample_batch["features"].shape[1]

    # Create model
    num_task_vectors = 8  # Default, should match training
    model = PrecomputedMetaNet(
        feature_dim=feature_dim,
        task_vectors=num_task_vectors,
        blockwise=True
    )

    # Load saved state
    model.load_state_dict(state_dict['meta_net'])
    model = model.to(device)

    # Create classifier
    num_classes = len(dataset.classnames)
    classifier = torch.nn.Linear(feature_dim, num_classes)
    classifier.load_state_dict(state_dict['classifier'])
    classifier = classifier.to(device)

    # Evaluation
    model.eval()
    classifier.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataset.test_loader, desc="Evaluating"):
            batch = batch if not isinstance(batch, dict) else batch
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            transformed_features = model(features)
            outputs = classifier(transformed_features)

            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total

    # Calculate per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    per_class_acc = {}

    for cls_idx in range(len(dataset.classnames)):
        cls_mask = (all_labels == cls_idx)
        if np.sum(cls_mask) > 0:
            cls_acc = np.mean(all_preds[cls_mask] == cls_idx)
            per_class_acc[dataset.classnames[cls_idx]] = float(cls_acc)

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'num_samples': total
    }


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model name for paths
    model_name_safe = args.model.replace("-", "_").replace("/", "_")

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Overall results
    all_results = {}

    for dataset_name in args.datasets:
        print(f"=== Evaluating on {dataset_name} ===")

        # Get dataset with precomputed features
        try:
            dataset = get_dataset(
                f"precomputed_{dataset_name}",
                preprocess=None,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        # Find model path
        model_path = os.path.join(
            args.model_dir,
            args.model,
            f"{dataset_name}Val",
            "best_precomputed_model.pt"
        )

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, skipping.")
            continue

        try:
            # Evaluate model
            results = evaluate_model(model_path, dataset, device)

            # Print results
            print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
            print(f"Number of samples: {results['num_samples']}")

            # Store results
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error evaluating model for {dataset_name}: {e}")

    # Save all results
    results_path = os.path.join(args.save_dir, f"precomputed_evaluation_{model_name_safe}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"All evaluation results saved to {results_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Dataset':<15} {'Accuracy':<10}")
    print("-" * 25)

    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<15} {results['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()