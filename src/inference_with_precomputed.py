"""Inference Script Using Pre-computed Features

This script performs inference using trained MetaNet models with pre-computed
CLIP features, suitable for deployment scenarios or batch processing.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from src.metanet_precomputed import PrecomputedMetaNet


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with pre-computed features")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved model checkpoint")
    parser.add_argument("--features-path", type=str, required=True,
                        help="Path to pre-computed features file")
    parser.add_argument("--output-path", type=str, default="predictions.pt",
                        help="Path to save output predictions")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for inference")
    parser.add_argument("--num-task-vectors", type=int, default=8,
                        help="Number of task vectors used in model")
    parser.add_argument("--classnames-path", type=str, default=None,
                        help="Path to class names file (optional)")
    return parser.parse_args()


def load_model(model_path, feature_dim, num_task_vectors, num_classes, device):
    """Load trained model

    Args:
        model_path: path to saved model
        feature_dim: dimension of feature vectors
        num_task_vectors: number of task vectors
        num_classes: number of output classes
        device: computation device

    Returns:
        model: loaded model
        classifier: loaded classifier
    """
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    # Create model
    model = PrecomputedMetaNet(
        feature_dim=feature_dim,
        task_vectors=num_task_vectors,
        blockwise=True
    )
    model.load_state_dict(state_dict['meta_net'])
    model = model.to(device)

    # Create classifier
    classifier = torch.nn.Linear(feature_dim, num_classes)
    classifier.load_state_dict(state_dict['classifier'])
    classifier = classifier.to(device)

    return model, classifier


def run_inference(model, classifier, features, batch_size, device):
    """Run inference on features

    Args:
        model: trained model
        classifier: classifier layer
        features: feature tensors
        batch_size: batch size for processing
        device: computation device

    Returns:
        predictions: model predictions
        probabilities: prediction probabilities
    """
    model.eval()
    classifier.eval()

    all_probs = []

    # Process in batches
    num_samples = features.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Running inference"):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_features = features[start_idx:end_idx].to(device)

            # Forward pass
            transformed_features = model(batch_features)
            outputs = classifier(transformed_features)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

    # Combine results
    all_probs = torch.cat(all_probs, dim=0)
    predictions = torch.argmax(all_probs, dim=1)

    return predictions, all_probs


def main():
    """Main inference function"""
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load features
    print(f"Loading features from {args.features_path}")
    features = torch.load(args.features_path)

    # Get feature dimension
    feature_dim = features[0].size(0) if features.dim() > 1 else features.size(1)
    print(f"Feature dimension: {feature_dim}")

    # Load class names if provided
    num_classes = 0
    class_names = None

    if args.classnames_path and os.path.exists(args.classnames_path):
        with open(args.classnames_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        num_classes = len(class_names)
        print(f"Loaded {num_classes} class names")
    else:
        # Try to infer from model
        state_dict = torch.load(args.model_path, map_location='cpu')
        if 'classifier.weight' in state_dict:
            num_classes = state_dict['classifier.weight'].size(0)
        elif 'classifier' in state_dict and 'weight' in state_dict['classifier']:
            num_classes = state_dict['classifier']['weight'].size(0)
        else:
            raise ValueError("Cannot determine number of classes. Please provide classnames file.")

        print(f"Inferred {num_classes} classes from model")

    # Load model
    print(f"Loading model from {args.model_path}")
    model, classifier = load_model(
        args.model_path,
        feature_dim,
        args.num_task_vectors,
        num_classes,
        device
    )

    # Run inference
    print("Running inference...")
    predictions, probabilities = run_inference(
        model,
        classifier,
        features,
        args.batch_size,
        device
    )

    # Save results
    results = {
        'predictions': predictions,
        'probabilities': probabilities
    }

    torch.save(results, args.output_path)
    print(f"Results saved to {args.output_path}")

    # Print some statistics
    print("\nPrediction Statistics:")
    print(f"Total samples: {len(predictions)}")

    # Class distribution
    unique_classes, counts = torch.unique(predictions, return_counts=True)
    print("\nClass Distribution:")

    for cls_idx, count in zip(unique_classes.tolist(), counts.tolist()):
        class_name = class_names[cls_idx] if class_names else f"Class {cls_idx}"
        print(f"{class_name}: {count} samples ({count / len(predictions) * 100:.2f}%)")


if __name__ == "__main__":
    main()