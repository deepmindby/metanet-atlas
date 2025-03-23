"""
Precompute and save CLIP features for datasets to accelerate training

This script precomputes features for all specified datasets using the CLIP ViT model
and saves them to disk for faster training. It uses fixed preprocessing without
random augmentation to ensure consistency.

Usage:
    python precompute_features.py --model ViT-B-32 --save-dir features
"""

import os
import torch
import argparse
from tqdm import tqdm
from src.modeling import ImageEncoder
from src.datasets.registry import get_dataset

# Define datasets to precompute features for
DATASETS = [
    "Cars", "DTD", "EuroSAT", "GTSRB",
    "MNIST", "RESISC45", "SUN397", "SVHN"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute dataset features")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="CLIP model to use (e.g. ViT-B-32)")
    parser.add_argument("--save-dir", type=str, default="precomputed_features",
                        help="Directory to save features")
    parser.add_argument("--data-location", type=str,
                        default=os.path.expanduser("/home/haichao/zby/atlas/data"),
                        help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for feature extraction")
    parser.add_argument("--openclip-cachedir", type=str,
                        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
                        help="OpenCLIP cache directory")
    return parser.parse_args()


def extract_and_save_features(model, dataset_name, save_dir, data_location, batch_size):
    """
    Extract features for a dataset and save them to disk

    Args:
        model: CLIP image encoder model
        dataset_name: Name of the dataset
        save_dir: Directory to save features
        data_location: Root directory for datasets
        batch_size: Batch size for feature extraction
    """
    print(f"Processing {dataset_name}...")

    # Use validation preprocessing (no random augmentations)
    preprocess = model.val_preprocess

    # Get datasets (both train/val and test)
    train_val_dataset = get_dataset(
        dataset_name + "Val",
        preprocess,
        location=data_location,
        batch_size=batch_size,
        num_workers=8,
    )

    test_dataset = get_dataset(
        dataset_name,
        preprocess,
        location=data_location,
        batch_size=batch_size,
        num_workers=8,
    )

    # Create save directories
    save_dir_train_val = os.path.join(save_dir, dataset_name + "Val")
    save_dir_test = os.path.join(save_dir, dataset_name)
    os.makedirs(save_dir_train_val, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    # Save classnames
    if hasattr(train_val_dataset, 'classnames'):
        with open(os.path.join(save_dir_train_val, "classnames.txt"), "w") as f:
            f.write("\n".join(train_val_dataset.classnames))
        with open(os.path.join(save_dir_test, "classnames.txt"), "w") as f:
            f.write("\n".join(test_dataset.classnames))

    # Extract features for training set
    extract_features_from_loader(model, train_val_dataset.train_loader,
                                 os.path.join(save_dir_train_val, "train_features.pt"),
                                 os.path.join(save_dir_train_val, "train_labels.pt"))

    # Extract features for validation set
    extract_features_from_loader(model, train_val_dataset.test_loader,
                                 os.path.join(save_dir_train_val, "val_features.pt"),
                                 os.path.join(save_dir_train_val, "val_labels.pt"))

    # Extract features for test set
    extract_features_from_loader(model, test_dataset.test_loader,
                                 os.path.join(save_dir_test, "test_features.pt"),
                                 os.path.join(save_dir_test, "test_labels.pt"))

    print(f"Completed processing {dataset_name}")


def extract_features_from_loader(model, loader, features_path, labels_path):
    """
    Extract features from a data loader and save them

    Args:
        model: CLIP image encoder model
        loader: DataLoader with images
        features_path: Path to save features
        labels_path: Path to save labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["images"].to(device)
                labels = batch["labels"]
            else:
                images, labels = batch
                images = images.to(device)

            # Extract features
            features = model(images)

            # Save to lists
            all_features.append(features.cpu())
            all_labels.append(labels)

    # Concatenate features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0) if all_labels[0] is not None else None

    # Save to files
    torch.save(all_features, features_path)
    if all_labels is not None:
        torch.save(all_labels, labels_path)

    print(f"Saved features to {features_path} and labels to {labels_path}")
    print(f"Feature shape: {all_features.shape}")


def main():
    args = parse_args()

    # Create a directory structure that includes the model name to avoid overwriting
    # when using different models
    model_name_safe = args.model.replace("-", "_").replace("/", "_")
    model_save_dir = os.path.join(args.save_dir, model_name_safe)

    print(f"Features will be saved to: {model_save_dir}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Save model information for reference
    with open(os.path.join(model_save_dir, "model_info.txt"), "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Feature extraction date: {torch.datetime.datetime.now()}\n")

    # Initialize model
    print(f"Initializing {args.model} model...")
    model_args = type('Args', (), {
        "model": args.model,
        "openclip_cachedir": args.openclip_cachedir,
        "cache_dir": None
    })
    image_encoder = ImageEncoder(model_args)

    # Process each dataset
    for dataset_name in DATASETS:
        extract_and_save_features(
            model=image_encoder,
            dataset_name=dataset_name,
            save_dir=model_save_dir,
            data_location=args.data_location,
            batch_size=args.batch_size
        )

    print(f"Feature precomputation complete! All features saved to {model_save_dir}")
    print("Directory structure:")
    for dataset_name in DATASETS:
        print(f"  - {model_save_dir}/{dataset_name}Val/ (training/validation features)")
        print(f"  - {model_save_dir}/{dataset_name}/ (test features)")


if __name__ == "__main__":
    main()