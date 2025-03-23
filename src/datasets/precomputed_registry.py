"""Registry for Pre-computed Feature Datasets

This module extends the dataset registry to support datasets with pre-computed features.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader


class PrecomputedFeatureDataset(Dataset):
    """Dataset for pre-computed features"""

    def __init__(self, features_path, labels_path):
        """Initialize dataset with paths to pre-computed features and labels

        Args:
            features_path: Path to pre-computed features tensor
            labels_path: Path to labels tensor
        """
        super().__init__()

        # Load features and labels
        try:
            self.features = torch.load(features_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load features from {features_path}: {e}")

        try:
            self.labels = torch.load(labels_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load labels from {labels_path}: {e}")

        # Verify dimensions
        if len(self.features) != len(self.labels):
            raise ValueError(f"Features ({len(self.features)}) and labels ({len(self.labels)}) count mismatch")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "index": idx
        }


class PrecomputedFeatures:
    """Dataset container class for pre-computed features"""

    def __init__(self,
                 feature_dir,
                 preprocess=None,  # Not used but kept for API compatibility
                 location=None,  # Not used but kept for API compatibility
                 batch_size=128,
                 num_workers=8):
        """Initialize with directory containing pre-computed features

        Args:
            feature_dir: Path to directory with pre-computed features
            preprocess: Unused, kept for API compatibility
            location: Unused, kept for API compatibility
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        # Verify paths
        train_features_path = os.path.join(feature_dir, "train_features.pt")
        train_labels_path = os.path.join(feature_dir, "train_labels.pt")
        val_features_path = os.path.join(feature_dir, "val_features.pt")
        val_labels_path = os.path.join(feature_dir, "val_labels.pt")

        if not os.path.exists(train_features_path):
            raise FileNotFoundError(f"Train features not found at {train_features_path}")

        # Create train dataset and loader
        self.train_dataset = PrecomputedFeatureDataset(
            train_features_path,
            train_labels_path
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # Create test dataset and loader
        self.test_dataset = PrecomputedFeatureDataset(
            val_features_path if os.path.exists(val_features_path) else train_features_path,
            val_labels_path if os.path.exists(val_labels_path) else train_labels_path
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # Load classnames
        classnames_path = os.path.join(feature_dir, "classnames.txt")
        if os.path.exists(classnames_path):
            with open(classnames_path, "r") as f:
                self.classnames = [line.strip() for line in f.readlines()]
        else:
            # Create dummy classnames
            unique_labels = torch.unique(self.train_dataset.labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]


def get_precomputed_dataset(dataset_name, model_name, location, batch_size=128, num_workers=8):
    """Get dataset with pre-computed features

    Args:
        dataset_name: Name of the dataset (without 'precomputed_' prefix)
        model_name: Name of the model used for feature extraction
        location: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker threads

    Returns:
        dataset: Dataset with pre-computed features
    """
    # Clean dataset name if it has "precomputed_" prefix
    if dataset_name.startswith("precomputed_"):
        dataset_name = dataset_name[len("precomputed_"):]

    # Normalize model name for directory structure
    model_name_safe = model_name.replace("-", "_").replace("/", "_")

    # Build feature directory path
    feature_dir = os.path.join(
        location,
        "precomputed_features",
        model_name_safe,
        dataset_name
    )

    # Check if features exist
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Pre-computed features not found at {feature_dir}")

    # Create and return dataset
    return PrecomputedFeatures(
        feature_dir=feature_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )