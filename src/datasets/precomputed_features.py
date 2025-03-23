"""
Dataset classes for precomputed features

This module provides dataset classes for working with precomputed CLIP features,
eliminating the need to run the ViT encoder during training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


class PrecomputedFeatureDataset(Dataset):
    """Dataset for precomputed features"""

    def __init__(self, features_path, labels_path):
        """
        Initialize dataset with paths to precomputed features and labels

        Args:
            features_path: Path to precomputed features tensor
            labels_path: Path to labels tensor
        """
        super().__init__()
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)

        assert len(self.features) == len(self.labels), \
            f"Features ({len(self.features)}) and labels ({len(self.labels)}) count mismatch"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx]
        }


class PrecomputedFeatures:
    """Dataset container class for precomputed features"""

    def __init__(self,
                 feature_dir,
                 preprocess=None,  # Not used but kept for API compatibility
                 location=None,  # Not used but kept for API compatibility
                 batch_size=128,
                 num_workers=8):
        """
        Initialize with directory containing precomputed features

        Args:
            feature_dir: Path to directory with precomputed features
            preprocess: Unused, kept for API compatibility
            location: Unused, kept for API compatibility
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        self.train_dataset = PrecomputedFeatureDataset(
            os.path.join(feature_dir, "train_features.pt"),
            os.path.join(feature_dir, "train_labels.pt")
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = PrecomputedFeatureDataset(
            os.path.join(feature_dir, "val_features.pt"),
            os.path.join(feature_dir, "val_labels.pt")
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # Load classnames if available
        classnames_path = os.path.join(feature_dir, "classnames.txt")
        if os.path.exists(classnames_path):
            with open(classnames_path, "r") as f:
                self.classnames = [line.strip() for line in f.readlines()]
        else:
            # Create dummy classnames if file doesn't exist
            unique_labels = torch.unique(self.train_dataset.labels)
            self.classnames = [f"class_{i}" for i in range(len(unique_labels))]