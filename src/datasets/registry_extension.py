"""Extension to Dataset Registry to Support Pre-computed Features

This module extends the existing dataset registry to handle pre-computed features.
It detects when a dataset name starts with "precomputed_" and loads the appropriate
pre-computed features instead of processing images.
"""

import os
import sys
from src.datasets.precomputed_registry import get_precomputed_dataset


def get_dataset_with_precomputed(original_get_dataset):
    """Create a wrapper around the original get_dataset function to support pre-computed features

    Args:
        original_get_dataset: Original get_dataset function

    Returns:
        wrapped function that handles pre-computed features
    """

    def wrapped_get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16,
                            val_fraction=0.1, max_val_samples=5000):
        """Get dataset, with support for pre-computed features

        Args:
            dataset_name: Name of the dataset
            preprocess: Preprocessing function (ignored for pre-computed features)
            location: Root data directory
            batch_size: Batch size for dataloaders
            num_workers: Number of worker threads
            val_fraction: Fraction of data to use for validation
            max_val_samples: Maximum number of validation samples

        Returns:
            dataset: Dataset object
        """
        # Check if requesting pre-computed features
        if dataset_name.startswith('precomputed_'):
            # Extract model name, default to ViT-B-32 if not specified
            if '@' in dataset_name:
                prefix, model_name = dataset_name.split('@')
                actual_dataset_name = prefix.replace('precomputed_', '')
            else:
                model_name = "ViT-B-32"
                actual_dataset_name = dataset_name.replace('precomputed_', '')

            print(f"Loading pre-computed features for {actual_dataset_name} with model {model_name}")

            # Get pre-computed dataset
            try:
                return get_precomputed_dataset(
                    dataset_name=actual_dataset_name,
                    model_name=model_name,
                    location=location,
                    batch_size=batch_size,
                    num_workers=num_workers
                )
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print(f"Falling back to processing images for {actual_dataset_name}")
                # Fall back to original method if pre-computed features not found
                return original_get_dataset(
                    actual_dataset_name, preprocess, location,
                    batch_size, num_workers, val_fraction, max_val_samples
                )

        # For regular datasets, use the original function
        return original_get_dataset(
            dataset_name, preprocess, location,
            batch_size, num_workers, val_fraction, max_val_samples
        )

    return wrapped_get_dataset