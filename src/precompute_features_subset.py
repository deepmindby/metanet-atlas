"""
Precompute features for a subset of datasets in parallel with robust error handling
"""

import os
import torch
import argparse
from tqdm import tqdm
import signal
import time
from datetime import datetime
import gc
import sys
from src.modeling import ImageEncoder
from src.datasets.registry import get_dataset

# Global variables for timeout handling
TIMEOUT_SECONDS = 3600  # 1 hour timeout
start_time = None
current_dataset = "unknown"

def timeout_handler(signum, frame):
    """Handle timeout signal with proper cleanup and termination"""
    global start_time, current_dataset
    elapsed = time.time() - start_time
    print(f"ERROR: Operation on dataset {current_dataset} timed out after {elapsed:.1f} seconds")
    # Force cleanup and exit
    torch.cuda.empty_cache()
    gc.collect()
    # Exit with error code to ensure process termination
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute features for dataset subset")
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
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated list of datasets to process")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout in seconds for each dataset processing")
    return parser.parse_args()

def extract_and_save_features(model, dataset_name, save_dir, data_location, batch_size):
    """
    Extract features for a dataset and save them to disk with proper cleanup
    """
    global start_time, current_dataset
    current_dataset = dataset_name
    start_time = time.time()

    print(f"Processing {dataset_name}...")

    # Set signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        # Use validation preprocessing (no random augmentations)
        preprocess = model.val_preprocess

        # Get datasets with error handling
        try:
            train_val_dataset = get_dataset(
                dataset_name + "Val",
                preprocess,
                location=data_location,
                batch_size=batch_size,
                num_workers=4,
            )
        except Exception as e:
            print(f"ERROR loading {dataset_name}Val dataset: {e}")
            raise

        try:
            test_dataset = get_dataset(
                dataset_name,
                preprocess,
                location=data_location,
                batch_size=batch_size,
                num_workers=4,
            )
        except Exception as e:
            print(f"ERROR loading {dataset_name} test dataset: {e}")
            raise

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

    except Exception as e:
        print(f"ERROR processing dataset {dataset_name}: {e}")
        # Don't exit here, cleanup in finally block
    finally:
        # Cancel timeout alarm
        signal.alarm(0)
        # Force clean GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU resources released after processing {dataset_name}")
        # Flush output to ensure logs are displayed
        sys.stdout.flush()

def extract_features_from_loader(model, loader, features_path, labels_path):
    """
    Extract features from a data loader and save them with robust error handling
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    try:
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

                # Release memory after each batch
                torch.cuda.empty_cache()

        # Concatenate features and labels
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0) if all_labels[0] is not None else None

        # Save to files
        torch.save(all_features, features_path)
        if all_labels is not None:
            torch.save(all_labels, labels_path)

        print(f"Saved features to {features_path} and labels to {labels_path}")
        print(f"Feature shape: {all_features.shape}")

    except Exception as e:
        print(f"ERROR extracting features: {e}")
    finally:
        # Ensure memory is released
        all_features = None
        all_labels = None
        torch.cuda.empty_cache()
        gc.collect()

def main():
    args = parse_args()

    # Set global timeout value
    global TIMEOUT_SECONDS
    TIMEOUT_SECONDS = args.timeout

    # Parse datasets to process
    datasets = [d.strip() for d in args.datasets.split(",")]

    # Create directory with model name
    model_name_safe = args.model.replace("-", "_").replace("/", "_")
    model_save_dir = os.path.join(args.save_dir, model_name_safe)

    print(f"Features will be saved to: {model_save_dir}")
    os.makedirs(model_save_dir, exist_ok=True)

    # Log execution information
    try:
        # Save model info
        info_file = os.path.join(model_save_dir, "model_info.txt")
        if not os.path.exists(info_file):
            with open(info_file, "w") as f:
                f.write(f"Model: {args.model}\n")
                f.write(f"Feature extraction date: {datetime.now()}\n")
                f.write(f"Datasets: {', '.join(datasets)}\n")

        # Initialize model
        print(f"Initializing {args.model} model...")
        model_args = type('Args', (), {
            "model": args.model,
            "openclip_cachedir": args.openclip_cachedir,
            "cache_dir": None
        })

        image_encoder = ImageEncoder(model_args)

        # Process each dataset
        for dataset_name in datasets:
            try:
                extract_and_save_features(
                    model=image_encoder,
                    dataset_name=dataset_name,
                    save_dir=model_save_dir,
                    data_location=args.data_location,
                    batch_size=args.batch_size
                )
            except Exception as e:
                print(f"ERROR processing dataset {dataset_name}: {e}")
                # Continue to next dataset
                continue

        print(f"Feature computation complete for datasets: {datasets}")

    except Exception as e:
        print(f"ERROR in main process: {e}")
    finally:
        # Ensure resources are released
        print("Cleaning up resources...")
        torch.cuda.empty_cache()
        gc.collect()
        print("Done.")
        # Force exit to ensure the process terminates
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        torch.cuda.empty_cache()
        sys.exit(1)