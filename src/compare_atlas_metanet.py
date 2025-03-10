"""Comparison of aTLAS and MetaNet-aTLAS Performance

This script compares the performance of the original aTLAS and the improved
MetaNet-aTLAS models on the same datasets.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(ckpt_dir, model_name, finetuning_mode="standard", datasets=None):
    """Main function to compare aTLAS and MetaNet-aTLAS performance

    Args:
        ckpt_dir: Checkpoint directory path
        model_name: Model name (e.g., "ViT-B-32")
        finetuning_mode: Finetuning mode ("standard" or "linear")
        datasets: List of datasets to compare, defaults to all datasets
    """
    save_dir = os.path.join(ckpt_dir, model_name)

    # Load results based on finetuning mode
    if finetuning_mode == "linear":
        atlas_path = os.path.join(save_dir, "learned_linear_additions.json")
        meta_atlas_path = os.path.join(save_dir, "metanet_linear_results.json")
    else:
        # Default to standard finetuning mode
        atlas_path = os.path.join(save_dir, "learned_additions_std_aniso.json")
        meta_atlas_path = os.path.join(save_dir, "metanet_results.json")

    if not os.path.exists(atlas_path):
        print(f"Error: aTLAS results file not found {atlas_path}")
        return

    if not os.path.exists(meta_atlas_path):
        print(f"Error: MetaNet-aTLAS results file not found {meta_atlas_path}")
        return

    with open(atlas_path, "r") as f:
        atlas_results = json.load(f)

    with open(meta_atlas_path, "r") as f:
        meta_atlas_results = json.load(f)

    if datasets is None:
        datasets = [k for k in meta_atlas_results.keys() if not k.endswith("Val") and k != "average"]

    atlas_test_accs = {}
    for dataset in datasets:
        if "test" in atlas_results and dataset in atlas_results["test"]:
            atlas_test_accs[dataset] = atlas_results["test"].get(f"{dataset}:top1", 0)
        elif "test" in atlas_results and f"{dataset}:top1" in atlas_results["test"]:
            atlas_test_accs[dataset] = atlas_results["test"][f"{dataset}:top1"]
        elif dataset in atlas_results:
            atlas_test_accs[dataset] = atlas_results[dataset]

    meta_test_accs = {k: v for k, v in meta_atlas_results.items() if k in datasets}

    atlas_values = list(atlas_test_accs.values())
    meta_values = list(meta_test_accs.values())
    atlas_avg = np.mean(atlas_values) if atlas_values else 0
    meta_avg = np.mean(meta_values) if meta_values else 0

    print("=" * 100)
    print(f"Model: {model_name}, Finetuning Mode: {finetuning_mode}")
    print("=" * 100)
    print(f"{'Dataset':<15} {'aTLAS':<15} {'MetaNet-aTLAS':<15} {'Improvement':<10}")
    print("-" * 60)

    for dataset in datasets:
        atlas_acc = atlas_test_accs.get(dataset, 0) * 100
        meta_acc = meta_test_accs.get(dataset, 0) * 100
        improvement = meta_acc - atlas_acc

        atlas_str = f"{atlas_acc:.2f}%"
        meta_str = f"{meta_acc:.2f}%"
        imp_str = f"{improvement:+.2f}%"

        print(f"{dataset:<15} {atlas_str:<15} {meta_str:<15} {imp_str:<10}")

    print("-" * 60)

    atlas_avg_str = f"{atlas_avg * 100:.2f}%"
    meta_avg_str = f"{meta_avg * 100:.2f}%"
    avg_imp_str = f"{(meta_avg - atlas_avg) * 100:+.2f}%"

    print(f"{'Average':<15} {atlas_avg_str:<15} {meta_avg_str:<15} {avg_imp_str:<10}")

    plt.figure(figsize=(12, 8))

    x = np.arange(len(datasets))
    width = 0.35

    atlas_accs = [atlas_test_accs.get(d, 0) * 100 for d in datasets]
    meta_accs = [meta_test_accs.get(d, 0) * 100 for d in datasets]

    plt.bar(x - width / 2, atlas_accs, width, label='aTLAS')
    plt.bar(x + width / 2, meta_accs, width, label='MetaNet-aTLAS')

    plt.xlabel('Datasets')
    plt.ylabel('Accuracy (%)')
    plt.title(f'aTLAS vs MetaNet-aTLAS ({model_name}, {finetuning_mode} mode)')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"atlas_vs_metanet_{finetuning_mode}.png"))
    plt.close()

    print(f"Comparison chart saved to {os.path.join(save_dir, f'atlas_vs_metanet_{finetuning_mode}.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare aTLAS and MetaNet-aTLAS performance")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="Model name")
    parser.add_argument("--finetuning-mode", type=str, default="standard", choices=["standard", "linear"],
                        help="Finetuning mode (standard or linear)")
    parser.add_argument("--datasets", nargs="+", default=None, help="List of datasets to compare")

    args = parser.parse_args()

    datasets = args.datasets
    if datasets is None:
        datasets = [
            "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"
        ]

    main(args.ckpt_dir, args.model_name, args.finetuning_mode, datasets)