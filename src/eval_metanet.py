"""MetaNet-aTLAS Model Evaluation Script

This script is used to evaluate the performance of MetaNet-aTLAS models on different datasets.
"""

import os
import json
import torch

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.metanet_composition import MetaNetImageEncoder, MetaNetLinearizedModel


def main(args):
    """Main function for evaluating MetaNet-aTLAS model

    Parameters:
    ----------
    args: argparse.Namespace
        Command line arguments
    """
    print("*" * 100)
    print(f"Evaluating MetaNet-aTLAS model ({args.finetuning_mode} mode)")
    print("*" * 100)

    # Load task vector pool
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
    ]

    # Record results
    all_results = {}

    for dataset in args.eval_datasets:
        print("-" * 100)
        print(f"Evaluating {dataset}")

        # Get task vectors, excluding the current dataset
        task_vectors = []
        for ds in pool:
            if ds == dataset:
                continue

            if args.finetuning_mode == "linear":
                pretrained_checkpoint = f"{args.save}/{ds}Val/linear_zeroshot.pt"
                finetuned_checkpoint = f"{args.save}/{ds}Val/linear_finetuned.pt"
                task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))
            else:
                pretrained_checkpoint = f"{args.save}/{ds}Val/zeroshot.pt"
                finetuned_checkpoint = f"{args.save}/{ds}Val/finetuned.pt"
                task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

        # Load model and Meta-Net
        if args.finetuning_mode == "linear":
            image_encoder = LinearizedImageEncoder(args, keep_lang=False)
            image_encoder.model = MetaNetLinearizedModel(
                image_encoder.model, task_vectors, blockwise=args.blockwise_coef
            )

            # Load trained Meta-Net
            meta_net_path = os.path.join(args.save, f"{dataset}Val", "learned_linear_metanet.pt")
            if os.path.exists(meta_net_path):
                image_encoder.model.meta_net.load_state_dict(torch.load(meta_net_path))
            else:
                print(f"Warning: Could not find {meta_net_path}, using untrained Meta-Net")

        else:
            image_encoder = ImageEncoder(args)
            image_encoder = MetaNetImageEncoder(
                image_encoder, task_vectors, blockwise=args.blockwise_coef
            )

            # Load trained Meta-Net
            meta_net_path = os.path.join(args.save, f"{dataset}Val", "learned_metanet.pt")
            if os.path.exists(meta_net_path):
                image_encoder.meta_net.load_state_dict(torch.load(meta_net_path))
            else:
                print(f"Warning: Could not find {meta_net_path}, using untrained Meta-Net")

        # Evaluate
        result = eval_single_dataset(image_encoder, dataset, args)
        all_results[dataset] = result["top1"]
        print(f"{dataset} test accuracy: {100 * result['top1']:.2f}%")

    # Calculate average accuracy
    avg_acc = sum(all_results.values()) / len(all_results)
    all_results["average"] = avg_acc
    print("-" * 100)
    print(f"Average accuracy: {100 * avg_acc:.2f}%")

    # Save results
    if args.finetuning_mode == "linear":
        results_path = os.path.join(args.save, "metanet_linear_results.json")
    else:
        results_path = os.path.join(args.save, "metanet_results.json")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    args = parse_arguments()

    # Set evaluation datasets
    if args.eval_datasets is None:
        args.eval_datasets = [
            "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"
        ]

    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"

    main(args)