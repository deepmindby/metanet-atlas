"""对比aTLAS和MetaNet-aTLAS的性能

此脚本比较原始aTLAS和基于Meta-Net的改进版MetaNet-aTLAS在相同数据集上的性能。
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main(ckpt_dir, model_name, datasets=None):
    """主函数，对比aTLAS和MetaNet-aTLAS性能

    参数:
    ----------
    ckpt_dir: str
        检查点目录
    model_name: str
        模型名称，如"ViT-B-32"
    datasets: List[str], optional
        要比较的数据集列表，默认为None（比较所有数据集）
    """
    save_dir = os.path.join(ckpt_dir, model_name)

    # 加载结果
    atlas_path = os.path.join(save_dir, "learned_additions.json")
    meta_atlas_path = os.path.join(save_dir, "learned_metanet.json")

    if not os.path.exists(atlas_path):
        print(f"错误：找不到aTLAS结果文件 {atlas_path}")
        return

    if not os.path.exists(meta_atlas_path):
        print(f"错误：找不到MetaNet-aTLAS结果文件 {meta_atlas_path}")
        return

    with open(atlas_path, "r") as f:
        atlas_results = json.load(f)

    with open(meta_atlas_path, "r") as f:
        meta_atlas_results = json.load(f)

    # 获取数据集列表
    if datasets is None:
        datasets = [k for k in meta_atlas_results.keys() if not k.endswith("Val") and k != "average"]

    # 提取测试结果
    atlas_test_accs = {}
    for dataset in datasets:
        if dataset in atlas_results.get("test", {}):
            atlas_test_accs[dataset] = atlas_results["test"].get(f"{dataset}:top1", 0)

    meta_test_accs = {k: v for k, v in meta_atlas_results.items() if k in datasets}

    # 计算平均准确率
    atlas_avg = np.mean(list(atlas_test_accs.values()))
    meta_avg = np.mean(list(meta_test_accs.values()))

    # 打印结果
    print("=" * 100)
    print(f"模型: {model_name}")
    print("=" * 100)
    print(f"{'数据集':<15} {'aTLAS':<10} {'MetaNet-aTLAS':<15} {'提升':<10}")
    print("-" * 60)

    for dataset in datasets:
        atlas_acc = atlas_test_accs.get(dataset, 0) * 100
        meta_acc = meta_test_accs.get(dataset, 0) * 100
        improvement = meta_acc - atlas_acc

        print(f"{dataset:<15} {atlas_acc:<10.2f}% {meta_acc:<15.2f}% {improvement:<+10.2f}%")

    print("-" * 60)
    print(f"{'平均':<15} {atlas_avg * 100:<10.2f}% {meta_avg * 100:<15.2f}% {(meta_avg - atlas_avg) * 100:<+10.2f}%")

    # 绘制对比图
    plt.figure(figsize=(12, 8))

    x = np.arange(len(datasets))
    width = 0.35

    atlas_accs = [atlas_test_accs.get(d, 0) * 100 for d in datasets]
    meta_accs = [meta_test_accs.get(d, 0) * 100 for d in datasets]

    plt.bar(x - width / 2, atlas_accs, width, label='aTLAS')
    plt.bar(x + width / 2, meta_accs, width, label='MetaNet-aTLAS')

    plt.xlabel('数据集')
    plt.ylabel('准确率 (%)')
    plt.title(f'aTLAS vs MetaNet-aTLAS ({model_name})')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "atlas_vs_metanet.png"))
    plt.close()

    print(f"对比图已保存到 {os.path.join(save_dir, 'atlas_vs_metanet.png')}")


if __name__ == "__main__":
    ckpt_dir = "checkpoints"  # 或 "checkpoints_{args.seed}"
    model_name = "ViT-B-32"  # 或其他模型

    datasets = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"
    ]

    main(ckpt_dir, model_name, datasets)