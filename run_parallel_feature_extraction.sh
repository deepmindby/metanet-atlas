#!/bin/bash

# 设置基本参数
SAVE_DIR="/home/haichao/zby/atlas/precomputed_features"
DATA_LOCATION="/home/haichao/zby/atlas/data"
MODEL="ViT-B-32"
BATCH_SIZE=256

# 确保保存目录存在
mkdir -p $SAVE_DIR

# 定义数据集列表
DATASETS="Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN"

# 使用GNU Parallel在8个GPU上并行处理
parallel --jobs 8 --link \
  CUDA_VISIBLE_DEVICES={1} python src/precompute_features_subset.py \
  --model $MODEL \
  --save-dir $SAVE_DIR \
  --data-location $DATA_LOCATION \
  --batch-size $BATCH_SIZE \
  --datasets {2} ::: 0 1 2 3 4 5 6 7 ::: $DATASETS

echo "所有特征计算完成！"