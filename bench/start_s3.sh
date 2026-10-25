#!/bin/bash
# S3: 任务相关性梯度(E1, 2 shards GPU0/1) + T20 剂量扫全(E2, 4 shards GPU2/3/0/1 串行复用)
# E2 用 branch_delp.py --own-bias {1,3} --ntasks 20, 分片 i%4
set -e
cd ~/bench
PY=/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python
rm -f s3_done.marker

# ---- E1: s3corr (2 shards, parallel on GPU0/1) ----
CUDA_VISIBLE_DEVICES=0 $PY s3_corr.py --shard 0 --n-shards 2 --out s3corr_mnist.json.part0 \
  > s3corr.gpu0.log 2>&1 &
E1a=$!
CUDA_VISIBLE_DEVICES=1 $PY s3_corr.py --shard 1 --n-shards 2 --out s3corr_mnist.json.part1 \
  > s3corr.gpu1.log 2>&1 &
E1b=$!

# ---- E2: T20 dose sweep (branch_delp.py --own-bias, 4 shards) ----
# 顺序串行复用：b=1 两数据集(GPU2,3), 等 E1 完成后 b=3 两数据集(GPU0,1)
CUDA_VISIBLE_DEVICES=2 $PY branch_delp.py --task-id 0 --n-tasks 4 \
  --ntasks 20 --lams 1000 --granularities neuron --seeds 0,1,2,3,4,5,6,7,8,9 \
  --own-bias 1.0 --dataset mnist --out c2_mnist_b1.json \
  > c2_t20b1_mnist.gpu2.log 2>&1 &
E2a=$!
CUDA_VISIBLE_DEVICES=3 $PY branch_delp.py --task-id 1 --n-tasks 4 \
  --ntasks 20 --lams 1000 --granularities neuron --seeds 0,1,2,3,4,5,6,7,8,9 \
  --own-bias 1.0 --dataset fashion --out c2_fashion_b1.json \
  > c2_t20b1_fashion.gpu3.log 2>&1 &
E2b=$!

wait $E1a $E1b $E2a $E2b
echo E1_E2_BATCH1_DONE >> s3_done.marker

# ---- E2b: b=3 (GPU0/1 空闲后) ----
CUDA_VISIBLE_DEVICES=0 $PY branch_delp.py --task-id 2 --n-tasks 4 \
  --ntasks 20 --lams 1000 --granularities neuron --seeds 0,1,2,3,4,5,6,7,8,9 \
  --own-bias 3.0 --dataset mnist --out c2_mnist_b3.json \
  > c2_t20b3_mnist.gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY branch_delp.py --task-id 3 --n-tasks 4 \
  --ntasks 20 --lams 1000 --granularities neuron --seeds 0,1,2,3,4,5,6,7,8,9 \
  --own-bias 3.0 --dataset fashion --out c2_fashion_b3.json \
  > c2_t20b3_fashion.gpu1.log 2>&1 &
wait
echo S3_ALL_DONE >> s3_done.marker
