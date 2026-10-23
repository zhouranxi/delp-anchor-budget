#!/bin/bash
# launch_s2r6.sh GPU_ID SHARD_ID LAM -- S2 round-6 main grid: Split-CIFAR-100
#
# Calibration (rounds a..a5) settled: GN backbone + masked CE (own 10 classes)
# + backbone-only globally-normalized Fisher anchors + grad clip 5.0;
# LAM=0.1 is the working point (0.01 too weak, 1.0 pins new-task learning).
#
# 4-GPU layout (GPUs 2/3 share memory with llama-server, ~3.5-5 GB free ->
# light arms only):
#   bash launch_s2r6.sh 0 0 0.1   # B3 (unbounded T=10 ref), then B1 T=20 grid
#   bash launch_s2r6.sh 1 1 0.1   # same, shard 1
#   bash launch_s2r6.sh 2 0 0.1   # B2: bounded policies, T=10
#   bash launch_s2r6.sh 3 1 0.1   # same, shard 1
GPU=$1; SHARD=$2; LAM=${3:-0.1}
cd /mnt/mydisk/xinguan-dendritic/bench
VP=/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$GPU" = "0" ] || [ "$GPU" = "1" ]; then
    # ---- B3: unbounded T=10 reference at LAM (10 jobs, ~15 min) ----
    $VP anchor_budget_cifar.py --task-id $SHARD --n-tasks 2 \
        --policies unbounded --budgets 5 --lams $LAM --ntasks 10 \
        --seeds 0,1,2,3,4,5,6,7,8,9 \
        --out s2r6_b_t10u.json > s2r6_b_t10u.gpu$GPU.log 2>&1
    # ---- B1: full policy grid at T=20 (heavy anchors live here) ----
    $VP anchor_budget_cifar.py --task-id $SHARD --n-tasks 2 \
        --policies unbounded,fifo,random,redun,merge,agew,agew2 \
        --budgets 2,5 --lams $LAM --ntasks 20 --seeds 0,1,2,3,4,5,6,7,8,9 \
        --out s2r6_b_t20.json > s2r6_b_t20.gpu$GPU.log 2>&1
else
    # ---- B2: bounded policies at T=10 on the llama-shared cards ----
    $VP anchor_budget_cifar.py --task-id $SHARD --n-tasks 2 \
        --policies fifo,random,redun,merge,agew,agew2 \
        --budgets 2,5 --lams $LAM --ntasks 10 --seeds 0,1,2,3,4,5,6,7,8,9 \
        --out s2r6_b_t10.json > s2r6_b_t10.gpu$GPU.log 2>&1
fi

echo "GPU$GPU shard$SHARD R6 DONE" > s2r6_gpu$GPU.done
