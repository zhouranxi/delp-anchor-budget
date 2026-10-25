#!/bin/bash
# launch_s1b.sh <GPU> <dataset> <ntasks> <out> -- S1 B: bfifo granularity x budget
# 40 jobs per shard: {neuron,branch} x M{2,5} x 10 seeds at lam=1000
cd /mnt/mydisk/xinguan-dendritic/bench
export CUDA_VISIBLE_DEVICES=$1
/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python branch_delp.py \
    --task-id 0 --n-tasks 1 \
    --ntasks $3 --lams 1000 \
    --granularities neuron,branch \
    --budgets 2,5 --skip-gran \
    --dataset $2 --out $4 \
    > ${4%.json}.gpu$1.log 2>&1
echo "$4" >> s1b_done.marker
