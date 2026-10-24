#!/bin/bash
# launch_r8.sh GPU_ID ROLE -- S2 round-8: strengthen-this-draft arms.
#
#   bash launch_r8.sh 0 gamma05  # online EWC gamma=0.5 sweep (60 jobs)
#   bash launch_r8.sh 1 gamma09  # online EWC gamma=0.9 sweep (60 jobs)
#   bash launch_r8.sh 2 mgrid    # fifo M dose-response T=20 (57 jobs)
#   bash launch_r8.sh 3 merge    # merge instrumentation rerun (40 jobs)
GPU=$1; ROLE=$2
set -u
cd /mnt/mydisk/xinguan-dendritic/bench
VP=/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S9=0,1,2,3,4,5,6,7,8,9

case $ROLE in
  gamma05)
    $VP ab_ewc_arms.py --task-id 0 --n-tasks 1 --modes online --gamma 0.5 \
        --ntasks 10,20 --lams 1000,5000,20000 --seeds $S9 \
        --out r8_gamma05.json > r8_gamma05.gpu$GPU.log 2>&1
    ;;
  gamma09)
    $VP ab_ewc_arms.py --task-id 0 --n-tasks 1 --modes online --gamma 0.9 \
        --ntasks 10,20 --lams 1000,5000,20000 --seeds $S9 \
        --out r8_gamma09.json > r8_gamma09.gpu$GPU.log 2>&1
    ;;
  mgrid)
    $VP anchor_budget.py --task-id 0 --n-tasks 1 --dataset mnist \
        --policies fifo --budgets 3,4,6,8,10 --lams 5000 --ntasks 20 \
        --seeds $S9 --out r8_mgrid.json > r8_mgrid.gpu$GPU.log 2>&1
    $VP anchor_budget.py --task-id 0 --n-tasks 1 --dataset mnist \
        --policies fifo --budgets 2 --lams 5000 --ntasks 20 \
        --seeds 3,4,5,6,7,8,9 --out r8_m2topup.json > r8_m2topup.gpu$GPU.log 2>&1
    ;;
  merge)
    $VP anchor_budget.py --task-id 0 --n-tasks 1 --dataset mnist \
        --policies merge --budgets 2,5 --lams 5000 --ntasks 10,20 \
        --seeds $S9 --out r8_merge.json > r8_merge.gpu$GPU.log 2>&1
    ;;
esac
echo "GPU$GPU $ROLE R8 DONE $(date +%H:%M:%S)" > r8_${ROLE}.done
