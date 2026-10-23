#!/bin/bash
# launch_r7.sh GPU_ID ROLE -- S2 round-7 reviewer-response arms.
#
# Layout (non-disruptive: llama keeps GPUs 2/3; light arms only there):
#   bash launch_r7.sh 0 ewc0     # plain/online EWC, MNIST, shard 0 (~60 jobs)
#   bash launch_r7.sh 1 ewc1     # same, shard 1
#   bash launch_r7.sh 2 cifar20  # CIFAR lam=0 T=20 (10 jobs), then fashion ext shard 0
#   bash launch_r7.sh 3 cifar10  # CIFAR lam=0 T=10 (10 jobs), then fashion ext shard 1
GPU=$1; ROLE=$2
set -u
cd /mnt/mydisk/xinguan-dendritic/bench
VP=/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S9=0,1,2,3,4,5,6,7,8,9

case $ROLE in
  ewc0|ewc1)
    TID=${ROLE#ewc}
    $VP ab_ewc_arms.py --task-id $TID --n-tasks 2 \
        --modes plain,online --ntasks 10,20 --lams 1000,5000,20000 \
        --seeds $S9 --out r7_ewc.json > r7_ewc.gpu$GPU.log 2>&1
    ;;
  cifar20)
    $VP anchor_budget_cifar.py --task-id 0 --n-tasks 1 \
        --policies unbounded --budgets 5 --lams 0 --ntasks 20 --seeds $S9 \
        --out r7_cifar_lam0_t20.json > r7_cifar_t20.gpu$GPU.log 2>&1
    $VP anchor_budget.py --task-id 0 --n-tasks 2 --dataset fashion \
        --policies unbounded,fifo --budgets 2,5 --lams 5000 --ntasks 10 \
        --seeds 3,4,5,6,7,8,9 --out r7_fashion.json > r7_fashion.gpu$GPU.log 2>&1
    ;;
  cifar10)
    $VP anchor_budget_cifar.py --task-id 0 --n-tasks 1 \
        --policies unbounded --budgets 5 --lams 0 --ntasks 10 --seeds $S9 \
        --out r7_cifar_lam0_t10.json > r7_cifar_t10.gpu$GPU.log 2>&1
    $VP anchor_budget.py --task-id 1 --n-tasks 2 --dataset fashion \
        --policies unbounded,fifo --budgets 2,5 --lams 5000 --ntasks 10 \
        --seeds 3,4,5,6,7,8,9 --out r7_fashion.json > r7_fashion.gpu$GPU.log 2>&1
    ;;
esac
echo "GPU$GPU $ROLE R7 DONE $(date +%H:%M:%S)" > r7_${ROLE}.done
