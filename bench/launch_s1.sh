#!/bin/bash
# launch_s1.sh GPU_ID ROLE -- S1 (Branch-DeLP) main experiment.
#
#   bash launch_s1.sh 0 mnist      # gran ladder lam5000 T10,20 s0-9 MNIST (60 jobs)
#   bash launch_s1.sh 1 fashion    # same on Fashion-MNIST (60 jobs)
#   bash launch_s1.sh 2 refsl1000  # SI/MAS refs T10,20 + gran lam1000 T10 (70 jobs)
#   bash launch_s1.sh 3 l1000t20   # gran lam1000 T20 (30 jobs)
GPU=$1; ROLE=$2
set -u
cd /mnt/mydisk/xinguan-dendritic/bench
VP=/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S9=0,1,2,3,4,5,6,7,8,9
G=neuron,branch,plain

case $ROLE in
  mnist)
    $VP branch_delp.py --task-id 0 --n-tasks 1 --granularities $G \
        --ntasks 10,20 --lams 5000 --seeds $S9 \
        --out s1_mnist.json > s1_mnist.gpu$GPU.log 2>&1 ;;
  fashion)
    $VP branch_delp.py --task-id 0 --n-tasks 1 --granularities $G \
        --dataset fashion --ntasks 10,20 --lams 5000 --seeds $S9 \
        --out s1_fashion.json > s1_fashion.gpu$GPU.log 2>&1 ;;
  refsl1000)
    $VP branch_delp.py --task-id 0 --n-tasks 1 --granularities $G \
        --ntasks 10 --lams 1000 --seeds $S9 --with-refs \
        --out s1_l1000_t10.json > s1_l1000_t10.gpu$GPU.log 2>&1
    $VP branch_delp.py --task-id 0 --n-tasks 1 --ntasks 10,20 --seeds $S9 \
        --with-refs \
        --out s1_refs.json > s1_refs.gpu$GPU.log 2>&1 ;;
  l1000t20)
    $VP branch_delp.py --task-id 0 --n-tasks 1 --granularities $G \
        --ntasks 20 --lams 1000 --seeds $S9 \
        --out s1_l1000_t20.json > s1_l1000_t20.gpu$GPU.log 2>&1 ;;
  *) echo "unknown role $ROLE"; exit 1 ;;
esac
echo "ROLE $ROLE DONE on GPU$GPU at $(date)" >> s1_done.marker
