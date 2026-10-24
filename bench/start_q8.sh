#!/bin/bash
cd /mnt/mydisk/xinguan-dendritic/bench
PY=/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python
rm -f q8_done.marker
# exp1: signed drift (GPU0, 6 runs sequential)
CUDA_VISIBLE_DEVICES=0 $PY signed_drift.py --out signeddrift.json > q8_sigdrift.log 2>&1 &
# exp2: learned bias, 3-way shard (GPU1-3), mnist then fashion
for ds in mnist fashion; do
  for g in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $PY branch_delp.py --task-id $g --n-tasks 3 \
      --ntasks 10 --lams 1000 --granularities neuron,branch --seeds 0,1,2,3,4,5,6,7,8,9 \
      --bias-mode learned --dataset $ds --out q8learned_${ds}.json \
      > q8learned_${ds}.gpu$g.log 2>&1 &
  done
  wait
done
echo LEARNED_DONE >> q8_done.marker
wait
echo ALL_DONE >> q8_done.marker
