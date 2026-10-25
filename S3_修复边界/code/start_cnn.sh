#!/bin/bash
# S5/CNN: 160 runs on 2 GPUs (GPU2/3 occupied by others).
# GPU0: mnist T10 -> mnist T20 ; GPU1: fashion T10 -> fashion T20.
cd ~/bench
PY=/mnt/mydisk/xinguan-dendritic/dendritic-lab/venv-gpu/bin/python
rm -f cnnc2_done.marker

run_all () {  # gpu ds sfx T
  local gpu=$1 ds=$2 T=$3 sfx=$4
  for b in 0 1 3 10; do
    CUDA_VISIBLE_DEVICES=$gpu $PY cnn_delp.py --dataset $ds --ntask $T \
      --bias ${b}.0 --out ${sfx}_${ds}_b${b}.json \
      >> cnnc2_${sfx}_${ds}.gpu${gpu}.log 2>&1
  done
}

( run_all 0 mnist   10 cnnc2;    run_all 0 mnist   20 cnnc2t20 ) &
P0=$!
( run_all 1 fashion 10 cnnc2;    run_all 1 fashion 20 cnnc2t20 ) &
P1=$!
wait $P0 $P1
echo CNNC2_ALL_DONE > cnnc2_done.marker
