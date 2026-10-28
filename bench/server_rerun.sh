#!/usr/bin/env bash
# Server re-run driver for 10.10.52.210 (64 CPU cores, 4x RTX 3090 of which
# GPUs 2/3 belong to other users and must never be touched).
#
# The models are small (width 256) and the bottleneck is the CPU BLAS, so
# this runs CPU-parallel: each worker is capped at 8 threads and several
# workers run concurrently.  A single process would otherwise serialise on
# one core per phase.
#
# Usage:  bash server_rerun.sh <stage>
#   stage 1 : reproduce-check
#   stage 2 : fill-in
#   stage 3 : extended seeds
set -u

REMOTE_DIR=/mnt/mydisk/xinguan-dendritic
BENCH=$REMOTE_DIR/bench
PY=$REMOTE_DIR/.venv/bin/python
OUT=$BENCH/results_s5
LOG=/tmp/rerun_logs
mkdir -p "$OUT" "$LOG"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""      # CPU-only torch in this venv

launch() {
  local name=$1; shift
  ( "$@" > "$LOG/$name.log" 2>&1; echo "DONE $name rc=$?" >> "$LOG/_status.log" ) &
  echo "  launched $name (pid $!)"
}

case "${1:-1}" in

1)
  echo "=== stage 1: reproduce-check ==="
  cd "$BENCH" || exit 1
  : > "$LOG/_status.log"

  launch s4_diag_mnist_s0   "$PY" s4_diag.py --ds mnist   --seed 0
  launch s4_diag_fashion_s0 "$PY" s4_diag.py --ds fashion --seed 0
  launch signed_drift       "$PY" signed_drift.py --out "$OUT/server_signeddrift.json"
  launch gate_attribution   "$PY" s5_gate_attribution.py \
        --datasets mnist,fashion --seeds 0,1,2 \
        --out "$OUT/server_gate_attribution.json"
  launch flip_attribution   "$PY" s5_flip_attribution.py \
        --datasets mnist,fashion --seeds 0,1,2 \
        --out "$OUT/server_flip_attribution.json"
  ;;

2)
  echo "=== stage 2: fill-in ==="
  cd "$BENCH" || exit 1
  : > "$LOG/_status.log"

  launch s4_ctxnoise      "$PY" s4_ctxnoise.py --out "$OUT/server_s4ctx.json"
  launch s4_ablation      "$PY" s4_ablation_gate.py --out "$OUT/server_s4abl.json"
  launch gate_attr_s5     "$PY" s5_gate_attribution.py \
        --datasets mnist,fashion --seeds 0,1,2,3,4 \
        --out "$OUT/server_gate_attribution_s5.json"
  launch flip_attr_s5     "$PY" s5_flip_attribution.py \
        --datasets mnist,fashion --seeds 0,1,2,3,4 \
        --out "$OUT/server_flip_attribution_s5.json"
  ;;

3)
  echo "=== stage 3: extended seeds ==="
  cd "$BENCH" || exit 1
  : > "$LOG/_status.log"

  for s in 0 1 2 3 4 5; do
    launch "s4_diag_mnist_s$s"  "$PY" s4_diag.py --ds mnist   --seed $s
    launch "s4_diag_fashion_s$s" "$PY" s4_diag.py --ds fashion --seed $s
  done
  ;;

*)
  echo "unknown stage: $1"; exit 2;;
esac

echo
echo "launched. monitor:"
echo "  cat $LOG/_status.log"
echo "  tail -f $LOG/*.log"
