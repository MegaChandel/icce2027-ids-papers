#!/bin/bash
# Runs on node1 (GPU): extra seeds of the context model, then stores their per-byte NLL.
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
P=/home/ailab/anaconda3/envs/roadmasq/bin/python
while pgrep -f "mode pae" >/dev/null || pgrep -f "dump_ctx_nll.py --mode pae" >/dev/null; do sleep 30; done
$P experiments/run_deep.py --mode ctx --seeds 2 --seed-start 1 --iters 20000 --out results/deep_ctx_s12.csv > results/deep_ctx_s12.log 2>&1
for s in 1 2; do $P experiments/dump_ctx_nll.py --mode ctx --seed $s > results/dump_ctx_s$s.log 2>&1; done
echo QUEUE_DONE > results/queue_ctx.done
