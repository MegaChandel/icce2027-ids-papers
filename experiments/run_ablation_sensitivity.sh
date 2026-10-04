#!/bin/bash
# Experiments (c) and (d) for Project 2, reusing run_matrix.py.
set -e
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
PY=/home/ailab/anaconda3/envs/mavx/bin/python
# (c) timing representation: IDs only vs + raw log inter-arrival vs + per-campaign z-scored
$PY experiments/run_matrix.py --seeds 5 --vocab train --models M1_LSTM M1_LSTM+dt M1_LSTM+dtz \
    --out results/ablation_timing.csv
# (d) window length
for L in 16 64 128; do
  $PY experiments/run_matrix.py --seeds 3 --vocab train --L $L --stride $((L/2)) \
      --models M1_LSTM M4_NGRAM R1_RATE R2_HIST --out results/sens_L$L.csv
done
echo ALLDONE
