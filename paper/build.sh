#!/bin/bash
# Regenerate tables/figures inputs and compile the paper. Usage: paper/build.sh
set -e
cd "$(dirname "$0")"
mkdir -p gen
cp ../results/tables/numbers.tex ../results/tables/paper_table_*.tex gen/
cp ../figures/fig_*.pdf gen/
/home/ailab/anaconda3/envs/texenv/bin/tectonic -X compile main.tex 2>&1 | grep -vE "^note: (downloading|Skipped)" || true
pdfinfo main.pdf | grep -E "Pages|Page size"
