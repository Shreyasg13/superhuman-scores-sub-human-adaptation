#!/usr/bin/env bash
# Reproduces every number and figure in the paper. Runtime: about 1-2 minutes.
set -e
cd "$(dirname "$0")/code"
mkdir -p fig
echo "1/4 Composite indices, z-scores, IQ*, landscape shares, projections";  python3 calc.py  | tee ../results_calc.txt
echo "2/4 Monte Carlo (weights, score noise, growth)";                        python3 mc.py    | tee ../results_mc.txt
echo "3/4 Figures 1-9";   python3 figs.py
echo "4/4 Figures 10-11"; python3 figs_mc.py
echo "Done. Figures in code/fig/, printed results in results_*.txt"
