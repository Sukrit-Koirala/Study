#!/bin/bash
#SBATCH --job-name=y1_per_cluster_function
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/y1_per_cluster_function-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/y1_per_cluster_function-%j.err

# First delivery of a new, deliberately small Y-series -- NOT a CRISP R-series
# continuation. Scoped to R1's exact base-20 cluster sample (10 largest-
# population + 10 smaller/rarer-label, population >= 30), per the spec's own
# "do not turn this into an end-to-end experiment" instruction -- no global
# centroid routing/mixing, no mass-aware aggregation, no kNN-LM evaluation.
#
# Main cost: 20 clusters x 4 m-values x 6 ranks x 2 function families x a
# 4-point ridge-lambda grid, all closed-form, bounded fit/tune/test query
# counts (<=150/50/100). The pilot (25 tiny synthetic clusters, full grid,
# all deliverable files) completed in under 20 seconds of real compute.
# 4h is a generous safety margin, not an expected runtime -- the real cost,
# as with every prior script in this project, is cache loading, GPT-2
# loading, and the one-time seed=42 MiniBatchKMeans clustering over the full
# 381,127-point datastore. Checkpoints to y1_summary.json after every major
# section and every ~20% of the main cluster loop.
#
# Per the spec's final instruction, this script does NOT automatically
# begin a Y2 follow-up.
set -e
echo "Y1 job started: $(date)"
echo "Node: $(hostname)"

BASE=/home/sukrit.koirala/ondemand/upload_me/Research/Study
cd $BASE/DIME

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate region_tokenizer 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate region_tokenizer 2>/dev/null || conda activate base
fi

python -c "import torch, numpy, pandas, sklearn, scipy, matplotlib" 2>/dev/null || \
    pip install -r $BASE/GPT_Module/requirements.txt --quiet

export CUDA_VISIBLE_DEVICES=0

python -u $BASE/DIME/run_y1_per_cluster_function.py

echo "Y1 job done: $(date)"
