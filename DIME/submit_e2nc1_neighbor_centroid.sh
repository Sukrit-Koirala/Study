#!/bin/bash
#SBATCH --job-name=e2nc1_neighbor_centroid
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2nc1_neighbor_centroid-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2nc1_neighbor_centroid-%j.err

# Requires results/stage3_e2tri1_anchor_conditioned/e2tri1_summary.json to
# already exist (E2-TRI1's real run) -- this script reconstructs TRI1's K=8
# config from its persisted hyperparameters for the N7 reference row and
# Experiment F's discrete-vs-continuous comparison, and hard-asserts the
# reconstruction reproduces TRI1's real mean NLL before trusting it further.
#
# Reuses the existing cache and re-derives DIME's clusters once (seed=42).
# Primary new cost beyond TRI1: one additional kNN query of the FULL
# datastore (381,127 points) against the 3811 centroids at k=3 (to assign
# each datastore entry's 2nd/3rd-nearest centroid), plus the main ablation
# grid itself -- which, unlike TRI1's vectorized anchor-distance math, is
# just O(1) dict/Counter lookups per (query, retrieved-cluster) pair, so
# each config should be considerably cheaper than TRI1's per-config cost.
# Lambda/gamma tuning runs on the FULL controller_train split (cheap enough
# not to need a subsample). 6h is a generous margin, not an expected
# runtime -- TRI1's own real run (a strictly heavier computation per config)
# finished in about 10 minutes. Checkpoints to e2nc1_summary.json after
# every major section.
set -e
echo "E2-NC1 job started: $(date)"
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

if [ ! -f "$BASE/DIME/results/stage3_e2tri1_anchor_conditioned/e2tri1_summary.json" ]; then
    echo "ERROR: results/stage3_e2tri1_anchor_conditioned/e2tri1_summary.json not found -- run E2-TRI1 first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

python -u $BASE/DIME/run_e2nc1_neighbor_centroid_conditional.py

echo "E2-NC1 job done: $(date)"
