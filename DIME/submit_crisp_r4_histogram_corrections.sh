#!/bin/bash
#SBATCH --job-name=crisp_r4_histogram_corrections
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r4_histogram_corrections-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r4_histogram_corrections-%j.err

# Self-contained: imports R1/R3's own functions directly rather than
# depending on R1/R2/R3's persisted results (R1's dir exists locally but is
# empty; R2/R3 do not exist locally at all -- checked directly). Reproduces
# R1's raw-evidence objective and R3's direct-distribution objective fresh
# via direct import, under the identical seed=42 clustering.
#
# Scope: Phases 1-3 of the spec's own staged plan. Cluster sample is
# base-20 (R1's rule) + a deterministic population-stratified expansion
# (~60 more) + sensitivity-enriched top-up if needed -- expect roughly
# 80-100 clusters, reported exactly in the output, not fixed in advance.
# 5 baselines (B2-B5 fitted; B1/B0/B6 trivial/closed-form) x 7 ranks x
# ~90 clusters, each fit a bounded closed-form ridge regression (<=150
# fit queries, <=32 features) -- no full-val pass anywhere. The pilot
# (45 tiny synthetic clusters, all baselines, all ranks, all deliverable
# files) completed in under 15 seconds of actual compute. The dominant
# real cost, as with every prior script in this series, is cache loading,
# GPT-2 loading, and the one-time seed=42 MiniBatchKMeans clustering over
# the full 381,127-point datastore. 8h is a generous safety margin scaled
# up from R2's 6h budget to account for the larger cluster sample and more
# baselines, not an expected runtime. Checkpoints to crisp_r4_summary.json
# after every major section.
#
# Phase 4 (adaptive rank allocation, E8) is explicitly NOT run here --
# deferred per the spec's own "do not automatically proceed" instruction.
set -e
echo "CRISP-R4 job started: $(date)"
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

python -u $BASE/DIME/run_crisp_r4_histogram_corrections.py

echo "CRISP-R4 job done: $(date)"
