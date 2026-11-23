#!/bin/bash
#SBATCH --job-name=y1_nonlinear_function_feasibility
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/y1_nonlinear_function_feasibility-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/y1_nonlinear_function_feasibility-%j.err

# This is the FIRST genuine Y1 (nonlinear). The script previously named
# Y1 (run_y1_per_cluster_function.py, results under
# results/stage3_y1_per_cluster_function/, both REAL and already present
# locally) is referred to as "R-extra" throughout -- its file, results,
# and submit script are left completely untouched by this script.
#
# Runs all seven required synthetic correctness checks FIRST (known-target
# consistency, m=1 one-hot behavior, probability validity, non-affine XOR
# expressivity, genuine discontinuity at a label-switch boundary, gauge
# shift-invariance, storage accounting) and asserts on failure before
# touching any real cache data, per the spec's own explicit execution
# order.
#
# Deliberately scoped as a FOCUSED PILOT (TARGET_N_CLUSTERS=15, audited
# from a BROADER candidate pool under STRICTER query-sufficiency
# thresholds than R-extra used, per the spec's own Phase E instruction --
# R-extra's real results showed only 11 of its base-20 sample had enough
# queries even under a looser bar). Main cost: for each of ~15 clusters,
# 4 models (D0 trivial, A0 R-extra's reused affine basis, N1/N2 genuinely
# nonlinear RBF features with a 3-point bandwidth sweep) x 4 ranks, all
# closed-form ridge fits, plus a 4-value m-sweep and a 5-point learning
# curve. The pilot (40 tiny synthetic clusters, full model/rank grid, the
# m-sweep, the learning curve, every deliverable file, including all
# seven synthetic correctness checks) completed in well under a minute of
# real compute. 6h is a generous safety margin given the real WikiText
# clusters are larger than the pilot's, not an expected runtime -- the
# dominant SHARED cost, as with every prior script in this series, is
# cache loading, GPT-2 loading, and the one-time seed=42 MiniBatchKMeans
# clustering over the full 381,127-point datastore. Checkpoints to
# y1_nonlinear_summary.json after every major section.
set -e
echo "Y1 (nonlinear) job started: $(date)"
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

python -u $BASE/DIME/run_y1_nonlinear_function_feasibility.py

echo "Y1 (nonlinear) job done: $(date)"
