#!/bin/bash
#SBATCH --job-name=crisp_r7_adaptive_basis_diagnostics
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r7_adaptive_basis_diagnostics-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r7_adaptive_basis_diagnostics-%j.err

# Phase A audit (r3_r4_method_audit.md) was written from direct inspection
# of the actual R3/R4 source files, not from the spec's summary equations.
# R3/R4's real persisted results are not available locally (checked
# directly) -- R4's exact cluster-sampling procedure is reproduced fresh
# here via the same deterministic (seed=42) code, not cited.
#
# Runs the spec's required synthetic correctness checks (known-optimal-
# basis recovery, monotonic residual decrease, score-vs-direct-refit
# cross-check, gauge shift-invariance) FIRST, before touching any real
# cache data, per the spec's own explicit execution order -- these must
# all pass or the script aborts (assert) before any real fitting begins.
#
# Main cost: R4's exact sampling procedure (base-20 + stratified expansion
# + sensitivity enrichment, reused unchanged) typically yields ~66-90
# clusters; for each, orthogonal greedy selection over a 64-candidate pool
# (32 batched linear-algebra steps) plus 5 methods (B0-B4) x 5 ranks x a
# 4-point ridge-lambda grid, all closed-form. The pilot (90 tiny synthetic
# clusters, full method/rank grid, all deliverable files, including the
# synthetic correctness checks) completed in under 20 seconds of real
# compute. 10h is a generous safety margin given the real WikiText clusters
# are larger and more numerous than the pilot's, not an expected runtime --
# the dominant SHARED cost, as with every prior script in this series, is
# cache loading, GPT-2 loading, and the one-time seed=42 MiniBatchKMeans
# clustering over the full 381,127-point datastore. Checkpoints to
# crisp_r7_summary.json after every ~10% of the main cluster loop.
#
# Per the spec's explicit instruction, this script does NOT start R8 or Y1.
set -e
echo "CRISP-R7 job started: $(date)"
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

python -u $BASE/DIME/run_crisp_r7_adaptive_basis_diagnostics.py

echo "CRISP-R7 job done: $(date)"
