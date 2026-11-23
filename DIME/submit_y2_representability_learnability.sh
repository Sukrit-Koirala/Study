#!/bin/bash
#SBATCH --job-name=y2_representability_learnability
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/y2_representability_learnability-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/y2_representability_learnability-%j.err

# Audits Y1's REAL results directly (results/stage3_y1_nonlinear_function_feasibility/,
# present locally and read, not reproduced) before any new code: real
# sample was 15 clusters from 86/150 candidates; at m=8/rank=8, D0=0.886,
# A0=0.829, N1=0.851, N2=0.830 mean held-out TV, with N2 NOT significantly
# beating A0 and query-change preservation failing for every model. These
# real numbers are the historical benchmark this script's operational
# thresholds are measured against.
#
# Adds a torch-based MLP family (torch is already a project dependency via
# gpt_class.FrozenGPT2, not new) and a larger-rank RBF family, plus a
# mandatory training-set memorization test that isolates architecture/
# optimization capacity from generalization/data limitations -- Y1 never
# separated these. Runs on CPU or GPU automatically (torch.cuda.is_available()).
#
# Deliberately scoped as a FOCUSED pilot: TARGET_N_CLUSTERS=5 (per the
# spec's own "approximately 3-5 clusters" instruction), MLP capacity grid
# limited to widths {128,512} x depths {2,4} (the spec's own "initial
# configurations, not a mandatory full Cartesian product"). All seven of
# Y1's own synthetic checks are reused as already-verified (not re-run);
# this script's own two NEW checks (MLP solves XOR; the exact boundary-
# jump formula matches a real numerical teacher evaluation) run first and
# assert-pass before any real cache is touched.
#
# The pilot (20 tiny synthetic clusters, the full capacity/memorization/
# learning-curve/boundary-enriched/query-change pipeline, all 12 required
# deliverable files) completed in under 4 minutes on CPU alone -- most of
# that time is MLP training (21 small torch fits per cluster x 5 clusters).
# 8h is a generous safety margin given the real WikiText clusters are
# larger, not an expected runtime -- the dominant SHARED cost, as with
# every prior script in this series, is cache loading, GPT-2 loading, and
# the one-time seed=42 MiniBatchKMeans clustering over the full
# 381,127-point datastore. Checkpoints to y2_summary.json throughout.
set -e
echo "Y2 job started: $(date)"
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

python -u $BASE/DIME/run_y2_representability_learnability.py

echo "Y2 job done: $(date)"
