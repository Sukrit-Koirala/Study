#!/bin/bash
#SBATCH --job-name=crisp_r5_global_retrieval
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r5_global_retrieval-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r5_global_retrieval-%j.err

# Self-contained: imports R1/R4's own functions directly rather than
# depending on R1/R2/R3/R4's persisted results (R1-R3's results dirs exist
# locally but are empty; R4's does not exist at all -- checked directly).
# Reuses E1B.3's real, present System-3 pattern for the S3 oracle, and
# reads results/wikitext103_gpt2_tier1.json (real, present) only as a
# secondary/tuned-config reference, not re-run.
#
# Substantially more total work than R4: fits a rank-8 CRISP-R4 correction
# for ALL ~3,811 clusters (not a ~90-cluster sample), then evaluates 7
# retrieval systems (S0-S6) on an 8,000-query JS-decile-stratified sample
# (CRISPR5_N_EVAL, overridable) -- S0/S1 statistics and the routing-overlap
# diagnostic are computed at FULL val scale (77,089 queries) cheaply (batched
# kNN + assignment lookups only, no per-member loops, following E1B.3's own
# established precedent that this is feasible at full scale). The expensive
# per-member computations (S2-S6) are bounded to the 8,000-query sample.
#
# The pilot (40 tiny synthetic clusters, full S0-S6, all experiments, all
# deliverable files) completed in under 15 seconds of real compute, with
# global fitting at ~2.6s/40 clusters and the main evaluation loop at
# ~5ms/query dominated by S3's per-member distance computation. Extrapolating
# to 3,811 clusters and 8,000 queries, both phases should still be a matter
# of minutes to tens of minutes -- 16h is a generous safety margin given the
# genuinely larger total scope (fitting ~42x more clusters, evaluating ~27x
# more queries than the pilot), not an expected runtime. As with every prior
# script in this series, the dominant SHARED cost is cache loading, GPT-2
# loading, and the one-time seed=42 MiniBatchKMeans clustering over the full
# 381,127-point datastore. Checkpoints to crisp_r5_summary.json after every
# major section (full-scale baseline, routing overlap, per ~10% of global
# fitting, sanity checks, per ~10% of the main evaluation loop, each
# experiment).
#
# Phase 5 (secondary tuned-config comparison, CRISP rank 4/16 sensitivity,
# alternative centroid counts) and S7 (optional compressed mass predictor)
# are explicitly NOT run here -- deferred per the spec's own "do not
# sacrifice the main comparison" instruction; see report.md's Limitations
# and Q7 for whether S7 would be justified from this run's own numbers.
set -e
echo "CRISP-R5 job started: $(date)"
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

python -u $BASE/DIME/run_crisp_r5_global_retrieval.py

echo "CRISP-R5 job done: $(date)"
