#!/bin/bash
#SBATCH --job-name=crisp_r6_gpt2_interpolation
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r6_gpt2_interpolation-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r6_gpt2_interpolation-%j.err

# Self-contained: imports R5's own functions directly (fit_global_crisp_cluster,
# crisp_predict, cluster_distances, exact_cluster_dist_and_mass) rather than
# depending on R5's persisted results, which do not exist locally (checked
# directly). Reuses the cached p_lm_true (GPT-2's own true-token probability,
# confirmed present in every .npz cache via cache_io.py) directly -- NO new
# GPT-2 forward passes are run anywhere in this script.
#
# Despite evaluating MORE systems (M0-M6) than R5's S0-S6, this script needs
# LESS expensive per-query work than R5: interpolated NLL only requires the
# retrieval method's probability on the SPECIFIC true token, not a full
# distribution, so no TV/JS/top-k bookkeeping is needed in the main loop.
# Per the spec's own "prioritize evaluating all eligible positions" instruction,
# the main per-cluster evaluation loop (needed for M3/M4/M5/M6's true-token
# probability) runs over ALL ~77,089 val queries directly, not a bounded
# sample -- the same R5-style 8,000-query JS-stratified sample is reproduced
# as a free paired diagnostic from the same already-computed per-query values.
#
# The pilot (40 tiny synthetic clusters, full M0-M6, all experiments, all
# deliverable files, run over its own full 4,000-query val set) completed in
# under a minute, with global fitting at ~3.4s/40 clusters and the main loop
# at ~7.6ms/query. Extrapolating to 3,811 clusters (~95x) and 77,089 queries
# (~19x), both phases should still be a matter of minutes; 16h is a generous
# safety margin given the genuinely larger real-cluster scale, not an
# expected runtime. As with every prior script in this series, the dominant
# SHARED cost is cache loading, GPT-2 loading, and the one-time seed=42
# MiniBatchKMeans clustering over the full 381,127-point datastore.
# Checkpoints to crisp_r6_summary.json after every major section and every
# ~5% of the main evaluation loop.
#
# Phase 5 sensitivity items beyond the spec's own required scope (a tuned-
# CRISP re-fit at the tuned raw/DIME configuration) are explicitly NOT run --
# only raw/DIME's tuned-config NLL is reproduced as a baseline-verification
# check (Phase 1), consistent with R5's own precedent of evaluating CRISP
# only at the matched rank=8 configuration.
set -e
echo "CRISP-R6 job started: $(date)"
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

python -u $BASE/DIME/run_crisp_r6_gpt2_interpolation.py

echo "CRISP-R6 job done: $(date)"
