#!/bin/bash
#SBATCH --job-name=dime_v2_token_conditioned_geometry
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_v2_token_conditioned_geometry-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_v2_token_conditioned_geometry-%j.err

# This is a diagnostic research study (DIME-V2), not part of the CRISP R-series
# or Y-series and not a new compressed retrieval algorithm. It asks whether
# centroid-relative, TOKEN-CONDITIONED residual geometry within a DIME cluster
# admits compact prototype representations that beat a matched-storage
# LABEL-AGNOSTIC geometric baseline on retrieval-relevant fidelity (both the
# all-member and local top-m truncated targets), not just on raw geometric
# reconstruction error. Writes only to
# results/dime_v2_token_conditioned_geometry_study/ (no "stage3_" prefix, per
# the spec's own Section 16 naming).
#
# No GPT-2 forward pass anywhere in this script -- every question here is
# about datastore GEOMETRY and RETRIEVAL-DISTRIBUTION fidelity, never GPT-2's
# own next-token distribution or NLL, so FrozenGPT2 is never imported. This is
# a deliberate scope reduction from the GPU-heavy R6-style pipeline, not an
# oversight -- the GPU request above is kept only for infra consistency with
# every other script in this series (this cluster's gpuGeneral partition is
# the only one used throughout the project) and because the real 381,127-point
# seed=42 MiniBatchKMeans clustering and cache loading are the dominant costs
# either way.
#
# Runs all seven required synthetic correctness checks (S0 centering identity,
# the exact-reduction identity, S1-S5 from the spec's Section 11) FIRST and
# asserts on failure before touching any real cache data.
#
# Self-review before the pilot caught and fixed two real bugs: (1) the
# reconstruction functions iterate over a cluster's FULL distinct-token set,
# but prototype fitting is bounded to the TOP_N_TOKENS_PER_CLUSTER most
# populous tokens -- any token outside that bounded subset (very likely in
# real data, e.g. singleton-count tokens) would have raised a KeyError. Fixed
# by giving every token outside the bounded subset an EXACT per-member
# fallback representation (verified to reduce exactly to R1's exact_evidence)
# in BOTH the label-conditioned and label-agnostic approaches, in both the
# Phase D reconstruction and the Phase F capacity-reallocation loop. (2) the
# label-conditioned and label-agnostic distortion metrics were being averaged
# over mismatched scopes (token-weighted mean across only the bounded subset
# vs. observation-weighted mean across the ENTIRE cluster) -- fixed so both
# are observation-weighted means over the identical bounded-subset scope,
# making the matched-storage comparison actually fair. A synthetic test (S5)
# also needed its hand-picked numeric values corrected after the first run
# showed the constructed example did not actually produce a measurable
# competition-weight difference while holding the candidate set fixed --
# verified independently before re-running, not just re-asserted.
#
# Deliberately scoped as a FOCUSED pilot: TARGET_N_CLUSTERS=10 clusters,
# rho_j-stratified (low/mid/high average observations per token), each with
# geometric/predictive analysis bounded to its TOP_N_TOKENS_PER_CLUSTER=15
# most populous tokens with population >= MIN_TOKEN_POP_FOR_GEOMETRY=5.
# Component grid r in {1,2,4,8}; Phase C2/C3 (label-agnostic baseline and
# observation-code/VQ baseline) are consolidated into one ordinary-k-means
# construction, per the spec's own explicit allowance that these are
# conceptually the same thing. Phase E's leave-one-out importance sweep is
# capped at the 50 most-frequently-selected prototypes per cluster, since the
# number of distinct selected prototypes is unbounded in principle and each
# one costs a full reconstruction pass.
#
# The pilot (18 tiny synthetic clusters, 5 distinct tokens each, the full
# Phase A-F pipeline, every deliverable file and all 7 plots) completed in
# under 10 seconds on CPU alone. 6h is a generous safety margin given the
# real WikiText clusters are larger and more numerous to sample from, not an
# expected runtime -- the dominant SHARED cost, as with every prior script in
# this series, is cache loading and the one-time seed=42 MiniBatchKMeans
# clustering over the full 381,127-point datastore. Checkpoints to
# dime_v2_summary.json after the synthetic checks and again at the end.
set -e
echo "DIME-V2 job started: $(date)"
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

python -c "import numpy, pandas, sklearn, scipy, matplotlib" 2>/dev/null || \
    pip install -r $BASE/GPT_Module/requirements.txt --quiet

python -u $BASE/DIME/run_dime_v2_token_conditioned_geometry.py

echo "DIME-V2 job done: $(date)"
