#!/bin/bash
#SBATCH --job-name=dime_v2_c1_cluster_structure_audit
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_v2_c1_cluster_structure_audit-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_v2_c1_cluster_structure_audit-%j.err

# DIME-V2 Experiment C1: a purely DIAGNOSTIC structural audit of the
# existing DIME-V1 cluster construction (MiniBatchKMeans, seed=42,
# n_clusters=3811, over the real 381,127-row WikiText-103/gpt2 datastore).
# Does NOT change n_clusters, does NOT replace MiniBatchKMeans, does NOT
# modify centroid routing, does NOT train anything -- characterizes
# population, geometry, local dimensionality/anisotropy, token composition,
# and query-routing traffic across ALL 3,811 configured clusters, as a
# prerequisite to any future compression-representation design.
#
# This script went through three preflight review rounds
# (DIME_V2_PREFLIGHT_REVIEW.md, Revisions 1-3) before being approved to run
# against the real cache; its former RuntimeError execution guard has been
# removed now that approval was given. No GPT-2 forward pass is needed
# (every question here is about datastore geometry/composition, not GPT-2's
# own NLL) -- the GPU request above is kept only for infra consistency with
# every other script in this series (gpuGeneral is the only partition used
# throughout the project) and because loading the full 381,127 x 768 cache
# and the one-time seed=42 MiniBatchKMeans fit are the dominant costs either
# way, not GPU compute.
#
# Review-driven fixes worth knowing about before reading the output:
#   - Routing traffic is reported as TWO genuinely different, separately
#     normalization-checked quantities: traffic_top1_* (single-nearest-
#     centroid diagnostic grouping, as build_splits_for uses elsewhere in
#     this project) and traffic_top20_* (the real multi-centroid production
#     mixing convention mix_dime_and_lm actually uses). Never conflate them.
#   - Local dimensionality (PCA/effective-rank) runs on a bounded, GENUINELY
#     2-D population x dispersion quantile-stratified sample (an 8x5=40-cell
#     grid, one deterministic cluster per occupied cell) -- see
#     dimensionality_sample_strata.csv for exactly which cells were occupied
#     and which cluster represents each. This is NOT an all-cluster
#     statistic; structural_relationships.csv's "scope" column makes this
#     explicit everywhere a dimensionality-derived correlation is reported.
#   - n_empty and n_singleton (population==0 vs ==1) are reported as
#     separate fields -- singleton clusters have well-defined geometry and
#     are processed through the same pipeline as every other nonempty
#     cluster; only truly empty clusters get a population-only row.
#   - This run's own fitted centroids/assignment are saved to
#     dime_v1_cluster_artifact.npz, with a SHA-256 fingerprint (verified
#     against the actually-persisted bytes, not just the in-memory arrays)
#     and full environment metadata (python/numpy/sklearn/scipy/pandas
#     versions, platform) in construction_fingerprint.json -- no such
#     artifact existed anywhere in this repository before this script, and
#     it exists specifically so a FUTURE run can compare directly instead of
#     assuming bitwise reproducibility from the seed alone. It does NOT, and
#     cannot, retroactively recover the sklearn version that produced the
#     historical wikitext103_gpt2_tier1.json / run_manifest.json artifacts
#     this script cites for context -- that remains a genuinely unresolved,
#     explicitly stated limitation (see DIME_V2_PREFLIGHT_REVIEW.md, Section
#     F), not something this script claims to have fixed.
#   - The key/label alignment check verifies only that the two arrays have
#     equal row counts; the cache format stores no row-id field, so true
#     semantic row-for-row provenance cannot be independently verified from
#     the cache alone -- stated as a limitation, not implied to be proven.
#
# All 13 required synthetic correctness checks (population/geometry/
# singleton/effective-rank/token-composition/routing-normalization/
# duplicate-index-rejection/Spearman-vs-Pearson/cluster-grouping/fingerprint-
# integrity/2-D-stratification-coverage) run first and assert-pass before
# any real cache is touched. A full end-to-end pilot (synthetic cache, 15
# tiny clusters, every deliverable file including the new
# dime_v1_cluster_artifact.npz/construction_fingerprint.json/
# dimensionality_sample_strata.csv) completed in under 10 seconds on CPU
# alone and surfaced one cosmetic reporting fix (final_report.md's Q2 now
# explicitly states whether the computed n_clusters formula matches the
# persisted tier1 value, rather than just printing both numbers side by
# side). 4h is a generous safety margin given the real datastore is larger
# and this is a single-pass descriptive audit (no fitting, no grid search),
# not an expected runtime -- the dominant cost, as with every prior script
# in this series, is cache loading and the one-time seed=42 MiniBatchKMeans
# clustering over the full 381,127-point datastore.
set -e
echo "DIME-V2 C1 job started: $(date)"
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

echo "scikit-learn version actually used for this run:"
python -c "import sklearn; print(sklearn.__version__)"

python -u $BASE/DIME/run_dime_v2_c1_cluster_structure_audit.py

echo "DIME-V2 C1 job done: $(date)"
