#!/bin/bash
#SBATCH --job-name=dime_v2_c2_1_local_retrieval_audit
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_v2_c2_1_local_retrieval_audit-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_v2_c2_1_local_retrieval_audit-%j.err

# DIME-V2 Experiment C2.1: Exact Local Retrieval Behavior Audit.
#
# Loads C1's PERSISTED cluster artifact (dime_v1_cluster_artifact.npz +
# construction_fingerprint.json, both verified by SHA-256 against the
# externally supplied expected hashes AND the fingerprint's own recorded
# values) -- never retrains MiniBatchKMeans. Reuses the exact production
# centroid-routing mechanism (k_route=20, tau_route=2.0, the real DIME
# matched-configuration centroid softmax) and the exact DIME-V1 static
# per-cluster histogram. Stratified-samples ~60-80 clusters (plus a handful
# of deliberately unusual ones, labeled separately) and investigates exact
# local top-m candidate selection, competition weighting, and predictive
# distribution variation across their routed val/controller_train queries --
# a behavioral diagnostic, not a compression algorithm.
#
# Went through two external review rounds before this run was authorized:
#   - Revision 0 -> Revision 1 (8 points): implemented the actual centroid
#     mixture-weight softmax (previously only distances/indices were
#     computed); repaired two ineffective synthetic tests (one had a tied
#     scenario masked by a dead "or True"; one never actually exercised
#     bar_F_train's construction); added the margin-vs-query-displacement
#     stability guarantee as an ACTIVE assertion (not just a report) plus
#     directed-pair-reuse tracking; rebuilt failure_cases.md with full
#     local-position/global-row-id/label/distance/weight detail; added
#     per-cluster sampling cell/population/inclusion-probability reporting
#     and a `stratified_only` correlation scope (confirmed in the pilot to
#     give materially different numbers than the full sampled set); filled
#     in Q6/Q11 with real computed statistics (no more placeholders);
#     required C1's fingerprint JSON (was optional) and added dtype/dimension/
#     finite-vector/traffic-vs-C1 invariant checks; replaced the bypassable
#     "different cache path = synthetic" guard with an explicit
#     DIMEV2C21_MODE declaration cross-checked against the cache path.
#   - 3 final corrections: added the JS divergence between nearby-query
#     output distributions (spec Part 5 required it; TV alone was reported);
#     fixed the high/low "query dependence" failure cases to actually be
#     selected by I_C (the spec's own definition of query dependence), not
#     by nearby-pair TV; fixed frac_boundary_tie_diff_label to be CONDITIONAL
#     on landing on a tie (matches its "of those" description) instead of
#     dividing by all sampled queries, reporting None rather than a
#     misleading 0 when a cluster had zero ties.
#
# 20/20 synthetic correctness checks pass (isolated from any real cache,
# run unconditionally before either half of the execution guard is checked).
# The full pipeline was pilot-tested twice more against a 40-cluster
# synthetic dataset after every fix above, including once with the real
# k_route=20 specifically to exercise the routing-traffic-vs-C1 comparison
# (passed with float-noise-level agreement, 5.55e-17, well under the 1e-6
# explained tolerance) -- never against real data.
#
# Execution guard: this script sets BOTH DIMEV2C21_MODE=real_approved and
# DIMEV2C21_APPROVED=1 below -- both are required unconditionally for the
# real-data pipeline to run at all (see run_dime_v2_c2_1_local_retrieval_audit.py's
# module docstring / DIME_V2_C2_1_PREFLIGHT_REVIEW.md Part G). A run that
# omits either, or that points DIMEV2C21_CACHE_DIR anywhere other than the
# real cache, is refused before any directory is created or any file written.
set -e
echo "DIME-V2 C2.1 job started: $(date)"
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

# Required, unconditionally, for the real-data pipeline to run at all.
export DIMEV2C21_MODE=real_approved
export DIMEV2C21_APPROVED=1

python -u $BASE/DIME/run_dime_v2_c2_1_local_retrieval_audit.py

echo "DIME-V2 C2.1 job done: $(date)"
