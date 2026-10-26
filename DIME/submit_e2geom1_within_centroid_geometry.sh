#!/bin/bash
#SBATCH --job-name=e2geom1_within_centroid
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2geom1_within_centroid-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2geom1_within_centroid-%j.err

# Requires results/stage3_e2tri1_anchor_conditioned/e2tri1_summary.json to
# already exist -- System C and System D both reuse TRI1's real persisted
# anchor-selection method and K, and System D's reconstruction is hard-
# verified against TRI1's real mean NLL before use. results/stage3_e2_
# information_allocation/e2_summary.json is used as an optional cross-check
# for System B (its real S3 result) if present; not required.
#
# Reuses the existing cache and re-derives DIME's clusters once (seed=42).
# Primary cost: Systems B and C are full-cluster, member-level local
# retrieval at FULL val scale (77,089 queries x k=20 retrieved clusters x a
# cluster's full original membership) -- the same cost class as the old S3
# computation from E1B.4/E1C.2/E2 (~30-40 min each per that established
# precedent), done twice here (raw 768-dim space, then K=8 anchor-projected
# space). Experiment B's query-level analyses and Experiment E's K-sweep
# (18 point-level System-C configurations) are both scoped to bounded val
# subsamples specifically to avoid repeating that cost many more times. 12h
# is a generous safety margin, not an expected runtime. Checkpoints to
# e2geom1_summary.json after every major section.
set -e
echo "E2-GEOM1 job started: $(date)"
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

python -u $BASE/DIME/run_e2geom1_within_centroid_geometry.py

echo "E2-GEOM1 job done: $(date)"
