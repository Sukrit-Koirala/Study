#!/bin/bash
#SBATCH --job-name=crisp_subspace_projection
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_subspace_projection-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_subspace_projection-%j.err

# Self-contained: does not depend on any other Stage-3 script's persisted
# results. Phase 1-3 scope only (per the spec's own explicit execution
# priorities) -- 20 sampled clusters (of 3,811), bounded fit/tune/test query
# counts per cluster, rank grid capped at 32. Unlike TRI1/GEOM1/KM1, this is
# NOT a full-val pass: per-cluster ridge-regression fitting is closed-form
# and cheap, and the exact-evidence oracle is only evaluated for a few
# hundred queries per sampled cluster, not all 77,089 val queries. The main
# real cost is cache loading, GPT-2 loading, and the one-time seed=42
# MiniBatchKMeans clustering over the full 381,127-point datastore, shared
# with every prior script in this series. 4h is a generous safety margin,
# not an expected runtime -- the pilot (40 tiny synthetic clusters)
# completed in well under a minute. Checkpoints to crisp_summary.json after
# every major section. Phase 4 (full 3,811-cluster, full-val scale) is
# explicitly deferred to a follow-up, per the spec's own "do not run a large
# sweep immediately" instruction.
set -e
echo "CRISP job started: $(date)"
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

python -u $BASE/DIME/run_crisp_subspace_projection.py

echo "CRISP job done: $(date)"
