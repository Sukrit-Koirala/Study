#!/bin/bash
#SBATCH --job-name=e2km1_dense_kernel_mean
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2km1_dense_kernel_mean-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2km1_dense_kernel_mean-%j.err

# Self-contained: does NOT depend on any other Stage-3 script's persisted
# results (the real E2-GEOM1 summary was not available locally when this
# script was written, so it reruns GEOM1's System B / "Kernel Variant A"
# computation itself, under the same matched config and seed=42 clustering,
# rather than citing an external number -- see the script's own docstring
# for the full equivalence-identification reasoning).
#
# Reuses the existing cache and re-derives DIME's clusters once (seed=42).
# Primary cost: two full-cluster, member-level local-retrieval passes at
# FULL val scale (77,089 queries x k=20 retrieved clusters x each cluster's
# full original membership) -- Kernel A (exponential, reused from
# run_e2geom1_within_centroid_geometry.py) and Kernel B (Gaussian squared,
# new here). Same cost class as the old S3/GEOM1-System-B computation
# (~30-40 min each per established precedent). B5's shuffle control and
# Diagnostics B/C are scoped to bounded val subsamples specifically to avoid
# further full-scale passes. 10h is a generous safety margin, not an
# expected runtime. Checkpoints to e2km1_summary.json after every major
# section.
set -e
echo "E2-KM1-DENSE job started: $(date)"
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

python -u $BASE/DIME/run_e2km1_dense_kernel_mean.py

echo "E2-KM1-DENSE job done: $(date)"
