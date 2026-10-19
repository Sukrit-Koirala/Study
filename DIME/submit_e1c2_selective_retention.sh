#!/bin/bash
#SBATCH --job-name=e1c2_selective_retention
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1c2_selective_retention-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1c2_selective_retention-%j.err

# Reuses the existing cache, re-derives DIME's clusters once (seed=42,
# deterministic), and loads the real, already-persisted E1C tuned per-query
# CSV directly (no recomputation for that part). The expensive new work is:
# ~16 restricted-index builds/evaluations (Part A, Sections 8-9) and two full-
# scale (77,089-query) System-3 computations (Part B, ~30-40 min each per the
# timing established in E1B.4). 8h is a generous safety margin given the many
# moving pieces; checkpoints to e1c2_summary.json after each major section.
set -e
echo "E1C.2 job started: $(date)"
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

python -u $BASE/DIME/run_e1c2_selective_retention.py

echo "E1C.2 job done: $(date)"
