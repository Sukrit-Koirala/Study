#!/bin/bash
#SBATCH --job-name=e2_information_allocation
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2_information_allocation-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2_information_allocation-%j.err

# Reuses the existing cache (datastore/controller_train/val) and re-derives
# DIME's clusters once (seed=42, deterministic). Primary cost: 4 strategies x
# 3 budgets x full 77,089-query evaluation (member-aware, but with far fewer
# members/cluster than full S3, so each should be noticeably faster than
# E1B.4/E1C.2's ~30-40 min full-S3 runs), plus one full S3 computation for the
# R_S3 reference, plus Strategy A's 3-seed random-vs-centroid comparison, plus
# Part E's worst-query re-evaluation per strategy/budget. 12h is a generous
# safety margin given how many pieces this has; checkpoints to e2_summary.json
# after every major section.
set -e
echo "E2 job started: $(date)"
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

python -u $BASE/DIME/run_e2_information_allocation.py

echo "E2 job done: $(date)"
