#!/bin/bash
#SBATCH --job-name=e1b4_nll_recovery
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1b4_nll_recovery-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1b4_nll_recovery-%j.err

# Full 77,089-query evaluation (not E1B.3's 3,000-query sample) -- confirmed
# computationally feasible via a realistic-dimension local pilot (~40 queries/sec
# for System 3's per-cluster-member computation, extrapolating to ~30-40 min for
# the full set, per config). Reuses the existing cache and re-derives DIME's
# clusters once (seed=42, deterministic) -- no GPT-2 re-extraction, no re-tuning.
# 3h is a safety margin well above the extrapolated runtime.
set -e
echo "E1B.4 job started: $(date)"
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

python -u $BASE/DIME/run_e1b4_nll_recovery.py

echo "E1B.4 job done: $(date)"
