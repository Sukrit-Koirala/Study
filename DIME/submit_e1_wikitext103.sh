#!/bin/bash
#SBATCH --job-name=e1_wikitext103
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1_wikitext103-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1_wikitext103-%j.err

# E1 diagnostic pipeline — reuses the already-cached WikiText-103/gpt2 extraction
# and the already-tuned (k,tau,alpha) from results/wikitext103_gpt2_tier1.json, so
# no GPU-heavy extraction or grid search happens here, only: DIME re-clustering
# (same seed=42, ~1-2 min at this scale per the original tier1 timing), one batched
# kneighbors query per index, and CPU-bound per-query metric computation on 77,089
# val positions. 4h is a generous safety margin, not an estimate.
set -e
echo "E1 (WikiText-103/gpt2) job started: $(date)"
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

python -u $BASE/DIME/run_e1_wikitext103.py

echo "E1 (WikiText-103/gpt2) job done: $(date)"
