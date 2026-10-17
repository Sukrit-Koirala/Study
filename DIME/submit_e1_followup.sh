#!/bin/bash
#SBATCH --job-name=e1_followup
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1_followup-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1_followup-%j.err

# Reuses the existing WikiText-103/gpt2 cache (no GPT-2 extraction of the full
# datastore) and re-derives DIME's clusters once (seed=42, deterministic, ~1-2 min).
# Items 5/6 additionally load GPT-2 and re-tokenize just the val split's chunks
# (cheap, CPU) plus run a handful of targeted forward passes (just the chunks
# containing the ~45-60 requested queries) to validate alignment before decoding
# context -- nowhere near a full re-extraction. 2h is a generous safety margin.
set -e
echo "E1 follow-up job started: $(date)"
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

python -c "import torch, transformers, datasets, numpy, pandas, tqdm, sklearn, scipy, matplotlib" 2>/dev/null || \
    pip install -r $BASE/GPT_Module/requirements.txt --quiet

export CUDA_VISIBLE_DEVICES=0

python -u $BASE/DIME/run_e1_followup.py

echo "E1 follow-up job done: $(date)"
