#!/bin/bash
#SBATCH --job-name=dime_gpt_baseline
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/dime_gpt_baseline-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/dime_gpt_baseline-%j.err

set -e
echo "GPT-only baseline job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

BASE=/home/sukrit.koirala/ondemand/upload_me/Research/Study
cd $BASE/GPT_Module

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate region_tokenizer 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate region_tokenizer 2>/dev/null || conda activate base
fi

python -c "import torch, transformers, datasets, numpy, pandas, tqdm, sklearn" 2>/dev/null || \
    pip install -r requirements.txt --quiet

export PYTHONPATH="$BASE/GPT_Module:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python $BASE/GPT_Module/run_baseline.py

echo "GPT-only baseline job done: $(date)"
