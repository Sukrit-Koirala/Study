#!/bin/bash
#SBATCH --job-name=verify_cache
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/verify_cache-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/verify_cache-%j.err

set -e
echo "Cache verification job started: $(date)"
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

python -c "import torch, transformers, datasets, numpy, pandas, tqdm, sklearn, scipy" 2>/dev/null || \
    pip install -r $BASE/GPT_Module/requirements.txt --quiet

export CUDA_VISIBLE_DEVICES=0

python $BASE/GPT_Module/extract_and_cache.py \
  --dataset wikitext103 \
  --model gpt2-medium \
  --seq_len 128 \
  --n_datastore 40 \
  --n_controller_train 5 \
  --n_val 5 \
  --batch_size 32 \
  --out_dir $BASE/GPT_Module/cache

echo "Cache verification job done: $(date)"
