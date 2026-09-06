#!/bin/bash
#SBATCH --job-name=extract_wikitext_medium
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/extract_wikitext_medium-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/extract_wikitext_medium-%j.err

set -e
echo "WikiText-103 / gpt2-medium extraction job started: $(date)"
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
  --n_datastore 3000 \
  --n_controller_train 500 \
  --n_val 500 \
  --batch_size 128 \
  --out_dir $BASE/GPT_Module/cache

echo "WikiText-103 / gpt2-medium extraction job done: $(date)"
