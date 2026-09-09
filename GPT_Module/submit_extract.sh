#!/bin/bash
#SBATCH --job-name=dime_extract
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/dime_extract-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/GPT_Module/logs/dime_extract-%j.err

# Usage: sbatch --export=DATASET=wikitext2,MODEL=gpt2,N_DS=1200,N_CT=200,N_VAL=200 submit_extract.sh
# Defaults below match the proven wikitext103/gpt2-medium extraction job.
# For wikitext2 (much smaller corpus, ~2M tokens total) lower N_DS or the job may
# run out of distinct chunks — run a small test (e.g. N_DS=100) first to confirm
# collect_chunks_split can satisfy the requested counts before committing to a full run.
set -e
echo "Extraction (${DATASET}/${MODEL}) job started: $(date)"
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

python -u $BASE/GPT_Module/extract_and_cache.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --seq_len "${SEQ_LEN:-128}" \
  --n_datastore "${N_DS:-3000}" \
  --n_controller_train "${N_CT:-500}" \
  --n_val "${N_VAL:-500}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --out_dir $BASE/GPT_Module/cache

echo "Extraction (${DATASET}/${MODEL}) job done: $(date)"
