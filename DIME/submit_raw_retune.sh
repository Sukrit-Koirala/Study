#!/bin/bash
#SBATCH --job-name=dime_raw_retune
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_raw_retune-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_raw_retune-%j.err

# Usage: sbatch --export=DATASET=wikitext2,MODEL=gpt2 submit_raw_retune.sh
# Optional: add SKIP_COVERAGE=1 to the --export list to skip raw_coverage.
# Requires results/${DATASET}_${MODEL}_tier1.json to already exist.
# 12h is a ceiling, not an estimate: TinyStories / WikiText-103 should finish far sooner.
# Works for the five settings that have a cached extraction + Tier 1 JSON. NOT for the original
# TinyStories/gpt2 setting: it streams and re-encodes data on the fly (no .npz cache, no Tier 1 JSON).
# Suggested smoke test first: DATASET=tinystories,MODEL=gpt2-medium (cheapest of the five).
set -e
echo "Raw re-tune (${DATASET}/${MODEL}) job started: $(date)"
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

python -c "import torch, transformers, datasets, numpy, pandas, tqdm, sklearn, scipy" 2>/dev/null || \
    pip install -r $BASE/GPT_Module/requirements.txt --quiet

export CUDA_VISIBLE_DEVICES=0

python -u $BASE/DIME/run_raw_retune.py --dataset "$DATASET" --model "$MODEL" ${SKIP_COVERAGE:+--skip_coverage}

echo "Raw re-tune (${DATASET}/${MODEL}) job done: $(date)"
