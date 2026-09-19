#!/bin/bash
#SBATCH --job-name=dime_exp
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_exp-%x-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/dime_exp-%x-%j.err

# Generic wrapper for the follow-up experiment scripts in DIME/.
# Usage: sbatch --job-name=NAME --time=HH:MM:SS \
#          --export=SCRIPT=run_latency.py,DATASET=wikitext103,MODEL=gpt2[,EXTRA="--ratios 1000,300"] submit_exp.sh
# Scripts that take no --dataset/--model (the seed rerun) can set NOARGS=1.
set -e
echo "Job ${SCRIPT} (${DATASET}/${MODEL}) started: $(date)"
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

if [ -n "$NOARGS" ]; then
    python -u $BASE/DIME/$SCRIPT $EXTRA
else
    python -u $BASE/DIME/$SCRIPT --dataset "$DATASET" --model "$MODEL" $EXTRA
fi

echo "Job ${SCRIPT} (${DATASET}/${MODEL}) done: $(date)"
