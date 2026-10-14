#!/bin/bash
#SBATCH --job-name=clean_rerun
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/clean_rerun-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/clean_rerun-%j.err

# Usage: sbatch --export=DATASET=tinystories,MODEL=gpt2 submit_clean_rerun.sh
#    or: sbatch --export=DATASET=wikitext103,MODEL=gpt2 submit_clean_rerun.sh
# 12h limit is a safety margin for the WikiText-103 scale (two full raw-kNN grid
# searches + two DIME tunings + two head trainings on a ~381K-entry datastore);
# TinyStories scale should finish in well under an hour. Checkpoints to
# results_clean_rerun/{dataset}_{model}_summary.json after every config, so a
# timeout doesn't lose completed work.
set -e
echo "Clean rerun (${DATASET}/${MODEL}) job started: $(date)"
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

python -u $BASE/DIME/run_clean_rerun.py --dataset "$DATASET" --model "$MODEL"

echo "Clean rerun (${DATASET}/${MODEL}) job done: $(date)"
