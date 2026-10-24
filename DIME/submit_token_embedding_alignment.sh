#!/bin/bash
#SBATCH --job-name=token_embedding_alignment
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/token_embedding_alignment-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/token_embedding_alignment-%j.err

# Reuses the existing cache (datastore/val only -- controller_train is not
# needed, this experiment is purely descriptive, nothing is tuned) and
# re-derives DIME's clusters once (seed=42, deterministic, identical to every
# prior E1/E2 script). Does NOT train NCA, change the hidden-state metric,
# recluster the datastore, or modify DIME-V1 -- see run_token_embedding_alignment.py's
# own docstring for the full constraint list. Unlike E1B.4/E1C.2/E2, there is
# no per-member distance loop here: Experiments A-D operate on <=20 centroids'
# worth of bounded (<=15-token) profile matrices and cosine similarities, so
# this should be far cheaper than any prior E1-series job; 2h is a generous
# safety margin, not an expected runtime. Writes only to
# results/stage3_token_embedding_alignment/ -- no existing E1/E2 outputs are
# read or modified.
set -e
echo "Token-embedding-alignment job started: $(date)"
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

python -u $BASE/DIME/run_token_embedding_alignment.py

echo "Token-embedding-alignment job done: $(date)"
