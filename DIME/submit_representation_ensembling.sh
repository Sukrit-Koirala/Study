#!/bin/bash
#SBATCH --job-name=rep_ensembling
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/rep_ensembling-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/rep_ensembling-%j.err

# Representation-ensembling control experiment (Xu et al. 2023 Section 4.1, adapted
# to DIME) on TinyStories/gpt2 — the smallest setting, approved as the first real run.
# Verified locally (CPU) before this ever reached the cluster: the attention-hook
# representation, the dual-extraction pipeline, and the learned-head training loop
# all ran end-to-end on a toy-scale slice with no crashes/NaNs. 6h limit is a safety
# margin, not an estimate — TinyStories scale should be much faster, dominated by
# GPT-2 forward passes (once) and head training (cheap, operates on cached features).
set -e
echo "Representation-ensembling experiment job started: $(date)"
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

python -u $BASE/DIME/run_representation_ensembling_experiment.py

echo "Representation-ensembling experiment job done: $(date)"
