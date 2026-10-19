#!/bin/bash
#SBATCH --job-name=e1b3_cluster_agg
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1b3_cluster_agg-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e1b3_cluster_agg-%j.err

# Reuses the existing cache and re-derives DIME's clusters once (seed=42,
# deterministic, same as every prior E1 script) -- no GPT-2 re-extraction, no
# re-tuning. Experiment A runs on the full 77,089-query val set (cheap, just
# assignment lookups). Experiments B/C sample 3,000 queries (stratified across
# JS deciles) for the per-member distance computations, which are the real cost
# here -- budgeted higher than E1B.2 given the extra per-cluster-member work.
set -e
echo "E1B.3 job started: $(date)"
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

python -u $BASE/DIME/run_e1b3_cluster_vs_aggregation.py

echo "E1B.3 job done: $(date)"
