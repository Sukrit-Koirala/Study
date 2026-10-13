#!/bin/bash
#SBATCH --job-name=gmm_deep_dive
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/gmm_deep_dive-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/gmm_deep_dive-%j.err

# This is a LOT of EM fitting (3 covariance families up to 300 iters each, a 3-point
# reg_covar sweep, 5-seed + 1 random-init stability check, all on the diagonal family)
# so the time limit is set to the cluster's known 24h max as a safety margin, not an
# estimate of real runtime. results/gmm_deep_dive.json checkpoints after every major
# section, so even a run that times out leaves usable partial results — check the
# JSON's "sections_completed" list to see how far it got.
set -e
echo "GMM deep-dive (TinyStories/gpt2) job started: $(date)"
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

python -c "import torch, transformers, datasets, numpy, pandas, tqdm, sklearn, scipy, matplotlib" 2>/dev/null || \
    pip install -r $BASE/GPT_Module/requirements.txt --quiet

export CUDA_VISIBLE_DEVICES=0

python -u $BASE/DIME/run_gmm_deep_dive.py

echo "GMM deep-dive job done: $(date)"
