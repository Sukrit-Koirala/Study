#!/bin/bash
#SBATCH --job-name=e2_1_beta_diagnostic
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2_1_beta_diagnostic-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2_1_beta_diagnostic-%j.err

# Reuses the existing cache and re-derives DIME's clusters (seed=42, deterministic).
# Main cost: one full S3 pass on val (~30-40 min, cached to results/.../cache/ for
# reruns). Every beta comes from one affine decomposition per budget, so the
# beta sweep itself is cheap (2 retention sets x 3 budgets x {val, controller_train}
# passes, each ~0.5-2 min at E2's measured throughput). 4h is a safety margin.
set -e
echo "E2.1 job started: $(date)"
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

python -u $BASE/DIME/run_e2_1_beta_diagnostic.py

echo "E2.1 job done: $(date)"
