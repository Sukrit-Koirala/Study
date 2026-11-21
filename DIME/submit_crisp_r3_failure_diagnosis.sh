#!/bin/bash
#SBATCH --job-name=crisp_r3_failure_diagnosis
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r3_failure_diagnosis-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r3_failure_diagnosis-%j.err

# Self-contained: imports R1's functions directly rather than depending on
# R1/R2's persisted results (neither was present locally when this was
# written). Scoped to a small, deep diagnostic set per the spec's own
# "start small, expand only if justified" instruction: the 2 real failing
# clusters named in the spec (923, 1465) plus 2 successful clusters chosen
# by a predetermined rule, computed fresh.
#
# Reuses the existing cache and re-derives DIME's clusters once (seed=42).
# Primary cost: Parts A-F each run bounded per-cluster closed-form fits
# (5 anchor variants x 4 fitting methods x 3 objectives x 5 ranks, all on
# only 4 clusters) -- no full-val pass anywhere. The pilot (4 tiny synthetic
# clusters) completed in well under a minute. 3h is a generous safety
# margin, not an expected runtime. Checkpoints to crisp_r3_summary.json
# after every major section.
set -e
echo "CRISP-R3 job started: $(date)"
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

python -u $BASE/DIME/run_crisp_r3_failure_diagnosis.py

echo "CRISP-R3 job done: $(date)"
