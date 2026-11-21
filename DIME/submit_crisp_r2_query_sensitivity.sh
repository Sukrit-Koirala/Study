#!/bin/bash
#SBATCH --job-name=crisp_r2_query_sensitivity
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r2_query_sensitivity-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/crisp_r2_query_sensitivity-%j.err

# Self-contained: imports run_crisp_subspace_projection.py's own functions
# directly rather than depending on its persisted results (which were not
# present locally when this script was written). Reproduces R1's exact
# cluster-sampling/split methodology fresh under the same seed=42 clustering.
#
# Reuses the existing cache and re-derives DIME's clusters once (seed=42).
# Primary cost beyond R1: Experiment A refits CRISP across 6 temperatures
# (full rank grid at tau=2.0 and tau=21.4592, rank=8 only at the other 4),
# each refit being R1's own cheap closed-form per-cluster regression -- the
# per-(cluster,temperature) cost is dominated by one exact_evidence call over
# bounded fit/tune/test query counts, not a full-val pass. The optional
# cluster-sample expansion (triggered only if fewer than 3 of the original 20
# clusters are high-sensitivity) adds a bounded 150-candidate screening pass.
# 6h is a generous safety margin -- the pilot (40 tiny synthetic clusters)
# completed in well under a minute even with expansion forced. Checkpoints to
# crisp_r2_summary.json after every major section.
set -e
echo "CRISP-R2 job started: $(date)"
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

python -u $BASE/DIME/run_crisp_r2_query_sensitivity.py

echo "CRISP-R2 job done: $(date)"
