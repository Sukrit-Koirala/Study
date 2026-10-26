#!/bin/bash
#SBATCH --job-name=e2tri1_anchor_conditioned
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2tri1_anchor_conditioned-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/e2tri1_anchor_conditioned-%j.err

# Reuses the existing cache (datastore/controller_train/val) and re-derives
# DIME's clusters once (seed=42, deterministic). Primary cost: the main
# ablation grid runs anchor_standalone_prob across 7 configs (A2 radial +
# A3-A7 diff K=1,2,4,8,16) at full val scale (77,089 queries x k=20 retrieved
# clusters each), each needing one CT-subsample pass too, plus 2 more full-val
# configs (Gaussian variant, random-projection baseline) at the chosen best K,
# plus the Section-8 distributional/preference-reversal/difficult-tail/
# cohesion analyses on bounded subsamples. No per-member distance loops like
# E1B.4/E1C.2/E2's S3 passes -- this is per-(query,cluster) vectorized over a
# cluster's distinct-token count, so each config should be noticeably faster
# than a full S3 pass, but there are ~9 full-val configs here vs. E2's single
# member-aware pass per budget, hence the larger 14h safety margin. Also
# writes stage3_e1_wikitext103-style checkpoints to e2tri1_summary.json after
# every major section.
set -e
echo "E2-TRI1 job started: $(date)"
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

python -u $BASE/DIME/run_e2tri1_anchor_conditioned_compression.py

echo "E2-TRI1 job done: $(date)"
