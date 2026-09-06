#!/bin/bash
#SBATCH --job-name=wikitext_medium_analysis
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/wikitext_medium_analysis-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/logs/wikitext_medium_analysis-%j.err

set -e
echo "WikiText-103 / gpt2-medium analysis job started: $(date)"
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

python $BASE/DIME/run_wikitext_medium_analysis.py

echo "WikiText-103 / gpt2-medium analysis job done: $(date)"
