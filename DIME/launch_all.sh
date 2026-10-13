#!/bin/bash
# Submits every remaining experiment with the right dependencies. Run ON THE CLUSTER, once:
#     cd ~/ondemand/upload_me/Research/Study && git pull && cd DIME && bash launch_all.sh
#
# The six re-tunes are rerun with alpha=0 in the grid (earlier results used alpha >= 0.01); the earlier
# outputs are copied to *_noalpha0.json first. The WikiText-2 re-tunes took ~1-1.5 h.
#
# What is submitted (all independent unless noted):
#   1. raw re-tune, all 6 settings, alpha=0 allowed (old outputs are kept as *_noalpha0.json)
#   2. latency + bytes under one protocol, all 6 settings              -> Efficiency table
#   3. quality-vs-B sweep with re-tuned raw baselines at each B (4 settings, N <= 381K)
#   4. He et al. (2021) k-means-pruning baseline (4 settings; after that setting's re-tune)
#   5. 100-seed TinyStories test against the re-tuned raw baselines (after the TinyStories/gpt2 re-tune)
#   6. WikiText-103 ARTICLE-level split check: extraction -> Tier 1 -> re-tune (chained)
set -e
cd "$(dirname "$0")"

mkdir -p logs results
for f in results/*_raw_retune.json; do
    [ -e "$f" ] && { cp -n "$f" "${f%.json}_noalpha0.json" || true; }
done

# job NAME TIME DEP SCRIPT DATASET MODEL [NOARGS]  -> echoes the job id
job() {
    local name=$1 time=$2 dep=$3 script=$4 ds=$5 model=$6 noargs=$7
    local depflag=""
    [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
    sbatch --parsable --job-name="$name" --time="$time" $depflag \
        --export=SCRIPT="$script",DATASET="$ds",MODEL="$model",NOARGS="$noargs" submit_exp.sh
}

declare -A RT   # re-tune job ids

echo "== 1. raw re-tunes"
RT[tinystories_gpt2]=$(job retune_ts_gpt2   01:00:00 "" run_raw_retune.py tinystories gpt2)
RT[tinystories_gpt2-medium]=$(job retune_ts_med 03:00:00 "" run_raw_retune.py tinystories gpt2-medium)
RT[wikitext103_gpt2]=$(job retune_wt103_gpt2 03:00:00 "" run_raw_retune.py wikitext103 gpt2)
RT[wikitext103_gpt2-medium]=$(job retune_wt103_med 03:00:00 "" run_raw_retune.py wikitext103 gpt2-medium)
RT[wikitext2_gpt2]=$(job retune_wt2_gpt2   24:00:00 "" run_raw_retune.py wikitext2 gpt2)
RT[wikitext2_gpt2-medium]=$(job retune_wt2_med 24:00:00 "" run_raw_retune.py wikitext2 gpt2-medium)
echo "   ${RT[@]}"

echo "== 2. latency + bytes"
job lat_ts_gpt2   02:00:00 "" run_latency.py tinystories gpt2
job lat_ts_med    04:00:00 "" run_latency.py tinystories gpt2-medium
job lat_wt103_gpt2 04:00:00 "" run_latency.py wikitext103 gpt2
job lat_wt103_med 04:00:00 "" run_latency.py wikitext103 gpt2-medium
job lat_wt2_gpt2  16:00:00 "" run_latency.py wikitext2 gpt2
job lat_wt2_med   16:00:00 "" run_latency.py wikitext2 gpt2-medium

echo "== 3. B sweep (ratios 1000,300,30,10)"
job sweep_ts_gpt2  06:00:00 "" run_sweep_B.py tinystories gpt2
job sweep_ts_med   30:00:00 "" run_sweep_B.py tinystories gpt2-medium
job sweep_wt103_gpt2 30:00:00 "" run_sweep_B.py wikitext103 gpt2
job sweep_wt103_med 30:00:00 "" run_sweep_B.py wikitext103 gpt2-medium

echo "== 4. He et al. baseline"
job he_ts_gpt2   06:00:00 "${RT[tinystories_gpt2]}" run_he_baseline.py tinystories gpt2
job he_ts_med    12:00:00 "${RT[tinystories_gpt2-medium]}" run_he_baseline.py tinystories gpt2-medium
job he_wt103_gpt2 12:00:00 "${RT[wikitext103_gpt2]}" run_he_baseline.py wikitext103 gpt2
job he_wt103_med 12:00:00 "${RT[wikitext103_gpt2-medium]}" run_he_baseline.py wikitext103 gpt2-medium

echo "== 5. 100-seed test vs re-tuned raw baselines"
job seeds_retuned 20:00:00 "${RT[tinystories_gpt2]}" run_tinystories_multiseed_retuned.py tinystories gpt2 1

echo "== 6. WikiText-103 article-level split check"
X=$(sbatch --parsable --export=DATASET=wikitext103art,MODEL=gpt2,N_DS=3000,N_CT=500,N_VAL=500,BATCH_SIZE=256 ../GPT_Module/submit_extract.sh)
T1=$(sbatch --parsable --dependency=afterok:$X --export=DATASET=wikitext103art,MODEL=gpt2 submit_tier1.sh)
job retune_wt103art 03:00:00 "$T1" run_raw_retune.py wikitext103art gpt2

echo "All submitted. Watch with: squeue -u \$USER"
