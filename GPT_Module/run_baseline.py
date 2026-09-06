from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import run_gpt_baseline
from results_io import save_results
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
gpt = FrozenGPT2()

# Creating the split
chunks, chunk_story_ids = collect_chunks_split(
    dataset, gpt.tokenizer, seq_len=128,
    n_chunks_needed={"datastore": 350, "controller_train": 75, "val": 75}
)

val_chunks = chunks["val"]
per_chunk_nll = run_gpt_baseline(gpt, val_chunks, batch_size=256)
mean_nll = sum(per_chunk_nll) / len(per_chunk_nll)

results = {
    "phase": "B4_gpt_only_baseline",
    "model": "gpt2",
    "seq_len": 128,
    "split": "val",
    "n_chunks": len(val_chunks),
    "mean_nll": mean_nll,
    "per_chunk_nll": per_chunk_nll,
}

save_results("results/gpt_only_baseline.json", results)
print("mean NLL on val:", mean_nll)

