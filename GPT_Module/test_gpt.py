from gpt_class import FrozenGPT2
from helpers import shift_for_next_token, true_token_stats, collect_chunks
from datasets import load_dataset
from extract import run_batch
import random
from dataset_helpers import assign_story_split, collect_chunks_split

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)

gpt = FrozenGPT2()
chunks = collect_chunks(dataset, gpt.tokenizer, seq_len=16, n_chunks_needed=5)
input_ids = gpt.encode_text("the cat sat on the mat")
hidden, logits = gpt.forward(input_ids)
h_pred, lgts_pred, y_target = shift_for_next_token(hidden, logits, input_ids)
p_true, nll = true_token_stats(lgts_pred, y_target)
h_pred, y_target, p_true, nll = run_batch(gpt, chunks)
chunks, chunk_story_ids = collect_chunks_split(dataset, gpt.tokenizer, seq_len=16, n_chunks_needed={"datastore": 5, "controller_train": 5, "val": 5})

seen = {}
leaks = []
for split, ids_list in chunk_story_ids.items():
    for sid in set(ids_list):
        if sid in seen and seen[sid] != split:
            leaks.append((sid, seen[sid], split))
        seen[sid] = split

print("leaks found:", leaks)  # should be empty list


