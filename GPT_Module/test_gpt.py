from gpt_class import FrozenGPT2
from helpers import shift_for_next_token, true_token_stats, collect_chunks
from datasets import load_dataset
from extract import run_batch



dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)

gpt = FrozenGPT2()
chunks = collect_chunks(dataset, gpt.tokenizer, seq_len=16, n_chunks_needed=5)

print("num chunks:", len(chunks))
for c in chunks:
    print(len(c), c)

print("device:", gpt.device, type(gpt.device))

input_ids = gpt.encode_text("the cat sat on the mat")
print("input_ids:", input_ids, input_ids.shape)

hidden, logits = gpt.forward(input_ids)
print("hidden.shape:", hidden.shape)
print("logits.shape:", logits.shape)

h_pred, lgts_pred, y_target = shift_for_next_token(hidden, logits, input_ids)
p_true, nll = true_token_stats(lgts_pred, y_target)

print("\n--- per-position NLL ---")
for pos in range(y_target.shape[1]):
    tok = gpt.tokenizer.decode([y_target[0, pos].item()])
    print(f"{tok!r:>10}  p_true={p_true[0, pos].item():.4f}  nll={nll[0, pos].item():.3f}")



h_pred, y_target, p_true, nll = run_batch(gpt, chunks)

print("\n--- batch results ---")
print("h_pred.shape:", h_pred.shape)
print("nll.shape:", nll.shape)
print("mean NLL across all chunks:", nll.mean().item())

print("\n--- per-position NLL for chunk 0 ---")
for pos in range(y_target.shape[1]):
    tok = gpt.tokenizer.decode([y_target[0, pos].item()])
    print(f"{tok!r:>10}  nll={nll[0, pos].item():.3f}")

