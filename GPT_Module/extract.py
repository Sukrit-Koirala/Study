import torch
from helpers import shift_for_next_token, true_token_stats


def run_batch(gpt, chunks):
    batch_ids = torch.tensor(chunks, dtype=torch.long, device=gpt.device)
    hidden, logits = gpt.forward(batch_ids)

    h_pred, lgts_pred, y_target = shift_for_next_token(hidden, logits, batch_ids)
    p_true, nll = true_token_stats(lgts_pred, y_target)

    return h_pred, y_target, p_true, nll


def run_gpt_baseline(gpt, chunks, batch_size=32):
    all_chunk_nll = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size] 
        h_pred, y_target, p_true, nll = run_batch(gpt, batch)
        chunk_means = nll.mean(dim=1)          # [B] — average NLL per chunk across positions
        all_chunk_nll.extend(chunk_means.tolist())
    return all_chunk_nll





