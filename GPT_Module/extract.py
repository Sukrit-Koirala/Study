import torch
from helpers import shift_for_next_token, true_token_stats


def run_batch(gpt, chunks):
    batch_ids = torch.tensor(chunks, dtype=torch.long, device=gpt.device)
    hidden, logits = gpt.forward(batch_ids)

    h_pred, lgts_pred, y_target = shift_for_next_token(hidden, logits, batch_ids)
    p_true, nll = true_token_stats(lgts_pred, y_target)

    return h_pred, y_target, p_true, nll
