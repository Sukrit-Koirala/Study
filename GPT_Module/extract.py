import torch
from helpers import shift_for_next_token, true_token_stats
import numpy as np

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


def build_datastore_from_chunks(gpt,chunks,batch_size=256):
    all_keys = []
    all_values = []
    for i in range(0,len(chunks),batch_size):
        batch = chunks[i:i + batch_size] #Just taking the current batch
        h_pred,y_target,p_true,nll = run_batch(gpt,batch)
        B, L, D = h_pred.shape #Batch size, length of seq, Dimension, Importnat thing to note is that GPT produces hidden states for every token state in that sentence, we don't really want that
        all_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
        all_values.append(y_target.reshape(B * L).cpu().numpy())

        #What B * L does, just multiples and simplifies, essentially collapsing two level of distinction down to a list

    keys = np.concatenate(all_keys, axis=0)
    values = np.concatenate(all_values, axis=0)
    return keys, values










