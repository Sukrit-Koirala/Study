import torch
from helpers import shift_for_next_token, true_token_stats
import numpy as np
from helpers import predictive_entropy

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

def build_datastore_with_nll_from_chunks(gpt, chunks, batch_size=256):
    all_keys, all_values, all_nll = [], [], []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        h_pred, y_target, p_true, nll = run_batch(gpt, batch)
        B, L, D = h_pred.shape
        all_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
        all_values.append(y_target.reshape(B * L).cpu().numpy())
        all_nll.append(nll.reshape(B * L).cpu().numpy())
    keys = np.concatenate(all_keys, axis=0)
    values = np.concatenate(all_values, axis=0)
    nll_all = np.concatenate(all_nll, axis=0)
    return keys, values, nll_all


def run_batch_with_entropy(gpt, chunks):
    """Same as run_batch, plus GPT's own predictive entropy — kept as a separate
    function so run_batch's existing callers are untouched."""
    batch_ids = torch.tensor(chunks, dtype=torch.long, device=gpt.device)
    hidden, logits = gpt.forward(batch_ids)
    h_pred, lgts_pred, y_target = shift_for_next_token(hidden, logits, batch_ids)
    p_true, nll = true_token_stats(lgts_pred, y_target)
    entropy = predictive_entropy(lgts_pred)
    return h_pred, y_target, p_true, nll, entropy









