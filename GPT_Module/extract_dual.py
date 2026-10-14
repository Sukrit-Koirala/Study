"""Dual-representation extraction — same shape/orchestration pattern as extract.py's
run_batch/build_datastore_from_chunks, but also returns the FFN-input (post-ln_2)
representation (attn_hook.py) alongside the existing hidden_states[-1] — verified
against Khandelwal et al. (2020)'s own ablation to be their actual best/adopted
choice, not "model output" (what hidden_states[-1] is). Kept as a separate module
(not folded into extract.py) since every one of extract.py's 7 existing callers
unpacks run_batch's return tuple positionally — this is additive, not a replacement.
"""
import torch
from helpers import shift_for_next_token, true_token_stats
from attn_hook import extract_attn_sublayer


def run_batch_dual(gpt, chunks, layer_idx=-1):
    """Like extract.py's run_batch, but also returns the attention-sublayer hidden
    state, shifted/chopped identically to h_pred (same drop-last-position rule)."""
    batch_ids = torch.tensor(chunks, dtype=torch.long, device=gpt.device)
    hidden_ffn, hidden_att, logits = extract_attn_sublayer(gpt, batch_ids, layer_idx=layer_idx)

    h_ffn_pred, lgts_pred, y_target = shift_for_next_token(hidden_ffn, logits, batch_ids)
    h_att_pred = hidden_att[:, :-1, :]  # same chop-last-position rule shift_for_next_token applies
    p_true, nll = true_token_stats(lgts_pred, y_target)

    return h_ffn_pred, h_att_pred, y_target, p_true, nll


def build_dual_datastore_from_chunks(gpt, chunks, layer_idx=-1, batch_size=256):
    """Returns ffn_keys, att_keys, values — [N,D] / [N,D] / [N], flattened over all
    chunks x positions, mirroring build_datastore_from_chunks exactly but for both
    representations at once (one forward pass per batch, not two)."""
    all_ffn, all_att, all_values = [], [], []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        h_ffn_pred, h_att_pred, y_target, p_true, nll = run_batch_dual(gpt, batch, layer_idx=layer_idx)
        B, L, D = h_ffn_pred.shape
        all_ffn.append(h_ffn_pred.reshape(B * L, D).cpu().numpy())
        all_att.append(h_att_pred.reshape(B * L, D).cpu().numpy())
        all_values.append(y_target.reshape(B * L).cpu().numpy())

    import numpy as np
    return (np.concatenate(all_ffn, axis=0),
            np.concatenate(all_att, axis=0),
            np.concatenate(all_values, axis=0))


def encode_dual_split(gpt, chunks_list, layer_idx=-1, batch_size=256):
    """Like run_gmm_experiment.py's encode_split, but returns both representations
    plus p_true (needed for every config's base-LM NLL and for mixing)."""
    import numpy as np
    all_ffn, all_att, all_targets, all_p_lm = [], [], [], []
    for i in range(0, len(chunks_list), batch_size):
        batch = chunks_list[i:i + batch_size]
        h_ffn_pred, h_att_pred, y_target, p_true, nll = run_batch_dual(gpt, batch, layer_idx=layer_idx)
        B, L, D = h_ffn_pred.shape
        all_ffn.append(h_ffn_pred.reshape(B * L, D).cpu().numpy())
        all_att.append(h_att_pred.reshape(B * L, D).cpu().numpy())
        all_targets.append(y_target.reshape(B * L).cpu().numpy())
        all_p_lm.append(p_true.reshape(B * L).cpu().numpy())
    return (np.concatenate(all_ffn, axis=0), np.concatenate(all_att, axis=0),
            np.concatenate(all_targets, axis=0), np.concatenate(all_p_lm, axis=0))
