"""Captures the FFN-input representation via a forward hook — verified against the
actual kNN-LM paper (Khandelwal et al. 2020, arxiv.org/abs/1911.00172, Section 5
"Key Function" ablation) rather than assumed. Their own ablation table (dev PPL,
WikiText-103, k=1024):

    Model output                       17.07
    Model output, layer normalized     17.01
    FFN input, after layer norm        16.06  <- their actual adopted choice (best)
    FFN input, before layer norm       17.06
    MHSA input, after layer norm       16.76
    MHSA input, before layer norm      17.14

"Model output"/"hidden_states[-1]" is what DIME's existing code has always used
(verified separately: identical to the tensor that produces out.logits) — NOT what
Khandelwal et al. actually use. Their adopted representation is the OUTPUT of ln_2
(post-attention-and-residual, post-layer-norm, immediately feeding the MLP) — this
module captures exactly that, via a forward hook (not a pre-hook) on ln_2.

GPT2Block's forward, per HuggingFace's implementation:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_output = self.attn(hidden_states)[0]
    hidden_states = attn_output + residual
    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)    <- OUTPUT of this call is captured here
    feed_forward_output = self.mlp(hidden_states)
    hidden_states = residual + feed_forward_output
"""
import torch


def extract_attn_sublayer(gpt, input_ids, layer_idx=-1):
    """gpt: FrozenGPT2. input_ids: [B, L]. layer_idx: which transformer block's
    FFN-input (post-ln_2) representation to capture (default -1 = last block,
    matching hidden_states[-1]'s depth).
    Returns (hidden_ffn [B,L,D], hidden_att [B,L,D], logits [B,L,V]) — hidden_ffn and
    logits come from gpt.forward() unchanged; hidden_att is the new captured tensor."""
    captured = {}

    def hook(module, args, output):
        captured["h_att"] = output.detach()

    block = gpt.model.transformer.h[layer_idx]
    handle = block.ln_2.register_forward_hook(hook)
    try:
        hidden_ffn, logits = gpt.forward(input_ids)
    finally:
        handle.remove()

    if "h_att" not in captured:
        raise RuntimeError("FFN-input hook never fired — check layer_idx / model structure")

    return hidden_ffn, captured["h_att"], logits
