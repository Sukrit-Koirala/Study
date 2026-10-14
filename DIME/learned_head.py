"""Trainable output head for the representation-ensembling experiment — the first
gradient-trained neural network in this project (everything else is frozen GPT-2,
non-parametric clustering, or sklearn's own solvers). Deliberately minimal: a single
linear layer (no bias, matching P(y|c) = softmax(W h) exactly as specified), trained
with plain Adam + cross-entropy on CACHED, already-extracted hidden-state features —
GPT-2 itself never runs during head training, so this is cheap regardless of how
many epochs it takes.
"""
import copy
import numpy as np
import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h):
        return self.linear(h)  # logits, [*, vocab_size]

    def log_probs(self, h):
        return torch.log_softmax(self.linear(h), dim=-1)

    def memory_bytes(self):
        return self.linear.weight.numel() * self.linear.weight.element_size()


def train_head(train_keys, train_values, val_keys, val_values, vocab_size=50257,
               epochs=30, lr=1e-3, batch_size=4096, patience=5, seed=42, device="cpu",
               weight_decay=0.0, verbose=True):
    """train_keys/val_keys: [N,D] float32 numpy. train_values/val_values: [N] int
    token ids. Checkpoint selection: keeps the state_dict with the lowest VALIDATION
    cross-entropy seen across epochs (not the last epoch), early-stops after
    `patience` epochs with no val improvement — this is what Step 2 calls
    'validation-based checkpoint selection'."""
    torch.manual_seed(seed)
    d_model = train_keys.shape[1]
    head = LinearHead(d_model, vocab_size).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    train_h = torch.as_tensor(train_keys, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train_values, dtype=torch.long, device=device)
    val_h = torch.as_tensor(val_keys, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(val_values, dtype=torch.long, device=device)

    rng = np.random.default_rng(seed)
    n_train = train_h.shape[0]

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0
    history = []

    for epoch in range(epochs):
        head.train()
        perm = rng.permutation(n_train)
        epoch_loss_sum = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch_h = train_h[idx]
            batch_y = train_y[idx]
            logits = head(batch_h)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(idx)
        train_loss = epoch_loss_sum / n_train

        head.eval()
        with torch.no_grad():
            val_losses = []
            for start in range(0, val_h.shape[0], batch_size):
                vb_h = val_h[start:start + batch_size]
                vb_y = val_y[start:start + batch_size]
                val_logits = head(vb_h)
                val_losses.append(torch.nn.functional.cross_entropy(val_logits, vb_y, reduction="sum").item())
            val_loss = sum(val_losses) / val_h.shape[0]

        history.append({"epoch": epoch + 1, "train_nll": train_loss, "val_nll": val_loss})
        if verbose:
            print(f"    head epoch {epoch+1}: train_nll={train_loss:.4f}  val_nll={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print(f"    early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    head.load_state_dict(best_state)
    head.eval()
    return head, history, best_val_loss


def head_log_probs_for_targets(head, keys, values, device="cpu", batch_size=8192):
    """Per-position log P(true_token | h) under the trained head — [N] numpy array."""
    head.eval()
    all_logp = []
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            h = torch.as_tensor(keys[start:start + batch_size], dtype=torch.float32, device=device)
            y = torch.as_tensor(values[start:start + batch_size], dtype=torch.long, device=device)
            logp = head.log_probs(h)
            true_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            all_logp.append(true_logp.cpu().numpy())
    return np.concatenate(all_logp, axis=0)
