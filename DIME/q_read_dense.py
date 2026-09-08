import torch
import numpy as np

def counters_to_dense(compressed_dists, top_k=64):
    """Converts a list of Counters into dense, padded arrays — same representation
    Track A's own state files use. Enables vectorized/GPU lookup instead of
    per-query Python dict access."""
    B = len(compressed_dists)
    token_ids = np.zeros((B, top_k), dtype=np.int64)
    token_counts = np.zeros((B, top_k), dtype=np.float32)
    total_counts = np.zeros(B, dtype=np.float32)
    for i, counter in enumerate(compressed_dists):
        total_counts[i] = sum(counter.values())
        for j, (tok, cnt) in enumerate(counter.most_common(top_k)):
            token_ids[i, j] = tok
            token_counts[i, j] = cnt
    return token_ids, token_counts, total_counts


def dense_p_state(retrieved_cluster_idx, true_targets, token_ids, token_counts, total_counts,
                   beta=0.0, global_freq=None, device="cuda"):
    """retrieved_cluster_idx: [N, k] cluster indices (from query_knn_indices).
    true_targets: [N]. token_ids/token_counts: [B, top_k] dense (from counters_to_dense).
    total_counts: [B]. Returns p_state: [N, k] — no Python loop over queries/neighbors."""
    idx_t = torch.as_tensor(retrieved_cluster_idx, dtype=torch.long, device=device)
    true_t = torch.as_tensor(true_targets, dtype=torch.long, device=device)
    tok_ids_t = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    tok_cnts_t = torch.as_tensor(token_counts, dtype=torch.float32, device=device)
    tot_t = torch.as_tensor(total_counts, dtype=torch.float32, device=device)

    retrieved_ids = tok_ids_t[idx_t]        # [N, k, top_k]
    retrieved_cnts = tok_cnts_t[idx_t]      # [N, k, top_k]
    retrieved_tot = tot_t[idx_t]            # [N, k]

    match_mask = (retrieved_ids == true_t.view(-1, 1, 1))
    matched_count = (retrieved_cnts * match_mask).sum(dim=-1)   # [N, k]

    if beta > 0 and global_freq is not None:
        global_freq_t = torch.as_tensor(global_freq, dtype=torch.float32, device=device)
        global_p = global_freq_t[true_t].view(-1, 1)
        p_state = (matched_count + beta * global_p) / (retrieved_tot + beta + 1e-12)
    else:
        p_state = matched_count / (retrieved_tot + 1e-12)

    return p_state  # [N, k]


def dense_mix_nll(distances, p_state, p_lm_true, tau=1.0, alpha=0.25, device="cuda"):
    """distances: [N, k] numpy. p_state: [N, k] tensor. p_lm_true: [N] numpy."""
    dist_t = torch.as_tensor(distances, dtype=torch.float32, device=device)
    weights = torch.softmax(-dist_t / tau, dim=-1)
    p_knn_true = (weights * p_state).sum(dim=-1)
    p_lm_t = torch.as_tensor(p_lm_true, dtype=torch.float32, device=device)
    p_mixed = alpha * p_knn_true + (1 - alpha) * p_lm_t
    return -torch.log(p_mixed.clamp(min=1e-12))


def compute_reward_matrix_dense(retrieved_cluster_idx_max, distances_max, true_targets, p_lm_true,
                                  token_ids, token_counts, total_counts, actions, global_freq=None, device="cuda"):
    """retrieved_cluster_idx_max, distances_max: [N, k_max] — computed ONCE via
    query_knn_indices at the grid's largest k. Returns reward: [N, A]."""
    N = len(true_targets)
    A = len(actions)
    nll_gpt = -np.log(p_lm_true + 1e-12)
    reward = np.zeros((N, A), dtype=np.float64)

    for ai, action in enumerate(actions):
        k = action["k"]
        if k == 0:
            continue  # gpt_only: reward stays 0
        idx_k = retrieved_cluster_idx_max[:, :k]
        dist_k = distances_max[:, :k]
        p_state = dense_p_state(idx_k, true_targets, token_ids, token_counts, total_counts,
                                  beta=action["beta"], global_freq=global_freq, device=device)
        nll_action = dense_mix_nll(dist_k, p_state, p_lm_true, tau=action["tau"], alpha=action["alpha"], device=device)
        reward[:, ai] = nll_gpt - nll_action.cpu().numpy()

    return reward


def dense_state_entropy_purity(cluster_idx, token_counts, total_counts, device="cuda"):
    """cluster_idx: [N, k]. token_counts: [B, top_k]. total_counts: [B].
    Returns entropy, purity: [N, k] — vectorized, reusing the same dense arrays
    compute_reward_matrix_dense already uses."""
    idx_t = torch.as_tensor(cluster_idx, dtype=torch.long, device=device)
    cnts_t = torch.as_tensor(token_counts, dtype=torch.float32, device=device)
    tot_t = torch.as_tensor(total_counts, dtype=torch.float32, device=device)

    retrieved_cnts = cnts_t[idx_t]              # [N, k, top_k]
    retrieved_tot = tot_t[idx_t].unsqueeze(-1)   # [N, k, 1]

    probs = retrieved_cnts / (retrieved_tot + 1e-12)
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)  # padded zero slots contribute exactly 0
    purity = probs.max(dim=-1).values

    return entropy, purity  # both [N, k]


def build_obs_features(gpt_entropy, distances, cluster_idx, token_counts, total_counts, device="cuda"):
    """gpt_entropy: [N] (from predictive_entropy). distances, cluster_idx: [N, k_max]
    (from query_knn_indices, k_max >= 8). Returns obs: [N, 11]."""
    entropy, purity = dense_state_entropy_purity(cluster_idx, token_counts, total_counts, device=device)
    entropy = entropy.cpu().numpy()
    purity = purity.cpu().numpy()

    obs = np.stack([
        gpt_entropy,
        distances[:, 0],
        distances[:, :4].mean(axis=1),
        distances[:, :8].mean(axis=1),
        distances[:, 1] - distances[:, 0],
        entropy[:, 0],
        entropy[:, :4].mean(axis=1),
        purity[:, 0],
        purity[:, :4].mean(axis=1),
        np.log1p(total_counts[cluster_idx[:, 0]]),
        np.log1p(total_counts[cluster_idx[:, :4]]).mean(axis=1),
    ], axis=1)
    return obs  # [N, 11]


# Since we will be using a MLPRegressor that uses (X,Y) we need to reshape X
def build_qsa_dataset(obs, action_features, reward_matrix):
    """obs: [N, 11]. action_features: [A, 4] (from build_action_features).
    reward_matrix: [N, A] (from compute_reward_matrix_dense).
    Returns X: [N*A, 15], y: [N*A] — one row per (query, action) pair."""
    N, A = reward_matrix.shape
    obs_tiled = np.repeat(obs, A, axis=0)                 # [N*A, 11]
    action_tiled = np.tile(action_features, (N, 1))       # [N*A, 4]
    X = np.concatenate([obs_tiled, action_tiled], axis=1)  # [N*A, 15]
    y = reward_matrix.reshape(-1)                          # [N*A]
    return X, y


