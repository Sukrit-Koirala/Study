"""E2-TRI1 -- Anchor-Conditioned Predictive Compression. Extends the E1/E2
series without modifying any existing output (writes only to
results/stage3_e2tri1_anchor_conditioned/). Exploratory proof-of-concept:
does NOT assume the method works, and does NOT commit to a final DIME-V2
architecture.

Central hypothesis: a small number of distances to EXISTING DIME centroids
(reused as anchors -- no new 768-dim reference vectors stored) can preserve
enough spatial information about a centroid's next-token labels to beat
DIME-V1's static P(y|c), without retraining GPT-2, changing the k-means
clustering/geometric routing, training NCA, or building a token graph.

Reuses exactly, not reimplemented: cache loading, seed=42 MiniBatchKMeans
clustering, matched config (k=20, tau=2.0, alpha=0.25), raw/DIME retrieval
(knn.py/mixing.py), e1_metrics.py's standalone/full-distribution/JS/coverage
helpers, and run_e2_information_allocation's build_cluster_to_members -- same
import-not-reimplement discipline E2.1 used for E2's own helpers.

No scikit-learn in any NEW computation here (anchor selection, feature
extraction, shrinkage, scoring, leave-one-out classification are all NumPy).
The only sklearn usage is the pre-existing MiniBatchKMeans clustering, reused
unchanged, exactly as every prior E1/E2 script does.

=====================================================================
Scoping decisions (stated up front, per this project's own established
practice of declaring cuts rather than silently narrowing scope):
=====================================================================
  - Anchor-selection-method comparison (nearest / random / diverse) is run
    at a single representative K=4 via Experiment A's leave-one-out
    nearest-mean classification accuracy, on ~15 largest-population
    clusters (a geometric diagnostic over DATASTORE members -- not a
    predictive-accuracy claim, so no train/eval leakage concern). Whichever
    method wins is carried forward into the full K-grid; this is decided
    from the comparison's own output, not assumed in advance.
  - Shrinkage strength N0 and the kernel bandwidth's base tau_z are each
    tuned ONCE (not per-K, not per-method) on a controller_train subsample
    at a representative K=4 diff-feature config, then held fixed across
    every other configuration -- the same "single global constant, not
    re-tuned per configuration" precedent E2 used for beta and E2.1 used for
    beta's grid. tau_z is rescaled by sqrt(K/4) for other K values to offset
    the mechanical growth of ||diff||^2 with feature dimensionality; this is
    a closed-form adaptation, not independent re-tuning.
  - Main ablation grid (Experiment D, Section 10.4's required table): A0
    (raw, existing), A1 (DIME-V1, existing), A2 (DIME+radial, 1-D), A3-A7
    (DIME+anchor-diff, K=1,2,4,8,16) -- all with the MEAN-ONLY kernel
    scoring variant, at FULL val scale (77,089 queries), matched config
    only (tuned-config secondary pass is explicitly NOT implemented, same
    precedent E2 itself stated).
  - The mean+diagonal-variance Gaussian variant and the dimension-matched
    random-projection baseline are each run ONLY at whichever K minimizes
    controller_train-subsample NLL among A3-A7 (2 extra full-val configs,
    not a full second grid).
  - Distributional metrics needing a full retrieval distribution (JS
    divergence, entropy, coverage) are computed for raw / DIME-V1 / the
    single best anchor-conditioned config only, on a bounded val subsample
    (not all 77,089) -- the true-token-only standalone probability (cheap,
    vectorized) is what the full-scale NLL grid actually needs.
  - Preference-reversal / margin analysis (Section 8): bounded to the same
    ~15 largest-population clusters x top-10 frequent tokens each.
  - Difficult-query tail: reuses the EXACT same worst-1/5/10% val-query
    groups (stable argsort of matched L_DIME - L_raw on val itself) that
    E1C.2/E2/E2.1 all used, not a new ranking.
  - Cohesion diagnostic (Experiment E): ~300 sampled (cluster, token) pairs
    with >=5 members, cohesion = mean pairwise raw hidden-state distance
    over <=50 sampled same-token pairs, correlated (Spearman) against each
    token's mean (anchor-config minus DIME-V1) standalone true-token-prob
    delta, aggregated across that token's occurrences.

Smoke testing: set E2TRI1_SMOKE=1 with E2TRI1_CACHE_DIR / E2TRI1_OUT_DIR /
E2TRI1_N_CLUSTERS pointing at a tiny synthetic cache before running -- same
convention E2.1 established. Never set it for the real run.
"""
import sys, os, json, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GPT_Module"))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats as scistats
from sklearn.cluster import MiniBatchKMeans

from cache_io import load_cache
from knn import build_datastore, query_knn_indices, softmax_np, mix_knn_and_lm
from mixing import mix_dime_and_lm, build_global_freq
from results_io import save_results
from e1_metrics import (raw_standalone_prob, dime_standalone_prob, raw_full_distribution,
                          dime_full_distribution, js_divergence_dict, retrieval_coverage,
                          distribution_shape_stats)
from run_e2_information_allocation import build_cluster_to_members

SMOKE = os.environ.get("E2TRI1_SMOKE") == "1"
CACHE_DIR = os.environ.get("E2TRI1_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("E2TRI1_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_e2tri1_anchor_conditioned"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("E2TRI1_N_CLUSTERS", "3811"))
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
SEED = 42
DIM, DTYPE_BYTES, VALUE_BYTES = 768, 4, 4

K_VALUES = [1, 2, 4, 8, 16]
ANCHOR_METHOD_COMPARE_K = 4
ANCHOR_METHODS = ["nearest", "random", "diverse"]
N_METHOD_COMPARE_CLUSTERS = 15
MIN_COUNT_FOR_VARIANCE = 2
N0_GRID = [1.0, 5.0, 20.0]
TAU_Z_FACTOR_GRID = [0.5, 1.0, 2.0]
N_CT_TUNE_SUBSAMPLE = 8000
N_SYNTH_PER_BLOB = 200
COHESION_TOKEN_SAMPLE = 300
COHESION_MIN_MEMBERS = 5
COHESION_PAIRS_PER_TOKEN = 50
PREF_PAIR_N_CENTROIDS = 15
PREF_PAIR_TOKENS_PER_CENTROID = 10
N_JS_SUBSAMPLE = 4000
N_BOOT = 2000
WORST_PCTS = [1, 5, 10]
VERBOSE_EVERY = 20000 if not SMOKE else 200


# ============================================================
# Anchor geometry
# ============================================================
def build_centroid_distance_matrix(dime_centroids):
    sq = np.sum(dime_centroids ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (dime_centroids @ dime_centroids.T)
    d2 = np.clip(d2, 0, None)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2)


def select_anchors_nearest(dist_matrix, c, k):
    return np.argsort(dist_matrix[c])[:k]


def select_anchors_random(n_clusters, c, k, rng):
    others = np.array([x for x in range(n_clusters) if x != c])
    return rng.choice(others, size=min(k, len(others)), replace=False)


def select_anchors_diverse(dist_matrix, c, k):
    min_dist = dist_matrix[c].copy()
    min_dist[c] = -np.inf
    selected = []
    for _ in range(k):
        best = int(np.argmax(min_dist))
        selected.append(best)
        min_dist = np.minimum(min_dist, dist_matrix[best])
        min_dist[best] = -np.inf
    return np.array(selected)


# ============================================================
# Feature extraction (generalized: diff / radial / random_proj)
# ============================================================
def compute_diff_features(H, c, anchor_idx, dime_centroids):
    """z_j(h) = ||h-a_j||^2 - ||h-a_0||^2, computed via the linear shortcut
    z_j(h) = -2 h.(a_j-a_0) + ||a_j||^2 - ||a_0||^2 (see module docstring's
    math; verified against the literal distance-difference formula before
    use -- see verify_feature_math()). Computed in float64: on larger-
    magnitude real hidden states, subtracting two close, large squared norms
    (||a_j||^2 - ||a_0||^2) loses real precision in float32 -- caught via a
    structured synthetic pilot where the two formulas diverged past a fixed
    absolute tolerance; the fix is float64 throughout this computation, not
    a looser tolerance."""
    H64 = H.astype(np.float64)
    a0 = dime_centroids[c].astype(np.float64)
    anchors = dime_centroids[anchor_idx].astype(np.float64)
    dirs = anchors - a0[None, :]
    const = np.sum(anchors ** 2, axis=1) - np.sum(a0 ** 2)
    return -2.0 * (H64 @ dirs.T) + const[None, :]


def compute_diff_features_literal(H, c, anchor_idx, dime_centroids):
    H64 = H.astype(np.float64)
    a0 = dime_centroids[c].astype(np.float64)
    d0 = np.sum((H64 - a0[None, :]) ** 2, axis=1)
    out = np.zeros((H.shape[0], len(anchor_idx)))
    for j, a_idx in enumerate(anchor_idx):
        aj = dime_centroids[a_idx].astype(np.float64)
        out[:, j] = np.sum((H64 - aj[None, :]) ** 2, axis=1) - d0
    return out


def compute_radial_feature(H, c, dime_centroids):
    a0 = dime_centroids[c].astype(np.float64)
    return np.sum((H.astype(np.float64) - a0[None, :]) ** 2, axis=1, keepdims=True)


def compute_random_proj_features(H, dirs):
    return H.astype(np.float64) @ dirs.T.astype(np.float64)


def verify_feature_math(ds_keys, dime_centroids, cluster_to_members, dist_matrix, n_check=3):
    """Spot-check the linear-shortcut diff-feature formula against the
    literal squared-distance-difference computation on a few real clusters
    -- the project's standing 'verify math before trusting it' discipline,
    applied to the derivation the spec's Section 3 itself invites."""
    checked = 0
    for c in range(len(dime_centroids)):
        members = cluster_to_members.get(c)
        if members is None or len(members) == 0:
            continue
        anchors = select_anchors_nearest(dist_matrix, c, 4)
        H = ds_keys[members[:20]]
        z_fast = compute_diff_features(H, c, anchors, dime_centroids)
        z_lit = compute_diff_features_literal(H, c, anchors, dime_centroids)
        scale = max(float(np.max(np.abs(z_lit))), 1e-8)
        max_rel_diff = float(np.max(np.abs(z_fast - z_lit))) / scale
        assert max_rel_diff < 1e-6, (f"cluster {c}: linear-shortcut vs literal diff feature mismatch, "
                                       f"max_rel_diff={max_rel_diff} (scale={scale})")
        checked += 1
        if checked >= n_check:
            break
    print(f"  verified linear-shortcut == literal distance-difference formula on {checked} clusters (max relative diff < 1e-6)")
    return checked


# ============================================================
# Experiment A -- synthetic Paris/Berlin diagnostic
# ============================================================
def synthetic_paris_berlin_test(rng):
    """Two label groups equidistant from a shared centroid but spatially
    separated. A single radial feature cannot distinguish them; 2 anchors
    placed along the separating direction should."""
    D = 8
    centroid = np.zeros(D)
    sep = np.zeros(D); sep[0] = 1.0
    paris = rng.normal(scale=0.1, size=(N_SYNTH_PER_BLOB, D)) + 2.0 * sep
    berlin = rng.normal(scale=0.1, size=(N_SYNTH_PER_BLOB, D)) - 2.0 * sep
    labels = np.array([0] * N_SYNTH_PER_BLOB + [1] * N_SYNTH_PER_BLOB)
    H = np.vstack([paris, berlin])

    r = np.sum((H - centroid[None, :]) ** 2, axis=1, keepdims=True)
    radial_acc = leave_one_out_accuracy(r, labels)

    anchor_a = centroid + 3.0 * sep
    anchor_b = centroid - 3.0 * sep
    dirs = np.stack([anchor_a - centroid, anchor_b - centroid])
    const = np.array([np.sum(anchor_a ** 2) - np.sum(centroid ** 2), np.sum(anchor_b ** 2) - np.sum(centroid ** 2)])
    z = -2.0 * (H @ dirs.T) + const[None, :]
    anchor_acc = leave_one_out_accuracy(z, labels)

    result = {"radial_only_accuracy": radial_acc, "two_anchor_accuracy": anchor_acc,
              "radial_distance_to_centroid_paris": float(np.mean(np.sum((paris - centroid) ** 2, axis=1))),
              "radial_distance_to_centroid_berlin": float(np.mean(np.sum((berlin - centroid) ** 2, axis=1))),
              "passed": bool(radial_acc is not None and anchor_acc is not None and anchor_acc > radial_acc + 0.1)}
    print(f"  synthetic Paris/Berlin: radial-only acc={radial_acc}, 2-anchor acc={anchor_acc}, passed={result['passed']}")
    return result


def leave_one_out_accuracy(feats, values):
    """Leave-one-out nearest-(token-)mean classification accuracy in feature
    space -- members of singleton groups are excluded from both numerator and
    denominator (can't fairly self-classify with an undefined leave-one-out
    mean)."""
    n = len(values)
    unique_vals, inv = np.unique(values, return_inverse=True)
    T = len(unique_vals)
    if T < 2 or n < 2:
        return None
    sums = np.zeros((T, feats.shape[1]))
    counts = np.zeros(T)
    for t in range(T):
        mask = inv == t
        sums[t] = feats[mask].sum(axis=0)
        counts[t] = mask.sum()
    base_means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    correct, valid_n = 0, 0
    for i in range(n):
        t_true = inv[i]
        if counts[t_true] <= 1:
            continue
        means = base_means.copy()
        means[t_true] = (sums[t_true] - feats[i]) / (counts[t_true] - 1)
        d = np.sum((means - feats[i][None, :]) ** 2, axis=1)
        pred = int(np.argmin(d))
        correct += int(pred == t_true)
        valid_n += 1
    return (correct / valid_n) if valid_n > 0 else None


# ============================================================
# Experiment B -- per-(cluster, token) anchor-signature statistics
# ============================================================
def compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, feature_fn, N0, var_floor):
    out = {}
    for c, members in cluster_to_members.items():
        if len(members) == 0:
            out[c] = None
            continue
        H = ds_keys[members]
        feats = feature_fn(H, c)
        mu_global = feats.mean(axis=0)
        var_global = feats.var(axis=0) if len(members) > 1 else np.full(feats.shape[1], var_floor)
        values = ds_values[members]
        tokens = {}
        for y in np.unique(values):
            mask = values == y
            n_y = int(mask.sum())
            f_y = feats[mask]
            mu_raw = f_y.mean(axis=0)
            lam = n_y / (n_y + N0)
            mu_shrunk = lam * mu_raw + (1 - lam) * mu_global
            if n_y >= MIN_COUNT_FOR_VARIANCE:
                var_raw = f_y.var(axis=0)
                var_shrunk = lam * var_raw + (1 - lam) * var_global + var_floor
            else:
                var_shrunk = var_global + var_floor
            tokens[int(y)] = {"n": n_y, "mu_raw": mu_raw, "mu_shrunk": mu_shrunk, "var_shrunk": var_shrunk}
        out[c] = {"mu_global": mu_global, "var_global": var_global, "n_members": len(members), "tokens": tokens}
    return out


def build_cluster_stat_arrays(cluster_token_stats, compressed_dists):
    out = {}
    for c, data in cluster_token_stats.items():
        if data is None or len(data["tokens"]) == 0:
            out[c] = None
            continue
        counter = compressed_dists[c]
        total = sum(counter.values())
        token_ids = np.array(sorted(data["tokens"].keys()))
        mu = np.stack([data["tokens"][int(y)]["mu_shrunk"] for y in token_ids])
        var = np.stack([data["tokens"][int(y)]["var_shrunk"] for y in token_ids])
        p_dime = np.array([(counter.get(int(y), 0) / total) if total > 0 else 0.0 for y in token_ids])
        tok_to_idx = {int(y): i for i, y in enumerate(token_ids)}
        out[c] = {"token_ids": token_ids, "mu": mu, "var": var, "p_dime": p_dime, "tok_to_idx": tok_to_idx}
    return out


# ============================================================
# Experiment C -- query-conditioned scoring
# ============================================================
def anchor_standalone_prob(query_keys, true_values, idx_dime, d_dime, tau, cluster_arrays, feature_fn,
                             use_variance, tau_z, verbose_every=VERBOSE_EVERY):
    N, k = idx_dime.shape
    w = softmax_np(-d_dime / tau, axis=1)
    p = np.zeros(N, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(N):
        q = query_keys[i:i + 1]
        true_tok = int(true_values[i])
        acc = 0.0
        for j in range(k):
            c = int(idx_dime[i, j])
            arr = cluster_arrays.get(c)
            if arr is None or true_tok not in arr["tok_to_idx"]:
                continue
            z_q = feature_fn(q, c)[0]
            diff = z_q[None, :] - arr["mu"]
            if use_variance:
                var = arr["var"]
                ls = np.log(arr["p_dime"]) - 0.5 * np.sum(diff ** 2 / var + np.log(var), axis=1)
            else:
                ls = np.log(arr["p_dime"]) - np.sum(diff ** 2, axis=1) / (2 * tau_z ** 2)
            m = ls.max()
            probs = np.exp(ls - m)
            probs /= probs.sum()
            p_c = float(probs[arr["tok_to_idx"][true_tok]])
            acc += float(w[i, j]) * p_c
        p[i] = acc
        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"      progress: {i+1}/{N}  ({(i+1)/elapsed:.1f} q/s, ETA {(N-i-1)/((i+1)/elapsed)/60:.1f} min)")
    return p


def anchor_full_distribution(query_keys, idx_dime, d_dime, tau, cluster_arrays, feature_fn, use_variance, tau_z):
    N, k = idx_dime.shape
    w = softmax_np(-d_dime / tau, axis=1)
    out = []
    for i in range(N):
        q = query_keys[i:i + 1]
        d = {}
        for j in range(k):
            c = int(idx_dime[i, j])
            arr = cluster_arrays.get(c)
            if arr is None or len(arr["token_ids"]) == 0:
                continue
            z_q = feature_fn(q, c)[0]
            diff = z_q[None, :] - arr["mu"]
            if use_variance:
                var = arr["var"]
                ls = np.log(arr["p_dime"]) - 0.5 * np.sum(diff ** 2 / var + np.log(var), axis=1)
            else:
                ls = np.log(arr["p_dime"]) - np.sum(diff ** 2, axis=1) / (2 * tau_z ** 2)
            m = ls.max()
            probs = np.exp(ls - m)
            probs /= probs.sum()
            wj = float(w[i, j])
            for tok, pr in zip(arr["token_ids"].tolist(), probs.tolist()):
                d[tok] = d.get(tok, 0.0) + wj * pr
        out.append(d)
    return out


def nll_from_standalone(p_anchor_true, p_lm_true, alpha):
    p_mixed = alpha * p_anchor_true + (1 - alpha) * p_lm_true
    return -np.log(p_mixed + 1e-12)


# ============================================================
# Compression accounting
# ============================================================
def storage_bytes_anchor_method(n_clusters, compressed_dists, dime_centroids_bytes_per, n_feat, use_variance,
                                  max_k_anchor_indices):
    import pickle
    centroid_bytes = n_clusters * dime_centroids_bytes_per
    counter_bytes = len(pickle.dumps(compressed_dists))
    n_cluster_token_entries = sum(len(c) for c in compressed_dists)
    per_entry_floats = n_feat * (2 if use_variance else 1)
    stat_bytes = n_cluster_token_entries * per_entry_floats * DTYPE_BYTES
    anchor_index_bytes = n_clusters * max_k_anchor_indices * 4
    total = centroid_bytes + counter_bytes + stat_bytes + anchor_index_bytes
    return {"centroid_bytes": int(centroid_bytes), "token_dist_bytes": int(counter_bytes),
            "anchor_stat_bytes": int(stat_bytes), "anchor_index_bytes": int(anchor_index_bytes),
            "n_cluster_token_entries": int(n_cluster_token_entries), "total_bytes": int(total)}


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.perf_counter()
    print(f"E2-TRI1 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}")
    rng = np.random.default_rng(SEED)

    print("Loading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"], ds["values"]
    ct_keys, ct_values, ct_p_lm = ct["keys"], ct["values"], ct["p_lm_true"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]

    print("Re-deriving DIME clusters (seed=42, deterministic, identical to every prior E1/E2 script)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys)
    dime_centroids = km.cluster_centers_.astype(np.float32)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))

    kM, tauM, alphaM = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]
    print("Querying val (matched config, full scale) -- raw and DIME retrieval, A0/A1 baselines...")
    d_raw_val, idx_raw_val = query_knn_indices(raw_index, val_keys, k=kM)
    d_dime_val, idx_dime_val = query_knn_indices(dime_index, val_keys, k=kM)
    r_raw_val = raw_values[idx_raw_val]
    r_dime_val = dime_values[idx_dime_val]
    p_raw_true_val = raw_standalone_prob(d_raw_val, r_raw_val, val_values, tau=tauM)
    p_dime_true_val = dime_standalone_prob(d_dime_val, r_dime_val, val_values, tau=tauM)
    nll_raw_val = nll_from_standalone(p_raw_true_val, val_p_lm, alphaM)
    nll_dime_val = nll_from_standalone(p_dime_true_val, val_p_lm, alphaM)
    delta_L_val = nll_dime_val - nll_raw_val

    print("Querying controller_train (matched config) -- for hyperparameter tuning only...")
    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=kM)
    ct_sub_idx = rng.choice(len(ct_values), size=min(N_CT_TUNE_SUBSAMPLE, len(ct_values)), replace=False)

    report = {"n_val": int(len(val_values)), "n_ct": int(len(ct_values)), "n_ds": int(len(ds_values)),
              "matched_config": MATCHED, "n_clusters": N_CLUSTERS, "seed": SEED, "smoke": SMOKE,
              "A0_raw_mean_nll": float(np.mean(nll_raw_val)), "A1_dime_mean_nll": float(np.mean(nll_dime_val))}
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    # ============================================================
    # Anchor geometry + math verification
    # ============================================================
    print("\nBuilding centroid-centroid distance matrix + verifying feature math...")
    dist_matrix = build_centroid_distance_matrix(dime_centroids)
    verify_feature_math(ds_keys, dime_centroids, cluster_to_members, dist_matrix)

    nearest_anchors = {c: select_anchors_nearest(dist_matrix, c, max(K_VALUES)) for c in range(N_CLUSTERS)}

    # ============================================================
    # Experiment A -- synthetic diagnostic + anchor-method comparison
    # ============================================================
    print("\nExperiment A: synthetic Paris/Berlin diagnostic...")
    synth_result = synthetic_paris_berlin_test(rng)
    report["experiment_A_synthetic"] = synth_result

    print("Experiment A: real anchor-method comparison (nearest/random/diverse) at K="
          f"{ANCHOR_METHOD_COMPARE_K} on {N_METHOD_COMPARE_CLUSTERS} largest-population clusters...")
    populations = np.array([sum(c.values()) for c in compressed_dists])
    compare_clusters = np.argsort(-populations)[:N_METHOD_COMPARE_CLUSTERS]
    compare_clusters = [int(c) for c in compare_clusters if len(cluster_to_members[int(c)]) > 1]

    method_results = {}
    for method in ANCHOR_METHODS:
        accs = []
        for c in compare_clusters:
            if method == "nearest":
                anchors = nearest_anchors[c][:ANCHOR_METHOD_COMPARE_K]
            elif method == "random":
                anchors = select_anchors_random(N_CLUSTERS, c, ANCHOR_METHOD_COMPARE_K, rng)
            else:
                anchors = select_anchors_diverse(dist_matrix, c, ANCHOR_METHOD_COMPARE_K)
            members = cluster_to_members[c]
            H = ds_keys[members]
            feats = compute_diff_features(H, c, anchors, dime_centroids)
            values = ds_values[members]
            acc = leave_one_out_accuracy(feats, values)
            if acc is not None:
                accs.append(acc)
        method_results[method] = {"mean_accuracy": float(np.mean(accs)) if accs else None, "n_clusters_evaluated": len(accs)}
        print(f"  {method}: mean leave-one-out accuracy = {method_results[method]['mean_accuracy']}")
    report["experiment_A_method_comparison"] = method_results

    valid_methods = {m: r["mean_accuracy"] for m, r in method_results.items() if r["mean_accuracy"] is not None}
    chosen_method = max(valid_methods, key=valid_methods.get) if valid_methods else "nearest"
    print(f"  chosen anchor-selection method for the full grid: '{chosen_method}' (highest leave-one-out accuracy)")
    report["chosen_anchor_method"] = chosen_method
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    if chosen_method == "nearest":
        anchor_assignment = nearest_anchors
    elif chosen_method == "random":
        anchor_assignment = {c: select_anchors_random(N_CLUSTERS, c, max(K_VALUES), rng) for c in range(N_CLUSTERS)}
    else:
        anchor_assignment = {c: select_anchors_diverse(dist_matrix, c, max(K_VALUES)) for c in range(N_CLUSTERS)}

    print("Experiment A: real diagnostic -- separability accuracy vs. K, chosen method...")
    acc_vs_k = {}
    for K in K_VALUES:
        accs = []
        for c in compare_clusters:
            anchors = anchor_assignment[c][:K]
            members = cluster_to_members[c]
            feats = compute_diff_features(ds_keys[members], c, anchors, dime_centroids)
            acc = leave_one_out_accuracy(feats, ds_values[members])
            if acc is not None:
                accs.append(acc)
        acc_vs_k[K] = float(np.mean(accs)) if accs else None
        print(f"  K={K}: mean separability accuracy = {acc_vs_k[K]}")
    report["experiment_A_accuracy_vs_K"] = acc_vs_k
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    fig, ax = plt.subplots(figsize=(6, 5))
    ks = [k for k in K_VALUES if acc_vs_k[k] is not None]
    ax.plot(ks, [acc_vs_k[k] for k in ks], marker="o")
    ax.set_xlabel("number of anchors (K)")
    ax.set_ylabel("leave-one-out within-cluster separability accuracy")
    ax.set_title(f"Experiment A: separability vs. K ({chosen_method} anchors)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "experiment_A_accuracy_vs_K.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Hyperparameter tuning (N0, tau_z base) on controller_train subsample, K=4
    # ============================================================
    print("\nTuning shrinkage N0 and base tau_z on a controller_train subsample (K=4, diff features)...")
    base_K = ANCHOR_METHOD_COMPARE_K
    feature_fn_base = lambda H, c: compute_diff_features(H, c, anchor_assignment[c][:base_K], dime_centroids)

    all_feats_sample = []
    for c in list(cluster_to_members.keys())[:200]:
        members = cluster_to_members[c]
        if len(members) == 0:
            continue
        all_feats_sample.append(feature_fn_base(ds_keys[members], c))
    z_std_ref = float(np.std(np.concatenate(all_feats_sample, axis=0))) if all_feats_sample else 1.0
    print(f"  empirical z std (K={base_K} reference sample): {z_std_ref:.4f}")

    best_cfg, best_nll = None, np.inf
    for N0 in N0_GRID:
        stats_base = compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, feature_fn_base, N0,
                                                    var_floor=1e-3 * max(z_std_ref ** 2, 1e-6))
        arrays_base = build_cluster_stat_arrays(stats_base, compressed_dists)
        for factor in TAU_Z_FACTOR_GRID:
            tau_z = factor * max(z_std_ref, 1e-6)
            p_ct = anchor_standalone_prob(ct_keys[ct_sub_idx], ct_values[ct_sub_idx], idx_dime_ct[ct_sub_idx],
                                            d_dime_ct[ct_sub_idx], tauM, arrays_base, feature_fn_base, False, tau_z,
                                            verbose_every=0)
            nll_ct = float(np.mean(nll_from_standalone(p_ct, ct_p_lm[ct_sub_idx], alphaM)))
            print(f"  N0={N0}, tau_z_factor={factor} (tau_z={tau_z:.4f}): CT subsample mean NLL = {nll_ct:.4f}")
            if nll_ct < best_nll:
                best_nll, best_cfg = nll_ct, {"N0": N0, "tau_z_factor": factor, "tau_z_base": tau_z}

    N0_CHOSEN, TAU_Z_FACTOR_CHOSEN, TAU_Z_BASE = best_cfg["N0"], best_cfg["tau_z_factor"], best_cfg["tau_z_base"]
    VAR_FLOOR = 1e-3 * max(z_std_ref ** 2, 1e-6)
    print(f"  chosen: N0={N0_CHOSEN}, tau_z_factor={TAU_Z_FACTOR_CHOSEN} (CT subsample mean NLL={best_nll:.4f})")
    report["hyperparameter_tuning"] = {"z_std_reference_K": base_K, "z_std_reference": z_std_ref,
                                         "N0_grid": N0_GRID, "tau_z_factor_grid": TAU_Z_FACTOR_GRID,
                                         "chosen_N0": N0_CHOSEN, "chosen_tau_z_factor": TAU_Z_FACTOR_CHOSEN,
                                         "chosen_tau_z_base": TAU_Z_BASE, "var_floor": VAR_FLOOR,
                                         "ct_subsample_mean_nll_at_choice": best_nll}
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    # ============================================================
    # Experiment D -- main ablation grid A2-A7 (mean-only kernel), full val scale
    # ============================================================
    print("\nExperiment D: main ablation grid (A2 radial + A3-A7 anchor-diff, mean-only kernel), full val...")
    grid_results = {"A0_raw": {"mean_nll": float(np.mean(nll_raw_val))},
                     "A1_dime": {"mean_nll": float(np.mean(nll_dime_val))}}

    radial_fn = lambda H, c: compute_radial_feature(H, c, dime_centroids)
    radial_stats = compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, radial_fn, N0_CHOSEN, VAR_FLOOR)
    radial_arrays = build_cluster_stat_arrays(radial_stats, compressed_dists)
    radial_z_std = float(np.std(np.concatenate([radial_fn(ds_keys[cluster_to_members[c]], c)
                                                  for c in list(cluster_to_members.keys())[:200]
                                                  if len(cluster_to_members[c]) > 0], axis=0)))
    tau_z_radial = TAU_Z_FACTOR_CHOSEN * max(radial_z_std, 1e-6)
    p_radial = anchor_standalone_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, radial_arrays,
                                        radial_fn, False, tau_z_radial)
    nll_radial = nll_from_standalone(p_radial, val_p_lm, alphaM)
    grid_results["A2_radial"] = {"mean_nll": float(np.mean(nll_radial)), "tau_z": tau_z_radial}
    print(f"  A2 (radial): mean NLL = {grid_results['A2_radial']['mean_nll']:.4f}")
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), {**report, "experiment_D_grid": grid_results})

    diff_arrays_by_K = {}
    diff_p_by_K = {}
    ct_nll_by_K = {}
    for K in K_VALUES:
        fn_K = lambda H, c, K=K: compute_diff_features(H, c, anchor_assignment[c][:K], dime_centroids)
        stats_K = compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, fn_K, N0_CHOSEN, VAR_FLOOR)
        arrays_K = build_cluster_stat_arrays(stats_K, compressed_dists)
        tau_z_K = TAU_Z_BASE * np.sqrt(K / base_K)
        p_val_K = anchor_standalone_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, arrays_K, fn_K,
                                           False, tau_z_K)
        nll_val_K = nll_from_standalone(p_val_K, val_p_lm, alphaM)
        p_ct_K = anchor_standalone_prob(ct_keys[ct_sub_idx], ct_values[ct_sub_idx], idx_dime_ct[ct_sub_idx],
                                          d_dime_ct[ct_sub_idx], tauM, arrays_K, fn_K, False, tau_z_K, verbose_every=0)
        nll_ct_K = float(np.mean(nll_from_standalone(p_ct_K, ct_p_lm[ct_sub_idx], alphaM)))
        grid_results[f"A{3 + K_VALUES.index(K)}_diff_K{K}"] = {"mean_nll": float(np.mean(nll_val_K)), "tau_z": tau_z_K,
                                                                  "ct_subsample_mean_nll": nll_ct_K}
        diff_arrays_by_K[K] = arrays_K
        diff_p_by_K[K] = p_val_K
        ct_nll_by_K[K] = nll_ct_K
        print(f"  A{3 + K_VALUES.index(K)} (K={K}): val mean NLL = {np.mean(nll_val_K):.4f}, CT subsample mean NLL = {nll_ct_K:.4f}")
        save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), {**report, "experiment_D_grid": grid_results})

    best_K = min(ct_nll_by_K, key=ct_nll_by_K.get)
    print(f"  best K by CT-subsample NLL: K={best_K}")
    report["best_K_by_ct_subsample"] = best_K

    # ============================================================
    # Secondary configs at best K: Gaussian (mean+var) and random-projection baseline
    # ============================================================
    print(f"\nSecondary configs at best K={best_K}: Gaussian (mean+variance) variant + random-projection baseline...")
    fn_best = lambda H, c: compute_diff_features(H, c, anchor_assignment[c][:best_K], dime_centroids)
    stats_best = compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, fn_best, N0_CHOSEN, VAR_FLOOR)
    arrays_best = build_cluster_stat_arrays(stats_best, compressed_dists)
    tau_z_best = TAU_Z_BASE * np.sqrt(best_K / base_K)

    p_gauss = anchor_standalone_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, arrays_best, fn_best,
                                       True, tau_z_best)
    nll_gauss = nll_from_standalone(p_gauss, val_p_lm, alphaM)
    grid_results[f"A_gaussian_K{best_K}"] = {"mean_nll": float(np.mean(nll_gauss))}
    print(f"  Gaussian variant (K={best_K}): val mean NLL = {np.mean(nll_gauss):.4f}")

    rand_dirs = rng.normal(size=(best_K, DIM)).astype(np.float32)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_fn = lambda H, c: compute_random_proj_features(H, rand_dirs)
    rand_stats = compute_cluster_token_stats(ds_keys, ds_values, cluster_to_members, rand_fn, N0_CHOSEN, VAR_FLOOR)
    rand_arrays = build_cluster_stat_arrays(rand_stats, compressed_dists)
    rand_z_std = float(np.std(np.concatenate([rand_fn(ds_keys[cluster_to_members[c]], c)
                                                for c in list(cluster_to_members.keys())[:200]
                                                if len(cluster_to_members[c]) > 0], axis=0)))
    tau_z_rand = TAU_Z_FACTOR_CHOSEN * max(rand_z_std, 1e-6)
    p_rand = anchor_standalone_prob(val_keys, val_values, idx_dime_val, d_dime_val, tauM, rand_arrays, rand_fn,
                                      False, tau_z_rand)
    nll_rand = nll_from_standalone(p_rand, val_p_lm, alphaM)
    grid_results[f"A_random_proj_K{best_K}"] = {"mean_nll": float(np.mean(nll_rand))}
    print(f"  random-projection baseline (K={best_K}): val mean NLL = {np.mean(nll_rand):.4f}")

    report["experiment_D_grid"] = grid_results
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels_plot = ["A0_raw", "A1_dime", "A2_radial"] + [f"A{3+i}_diff_K{K}" for i, K in enumerate(K_VALUES)] + \
                  [f"A_gaussian_K{best_K}", f"A_random_proj_K{best_K}"]
    vals_plot = [grid_results[l]["mean_nll"] for l in labels_plot]
    ax.bar(range(len(labels_plot)), vals_plot)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean interpolated NLL (matched config, val)")
    ax.set_title("Experiment D: ablation grid")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "experiment_D_ablation_grid.png"), dpi=130)
    plt.close(fig)

    # ============================================================
    # Section 8 -- distributional metrics (JS, entropy, coverage) on a val subsample
    # ============================================================
    print(f"\nDistributional metrics (JS divergence, entropy, coverage) on a {N_JS_SUBSAMPLE}-query val subsample "
          f"for raw / DIME-V1 / best anchor config (K={best_K})...")
    js_idx = rng.choice(len(val_values), size=min(N_JS_SUBSAMPLE, len(val_values)), replace=False)
    raw_dists_sub = raw_full_distribution(d_raw_val[js_idx], r_raw_val[js_idx], tau=tauM)
    dime_dists_sub = dime_full_distribution(d_dime_val[js_idx], r_dime_val[js_idx], tau=tauM)
    anchor_dists_sub = anchor_full_distribution(val_keys[js_idx], idx_dime_val[js_idx], d_dime_val[js_idx], tauM,
                                                  arrays_best, fn_best, False, tau_z_best)
    js_dime = [js_divergence_dict(r, d) for r, d in zip(raw_dists_sub, dime_dists_sub)]
    js_anchor = [js_divergence_dict(r, d) for r, d in zip(raw_dists_sub, anchor_dists_sub)]
    shape_dime = distribution_shape_stats(dime_dists_sub)
    shape_anchor = distribution_shape_stats(anchor_dists_sub)
    cov_raw = retrieval_coverage(p_raw_true_val[js_idx])
    cov_dime = retrieval_coverage(p_dime_true_val[js_idx])
    cov_anchor = retrieval_coverage(anchor_standalone_prob(val_keys[js_idx], val_values[js_idx], idx_dime_val[js_idx],
                                                             d_dime_val[js_idx], tauM, arrays_best, fn_best, False,
                                                             tau_z_best, verbose_every=0))
    dist_metrics = {
        "n_subsample": int(len(js_idx)),
        "mean_js_dime_vs_raw": float(np.mean(js_dime)), "mean_js_anchor_vs_raw": float(np.mean(js_anchor)),
        "mean_entropy_dime": float(np.mean(shape_dime["entropy"])),
        "mean_entropy_anchor": float(np.mean(shape_anchor["entropy"])),
        "correct_token_coverage_raw": cov_raw, "correct_token_coverage_dime": cov_dime,
        "correct_token_coverage_anchor": cov_anchor,
    }
    print(f"  mean JS(DIME-V1, raw)={dist_metrics['mean_js_dime_vs_raw']:.4f}, "
          f"mean JS(anchor, raw)={dist_metrics['mean_js_anchor_vs_raw']:.4f}")
    report["distributional_metrics"] = dist_metrics
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    # ============================================================
    # Preference-reversal / margin analysis (Section 8)
    # ============================================================
    print(f"\nPreference-reversal/margin analysis on {len(compare_clusters)} clusters x top-"
          f"{PREF_PAIR_TOKENS_PER_CENTROID} tokens each...")
    pref_examples = []
    for c in compare_clusters[:PREF_PAIR_N_CENTROIDS]:
        top_tokens = [t for t, _ in compressed_dists[c].most_common(PREF_PAIR_TOKENS_PER_CENTROID)]
        region_mask = idx_dime_val[:, 0] == c
        region_idx = np.where(region_mask)[0]
        if len(region_idx) < 5 or len(top_tokens) < 2:
            continue
        raw_region = raw_full_distribution(d_raw_val[region_idx], r_raw_val[region_idx], tau=tauM)
        anchor_region = anchor_full_distribution(val_keys[region_idx], idx_dime_val[region_idx], d_dime_val[region_idx],
                                                    tauM, arrays_best, fn_best, False, tau_z_best)
        for i in range(len(top_tokens)):
            for j in range(i + 1, min(i + 2, len(top_tokens))):
                yi, yj = top_tokens[i], top_tokens[j]
                raw_margin = np.array([r.get(yi, 0.0) - r.get(yj, 0.0) for r in raw_region])
                anchor_margin = np.array([a.get(yi, 0.0) - a.get(yj, 0.0) for a in anchor_region])
                p_dime_i = compressed_dists[c].get(yi, 0) / max(sum(compressed_dists[c].values()), 1)
                p_dime_j = compressed_dists[c].get(yj, 0) / max(sum(compressed_dists[c].values()), 1)
                dime_margin = p_dime_i - p_dime_j
                reversal_rate = float(np.mean(np.sign(raw_margin) != np.sign(anchor_margin)))
                pref_examples.append({
                    "centroid": int(c), "token_i": int(yi), "token_j": int(yj),
                    "mean_raw_margin": float(np.mean(raw_margin)), "mean_anchor_margin": float(np.mean(anchor_margin)),
                    "dime_static_margin": float(dime_margin), "sign_reversal_rate_anchor_vs_raw": reversal_rate,
                })
    report["preference_reversal_examples"] = pref_examples[:40]
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)
    print(f"  recorded {len(pref_examples)} token-pair preference comparisons")

    # ============================================================
    # Difficult-query tail (reuses E1C.2/E2 worst-decile convention on val)
    # ============================================================
    print("\nDifficult-query tail: worst-1/5/10% val queries by matched delta_L (DIME - raw)...")
    order = np.argsort(-delta_L_val, kind="stable")
    tail_results = {}
    for pct in WORST_PCTS:
        n_worst = max(1, int(len(order) * pct / 100))
        worst_idx = order[:n_worst]
        tail_results[pct] = {
            "n_queries": int(n_worst),
            "mean_nll_raw": float(np.mean(nll_raw_val[worst_idx])),
            "mean_nll_dime": float(np.mean(nll_dime_val[worst_idx])),
            "mean_nll_anchor_best": float(np.mean(nll_from_standalone(
                anchor_standalone_prob(val_keys[worst_idx], val_values[worst_idx], idx_dime_val[worst_idx],
                                         d_dime_val[worst_idx], tauM, arrays_best, fn_best, False, tau_z_best,
                                         verbose_every=0),
                val_p_lm[worst_idx], alphaM))),
        }
        print(f"  worst-{pct}%: raw={tail_results[pct]['mean_nll_raw']:.4f}, "
              f"dime={tail_results[pct]['mean_nll_dime']:.4f}, anchor={tail_results[pct]['mean_nll_anchor_best']:.4f}")
    report["difficult_query_tail"] = tail_results
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    # ============================================================
    # Experiment E -- cohesion diagnostic
    # ============================================================
    print(f"\nExperiment E: cohesion diagnostic on up to {COHESION_TOKEN_SAMPLE} (cluster,token) pairs...")
    eligible_pairs = []
    for c in compare_clusters:
        members = cluster_to_members[c]
        values_c = ds_values[members]
        for y, n_y in Counter(values_c.tolist()).items():
            if n_y >= COHESION_MIN_MEMBERS:
                eligible_pairs.append((c, y))
    rng.shuffle(eligible_pairs)
    eligible_pairs = eligible_pairs[:COHESION_TOKEN_SAMPLE]

    cohesion_records = []
    token_delta_accum = {}
    for c, y in eligible_pairs:
        members = cluster_to_members[c]
        values_c = ds_values[members]
        tok_members = members[values_c == y]
        n = len(tok_members)
        n_pairs = min(COHESION_PAIRS_PER_TOKEN, n * (n - 1) // 2)
        if n_pairs <= 0:
            continue
        idx_a = rng.integers(0, n, size=n_pairs)
        idx_b = rng.integers(0, n, size=n_pairs)
        valid = idx_a != idx_b
        if not valid.any():
            continue
        pa, pb = tok_members[idx_a[valid]], tok_members[idx_b[valid]]
        dists = np.linalg.norm(ds_keys[pa] - ds_keys[pb], axis=1)
        cohesion = float(np.mean(dists))

        region_mask = idx_dime_val[:, 0] == c
        region_idx = np.where(region_mask)[0]
        true_is_y = val_values[region_idx] == y
        if true_is_y.sum() >= 3:
            p_anchor_y = anchor_standalone_prob(val_keys[region_idx[true_is_y]], val_values[region_idx[true_is_y]],
                                                  idx_dime_val[region_idx[true_is_y]], d_dime_val[region_idx[true_is_y]],
                                                  tauM, arrays_best, fn_best, False, tau_z_best, verbose_every=0)
            p_dime_y = dime_standalone_prob(d_dime_val[region_idx[true_is_y]],
                                              dime_values[idx_dime_val[region_idx[true_is_y]]],
                                              val_values[region_idx[true_is_y]], tau=tauM)
            delta = float(np.mean(p_anchor_y - p_dime_y))
            token_delta_accum.setdefault(int(y), []).append(delta)
            cohesion_records.append({"centroid": int(c), "token": int(y), "n_members": n, "cohesion_mean_pair_dist": cohesion,
                                       "anchor_minus_dime_true_prob_delta": delta})

    token_mean_delta = {y: float(np.mean(ds)) for y, ds in token_delta_accum.items()}
    cohesion_by_record = [(r["cohesion_mean_pair_dist"], token_mean_delta.get(r["token"], r["anchor_minus_dime_true_prob_delta"]))
                           for r in cohesion_records]
    if len(cohesion_by_record) >= 5:
        cohesions = np.array([x[0] for x in cohesion_by_record])
        deltas = np.array([x[1] for x in cohesion_by_record])
        rho, pval = scistats.spearmanr(cohesions, deltas)
        cohesion_corr = {"spearman_rho": float(rho), "p_value": float(pval), "n": len(cohesion_by_record)}
    else:
        cohesion_corr = {"spearman_rho": None, "p_value": None, "n": len(cohesion_by_record),
                           "reason": "fewer than 5 eligible records"}
    print(f"  cohesion vs. (anchor-minus-DIME) true-token-prob delta: {cohesion_corr}")
    report["experiment_E_cohesion"] = {"n_records": len(cohesion_records), "correlation": cohesion_corr,
                                         "example_records": cohesion_records[:20]}
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    # ============================================================
    # Compression efficiency
    # ============================================================
    print("\nCompression efficiency accounting...")
    bytes_mean_only_bestK = storage_bytes_anchor_method(N_CLUSTERS, compressed_dists, DIM * DTYPE_BYTES,
                                                           best_K, False, max(K_VALUES))
    bytes_mean_var_bestK = storage_bytes_anchor_method(N_CLUSTERS, compressed_dists, DIM * DTYPE_BYTES,
                                                         best_K, True, max(K_VALUES))
    dime_v1_bytes = N_CLUSTERS * DIM * DTYPE_BYTES + bytes_mean_only_bestK["token_dist_bytes"]
    raw_bytes = len(ds_values) * (DIM * DTYPE_BYTES + VALUE_BYTES)
    compression_efficiency = {"dime_v1_bytes": int(dime_v1_bytes), "raw_kNN_bytes": int(raw_bytes),
                                f"anchor_mean_only_K{best_K}_bytes": bytes_mean_only_bestK,
                                f"anchor_mean_var_K{best_K}_bytes": bytes_mean_var_bestK,
                                "mean_only_overhead_vs_dime_v1_pct": float(
                                    (bytes_mean_only_bestK["total_bytes"] - dime_v1_bytes) / dime_v1_bytes * 100),
                                "mean_only_compression_ratio_vs_raw": float(raw_bytes / bytes_mean_only_bestK["total_bytes"])}
    print(f"  DIME-V1 bytes={dime_v1_bytes}, anchor mean-only (K={best_K}) bytes={bytes_mean_only_bestK['total_bytes']}, "
          f"raw bytes={raw_bytes}")
    report["compression_efficiency"] = compression_efficiency
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    # ============================================================
    # Failure analysis (Section 9) -- derived from already-computed quantities
    # ============================================================
    print("\nFailure analysis...")
    low_support_frac = float(np.mean([st["n"] < MIN_COUNT_FOR_VARIANCE for data in stats_best.values() if data is not None
                                        for st in data["tokens"].values()]))
    overconfidence_cases = int(np.sum((nll_from_standalone(diff_p_by_K[best_K], val_p_lm, alphaM)
                                         > nll_dime_val + 0.1)))
    failure_analysis = {
        "fraction_cluster_token_pairs_low_support": low_support_frac,
        "n_queries_anchor_worse_than_dime_by_0.1nats_or_more": overconfidence_cases,
        "n_queries_total": int(len(val_values)),
        "random_projection_vs_anchor_mean_nll_diff": float(np.mean(nll_rand) - np.mean(nll_from_standalone(
            diff_p_by_K[best_K], val_p_lm, alphaM))),
    }
    report["failure_analysis"] = failure_analysis
    print(f"  {failure_analysis}")
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)

    # ============================================================
    # Markdown report
    # ============================================================
    print("\nGenerating report.md...")
    best_diff_nll = grid_results[f"A{3+K_VALUES.index(best_K)}_diff_K{best_K}"]["mean_nll"]
    lines = [
        "# DIME Stage 3 -- E2-TRI1: Anchor-Conditioned Predictive Compression",
        "",
        f"Matched config (k={MATCHED['k']}, tau={MATCHED['tau']}, alpha={MATCHED['alpha']}), "
        f"{len(val_values)} val queries, {N_CLUSTERS} clusters. Chosen anchor-selection method: "
        f"**{chosen_method}** (Experiment A comparison: {method_results}).",
        "",
        "## Experiment D -- ablation grid (mean NLL, matched config, val)",
    ]
    for label, d in grid_results.items():
        lines.append(f"- {label}: {d['mean_nll']:.4f}")
    lines += [
        "",
        f"Best K by controller_train-subsample NLL: **K={best_K}** (mean-only diff NLL={best_diff_nll:.4f} on val).",
        "",
        "## Distributional metrics (val subsample)",
        f"Mean JS(DIME-V1, raw) = {dist_metrics['mean_js_dime_vs_raw']:.4f}; "
        f"Mean JS(anchor-conditioned, raw) = {dist_metrics['mean_js_anchor_vs_raw']:.4f}.",
        "",
        "## Difficult-query tail",
    ]
    for pct, d in tail_results.items():
        lines.append(f"- worst-{pct}%: raw={d['mean_nll_raw']:.4f}, DIME-V1={d['mean_nll_dime']:.4f}, "
                      f"anchor-conditioned={d['mean_nll_anchor_best']:.4f}")
    lines += [
        "",
        "## Experiment E -- cohesion diagnostic",
        f"Spearman correlation (cohesion vs. anchor-minus-DIME true-token-probability delta): {cohesion_corr}",
        "",
        "## Compression efficiency",
        f"DIME-V1: {dime_v1_bytes} bytes. Anchor mean-only (K={best_K}): "
        f"{bytes_mean_only_bestK['total_bytes']} bytes "
        f"({compression_efficiency['mean_only_overhead_vs_dime_v1_pct']:.2f}% overhead vs. DIME-V1, "
        f"{compression_efficiency['mean_only_compression_ratio_vs_raw']:.1f}x compression vs. raw kNN).",
        "",
        "## Answers to the research questions (from the measurements above only)",
        f"**Q1** (can anchor-relative signatures distinguish token groups raw frequency can't?): "
        f"synthetic test passed={synth_result['passed']}; real separability accuracy vs. K: {acc_vs_k}.",
        f"**Q2** (how much query-conditioned information is recoverable?): see the ablation grid above -- "
        f"compare A1 (DIME-V1) against A3-A7 (anchor-diff) directly.",
        "**Q3** (is mean sufficient, or is variance/multiple prototypes needed?): compare "
        f"A{3+K_VALUES.index(best_K)}_diff_K{best_K} ({best_diff_nll:.4f}) against "
        f"A_gaussian_K{best_K} ({grid_results[f'A_gaussian_K{best_K}']['mean_nll']:.4f}).",
        f"**Q4** (are existing DIME centroids good enough as anchors?): method comparison above "
        f"({method_results}); random-projection baseline NLL = {grid_results[f'A_random_proj_K{best_K}']['mean_nll']:.4f} "
        f"vs. anchor-structured K={best_K} NLL = {best_diff_nll:.4f}.",
        "**Q5** (does this improve NLL/distribution fidelity under practical budgets?): see the ablation grid, "
        "distributional metrics, and compression-efficiency sections above.",
        "**Q6** (does it help the difficult-query tail?): see the difficult-query tail section above.",
        "**Q7** (does the evidence justify further development?): judge directly from Q1-Q6's actual "
        "measured numbers above -- not pre-decided here.",
        "",
        "## Limitations",
        f"- Anchor-selection-method comparison done at a single K={ANCHOR_METHOD_COMPARE_K} on "
        f"{len(compare_clusters)} largest-population clusters, not the full grid.",
        f"- Shrinkage N0 and base tau_z tuned once on a {len(ct_sub_idx)}-query controller_train "
        f"subsample at K={base_K}, held fixed elsewhere (tau_z rescaled by sqrt(K/{base_K})).",
        "- Gaussian variant and random-projection baseline evaluated only at the single best K, not the full grid.",
        f"- Distributional metrics (JS, entropy) computed on a {len(js_idx)}-query val subsample, not full val.",
        "- Tuned-config secondary pass not implemented, consistent with E2's own stated precedent.",
        "- Correlation (e.g. cohesion vs. prediction delta) does not imply causation.",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["experiment_A", "hyperparameter_tuning", "experiment_D", "distributional_metrics",
                                      "preference_reversal", "difficult_query_tail", "experiment_E",
                                      "compression_efficiency", "failure_analysis", "report.md"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "e2tri1_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("No existing E1/E2/E2.1 outputs were modified.")


if __name__ == "__main__":
    main()
