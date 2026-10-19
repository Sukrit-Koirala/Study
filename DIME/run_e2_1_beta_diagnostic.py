"""E2.1 -- Final Diagnostic: Information Quantity vs. Utilization. Extends E2
without modifying any E1-series or E2 output (writes only to
results/stage3_e2_1_beta_diagnostic/).

Only beta changes. Everything else is E2's selective_sensitivity configuration,
reproduced with E2's OWN imported functions (allocate_selective,
select_members_nearest_centroid, hybrid_standalone_prob), not re-implemented:
  - same cache, same MiniBatchKMeans(seed=42) clusters/centroids/Counters
  - same matched retrieval (k=20, tau=2.0) and interpolation alpha=0.25
  - same sensitivity priority (positive controller_train delta_L, top-1 cluster
    attribution), same nearest-centroid selection, same max 50 members/cluster
  - same E1C matched worst-query groups (stable argsort of matched L_DIME - L_raw)

E2 never persisted its retained member IDs (only e2_summary.json exists), so
they are RECONSTRUCTED here deterministically and verified before anything else
is trusted: exact vector counts (3811/19056/38112), structural checks (every
retained ID belongs to its cluster, is among that cluster's nearest-to-centroid
members, <=50 per cluster, no duplicates), and beta=0.25 reproducing E2's
full-precision mean NLL and worst-10% repair fraction. The reconstructed IDs
are saved this time (retained_members/*.npz).

Key structural fact, used for efficiency AND interpretation: E2's rule is
affine in beta for every query,
    p_beta(y*|q) = A(q) + beta * B(q)
    A(q) = sum_j w_j P_cj^DIME(y*)                                  (= DIME)
    B(q) = sum_{j: R_cj nonempty} w_j [P_cj^member(y*|q) - P_cj^DIME(y*)]
so one pass per budget gives every beta exactly (cross-checked against E2's own
per-beta loop on a 3,000-query subset). Two consequences, stated up front:
  (1) for any beta>0, WHICH queries gain or lose correct-token probability vs
      DIME is fixed by sign(B), i.e. by the retained members, not by beta --
      beta only scales the size of each change;
  (2) per-query interpolated NLL is -log(affine in beta), hence convex, so mean
      NLL is convex in beta with a single minimum on [0,1].

beta selection (never on val):
  PRIMARY (exact E2 replay): grid argmin of controller_train NLL with E2's exact
    retained sets. LIMITATION: E2's sensitivity ranking was built from the SAME
    controller_train queries, so this is not independent validation.
  SECONDARY (split, reconstructed sets): controller_train split 50/50 by
    127-position chunk (seed 42); ranking rebuilt on half R only, beta tuned on
    half T only. These retention sets are NOT E2's sets -- kept separate and
    reported as such, together with their overlap with E2's sets.
  EXPLORATORY: val-optimal beta, reported as an oracle sensitivity only.

Uncertainty: paired bootstrap over 127-position chunks (cache layout: each
seq_len=128 extraction chunk contributes 127 contiguous positions -- checked by
divisibility of every split size). Chunks are the finest sequence-level unit
the cache persists; article identity is not persisted, so adjacent chunks of
one article are treated as independent -- stated limitation.

Distribution properties (entropy, JS vs raw kNN, top-1 agreement) are
STANDALONE retrieval-distribution metrics: only p_lm(y*) is cached, not the
full LM distribution, so no interpolated full distribution exists to measure.
They are computed on a fixed stratified sample (2,000 worst-10% + 3,000
remaining-90% queries, seed 42).

Smoke testing: E21_SMOKE=1 with E21_CACHE_DIR / E21_OUT_DIR / E21_N_CLUSTERS
pointing at a tiny synthetic cache turns the E2/E1C.2 reference assertions into
printed warnings. Never set it for the real run.
"""
import sys, os, json, csv, time, hashlib
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GPT_Module"))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import minimize_scalar
from sklearn.cluster import MiniBatchKMeans

from cache_io import load_cache
from knn import build_datastore, query_knn_indices, softmax_np, mix_knn_and_lm
from mixing import mix_dime_and_lm
from e1_metrics import dime_standalone_prob, raw_full_distribution, dime_full_distribution, js_divergence_dict
# E2's own deterministic code -- imported, not re-implemented, so the retention
# sets are produced by exactly the procedure E2 ran.
from run_e2_information_allocation import (build_cluster_to_members, select_members_nearest_centroid,
                                           allocate_selective, hybrid_standalone_prob, ratio_with_stability_guard)

SMOKE = os.environ.get("E21_SMOKE") == "1"
CACHE_DIR = os.environ.get("E21_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("E21_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_e2_1_beta_diagnostic"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
RETAINED_DIR = os.path.join(OUT_DIR, "retained_members")
CACHE_SUB = os.path.join(OUT_DIR, "cache")
E2_SUMMARY = os.path.join(SCRIPT_DIR, "results", "stage3_e2_information_allocation", "e2_summary.json")
E1C2_SUMMARY = os.path.join(SCRIPT_DIR, "results", "stage3_e1c2_selective_retention", "e1c2_summary.json")

N_CLUSTERS = int(os.environ.get("E21_N_CLUSTERS", "3811"))
MATCHED = {"k": 20, "tau": 2.0, "alpha": 0.25}
BUDGET_FRACTIONS = [0.01, 0.05, 0.10]
E2_EXPECTED_COUNTS = {0.01: 3811, 0.05: 19056, 0.10: 38112}
E2_MAX_PER_CLUSTER = 50  # allocate_selective's default, as E2 called it
E2_BETA = 0.25
BETA_GRID = [0.0, 0.10, 0.25, 0.40, 0.50, 0.75, 1.00]
BETA_FINE = [float(b) for b in np.round(np.linspace(0.0, 1.0, 41), 3)]  # 0.025 steps, curves only
WORST_PCTS = [1, 5, 10]
CHUNK_LEN = 127
N_BOOT = 2000
N_DIRECT_CHECK = 3000
N_DIST_WORST, N_DIST_REST = 2000, 3000
REF_TOL = 1e-9

# Mechanical decision aid, fixed BEFORE seeing E2.1 results. The written
# interpretation, not these flags, is the conclusion.
OUTCOME_MIN_RS3_GAIN_PP = 5.0      # selected beta must add >= 5 points of R_S3 over beta=0.25 ...
OUTCOME_NEW_SEVERE_FRAC = 0.005    # ... and Outcome C if it adds new severe failures in >0.5% of the remaining 90%

COLORS = {0.01: "#2a78d6", 0.05: "#eb6834", 0.10: "#1baf7a"}
INK, INK_MUTED, REF_DIME, REF_S3, REF_RAW = "#2b2b29", "#6b6a64", "#8a8984", "#2b2b29", "#b5b4ad"


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def require(cond, msg):
    if cond:
        return
    if SMOKE:
        print(f"  [SMOKE] check would FAIL on real data: {msg}")
        return
    raise AssertionError(msg)


def bkey(bf):
    return f"{bf * 100:g}pct"


def fmt_pct(x, scale=1.0):
    return "n/a (unstable denominator)" if x is None else f"{x * scale:.2f}%"


def interp_nll(p, p_lm, alpha):
    return -np.log(alpha * p + (1 - alpha) * p_lm + 1e-12)  # same epsilon convention as E2/E1B.4


def dump_json(path, obj):
    def default(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ============================================================
# The affine decomposition of E2's prediction rule
# ============================================================

def affine_decomposition(keys, values, idx_dime, d_dime, tau, retained, ds_keys, ds_values,
                         compressed_dists, cluster_totals, record_pairs=False, verbose_every=20000, label=""):
    """Per query: A = DIME true-token prob, B = member-minus-DIME term over
    clusters with retained members, w_ret = retrieval weight on those clusters.
    Same arithmetic as E2's hybrid_standalone_prob, just with beta factored out."""
    N, k = idx_dime.shape
    w_all = softmax_np(-d_dime / tau, axis=1)
    A, B, w_ret = np.zeros(N), np.zeros(N), np.zeros(N)
    pq, pc, pw, pdiff = [], [], [], []
    t0 = time.perf_counter()
    for i in range(N):
        q_key, true_tok = keys[i], int(values[i])
        a = b = wr = 0.0
        for j in range(k):
            c = idx_dime[i, j]
            w_c = float(w_all[i, j])
            total = cluster_totals[c]
            p_dime_c = (compressed_dists[c].get(true_tok, 0) / total) if total > 0 else 0.0
            a += w_c * p_dime_c
            r = retained.get(c)
            if r is None or len(r) == 0:
                continue  # E2's fallback: beta_effective = 0 for this cluster
            d_m = np.linalg.norm(ds_keys[r] - q_key[None, :], axis=1)
            w_m = softmax_np(-d_m / tau, axis=-1)
            match = ds_values[r] == true_tok
            p_member_c = float(w_m[match].sum()) if match.any() else 0.0
            b += w_c * (p_member_c - p_dime_c)
            wr += w_c
            if record_pairs:
                pq.append(i); pc.append(int(c)); pw.append(w_c); pdiff.append(p_member_c - p_dime_c)
        A[i], B[i], w_ret[i] = a, b, wr
        if verbose_every and (i + 1) % verbose_every == 0:
            el = time.perf_counter() - t0
            print(f"      [{label}] {i+1}/{N}  ({(i+1)/el:.0f} q/s, ETA {(N-i-1)/((i+1)/el)/60:.1f} min)")
    out = {"A": A, "B": B, "w_ret": w_ret}
    if record_pairs:
        out["pairs"] = {"q": np.asarray(pq, dtype=np.int64), "c": np.asarray(pc, dtype=np.int64),
                        "w": np.asarray(pw), "diff": np.asarray(pdiff)}
    return out


def member_delta_distribution(keys, idx_dime, d_dime, tau, retained, ds_keys, ds_values, compressed_dists, cluster_totals):
    """Full-vocab version of B: per query {token: sum_j w_j (P^member_cj - P^DIME_cj)}
    over clusters with retained members. P_beta = P_DIME + beta * delta."""
    N, k = idx_dime.shape
    w_all = softmax_np(-d_dime / tau, axis=1)
    out = []
    for i in range(N):
        delta = {}
        for j in range(k):
            c = idx_dime[i, j]
            r = retained.get(c)
            if r is None or len(r) == 0:
                continue
            w_c, total = float(w_all[i, j]), cluster_totals[c]
            d_m = np.linalg.norm(ds_keys[r] - keys[i][None, :], axis=1)
            w_m = softmax_np(-d_m / tau, axis=-1)
            for tok, wm in zip(ds_values[r].tolist(), w_m.tolist()):
                delta[tok] = delta.get(tok, 0.0) + w_c * wm
            for tok, cnt in compressed_dists[c].items():
                delta[tok] = delta.get(tok, 0.0) - w_c * cnt / total
        out.append(delta)
    return out


def align_sample(D_list, R_list):
    aligned = []
    for D, R in zip(D_list, R_list):
        toks = np.array(sorted(set(D) | set(R)), dtype=np.int64)
        pos = {int(t): n for n, t in enumerate(toks)}
        aligned.append((toks, pos, np.array([D.get(int(t), 0.0) for t in toks]), np.array([R.get(int(t), 0.0) for t in toks])))
    return aligned


def align_delta(aligned, delta_list):
    out = []
    for (toks, pos, d, r), De in zip(aligned, delta_list):
        v = np.zeros(len(toks))
        for t, x in De.items():
            v[pos[t]] += x  # KeyError here would mean a member token outside its cluster's Counter
        out.append(v)
    return out


def distribution_metrics(aligned, deltas, beta, true_tokens):
    n = len(aligned)
    ent, js, agree = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    mass_dev, min_p, p_true = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, (toks, pos, d, r) in enumerate(aligned):
        p = d + beta * deltas[i] if deltas is not None else d
        min_p[i] = p.min()
        mass_dev[i] = p.sum() - d.sum()  # member term must move mass, never create/destroy it
        pc = np.clip(p, 0.0, None)
        nz, nzr = pc > 0, r > 0
        ent[i] = float(-(pc[nz] * np.log(pc[nz])).sum())
        m = 0.5 * (pc + r)
        js[i] = 0.5 * float((pc[nz] * np.log(pc[nz] / m[nz])).sum()) + 0.5 * float((r[nzr] * np.log(r[nzr] / m[nzr])).sum())
        agree[i] = toks[np.argmax(pc)] == toks[np.argmax(r)]
        t = pos.get(int(true_tokens[i]))
        p_true[i] = p[t] if t is not None else 0.0
    return {"entropy": ent, "js_vs_raw": js, "top1_agree_raw": agree, "mass_dev": mass_dev, "min_p": min_p, "p_true": p_true}


# ============================================================
# E2's sensitivity ranking, optionally restricted to a subset of controller_train
# ============================================================

def sensitivity_priority(idx_dime_ct, L_dime_ct, L_raw_ct, n_clusters, query_mask=None):
    """Line-for-line E2 (run_e2_information_allocation.py, 'Cluster priority
    rankings'), with an optional query mask for the split analysis."""
    cluster_sens = np.zeros(n_clusters)
    cluster_hit_counts = np.zeros(n_clusters)
    with np.errstate(invalid="ignore"):
        delta_L_ct = L_dime_ct - L_raw_ct
    rows = range(len(L_dime_ct)) if query_mask is None else np.where(query_mask)[0]
    for i in rows:
        top_c = idx_dime_ct[i, 0]
        cluster_sens[top_c] += max(0.0, delta_L_ct[i])
        cluster_hit_counts[top_c] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        cluster_sens_mean = np.where(cluster_hit_counts > 0, cluster_sens / np.maximum(cluster_hit_counts, 1), 0.0)
    return np.argsort(-cluster_sens_mean).tolist()


def retained_ids(retained):
    ids = [np.asarray(v, dtype=np.int64) for v in retained.values() if len(v) > 0]
    return np.sort(np.concatenate(ids)) if ids else np.zeros(0, dtype=np.int64)


def retained_hash(retained):
    return hashlib.sha256(retained_ids(retained).tobytes()).hexdigest()


def verify_retained(retained, assignment, cluster_to_members, centroids, ds_keys, max_per):
    ids = retained_ids(retained)
    n_total = int(sum(len(v) for v in retained.values()))
    require(len(np.unique(ids)) == n_total, "duplicate member IDs in a retention set")
    n_not_nearest = 0
    for c, r in retained.items():
        r = np.asarray(r)
        require(bool((assignment[r] == c).all()), f"cluster {c}: retained member assigned to a different cluster")
        require(len(r) <= max_per, f"cluster {c}: {len(r)} > {max_per} retained members")
        members = cluster_to_members[c]
        if len(r) < len(members):
            d_all = np.linalg.norm(ds_keys[members] - centroids[c][None, :], axis=1)
            d_ret = np.linalg.norm(ds_keys[r] - centroids[c][None, :], axis=1)
            d_rest = d_all[~np.isin(members, r)]
            if d_ret.max() > d_rest.min() + 1e-6:
                n_not_nearest += 1
    require(n_not_nearest == 0, f"{n_not_nearest} clusters' retained members are not their nearest-to-centroid members")
    return {"n_retained_vectors": n_total, "n_clusters_with_retained": int(sum(1 for v in retained.values() if len(v) > 0)),
            "max_per_cluster_observed": int(max((len(v) for v in retained.values()), default=0)),
            "sha256_sorted_ids": retained_hash(retained), "n_clusters_not_nearest_centroid": n_not_nearest}


# ============================================================
# Chunk-level paired bootstrap
# ============================================================

class ChunkBootstrap:
    """Resamples whole 127-position chunks with replacement; every statistic is
    a ratio of per-chunk sums, so all systems share the same draws (paired)."""

    def __init__(self, chunk_of_query, n_boot=N_BOOT, seed=42):
        self.g = chunk_of_query
        self.G = int(chunk_of_query.max()) + 1
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.G, size=(n_boot, self.G))
        self.W = np.stack([np.bincount(d, minlength=self.G) for d in draws]).astype(np.float64)

    def _sums(self, x, mask):
        x = np.asarray(x, dtype=np.float64)
        if mask is not None:
            x = np.where(mask, x, 0.0)
        return np.bincount(self.g, weights=x, minlength=self.G)

    def mean(self, x, mask=None):
        s, c = self._sums(x, mask), self._sums(np.ones(len(x)), mask)
        boot = (self.W @ s) / np.maximum(self.W @ c, 1e-300)
        xm = np.asarray(x)[mask] if mask is not None else np.asarray(x)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"point": float(s.sum() / c.sum()), "ci_95_low": float(lo), "ci_95_high": float(hi),
                "chunk_boot_se": float(boot.std()), "iid_se": float(xm.std(ddof=1) / np.sqrt(len(xm)))}

    def ratio(self, num, den, mask=None, min_mean_den=None):
        sn, sd, c = self._sums(num, mask), self._sums(den, mask), self._sums(np.ones(len(num)), mask)
        bn, bd, bc = self.W @ sn, self.W @ sd, self.W @ c
        ok = (bd / bc > min_mean_den) if min_mean_den is not None else (bd > 0)
        out = {"point": float(sn.sum() / sd.sum()) if sd.sum() != 0 else None, "n_stable_draws": int(ok.sum())}
        if ok.sum() > 0.5 * len(bn):
            lo, hi = np.percentile(bn[ok] / bd[ok], [2.5, 97.5])
            out.update({"ci_95_low": float(lo), "ci_95_high": float(hi)})
        else:
            out["note"] = "too few bootstrap draws with a stable denominator -- CI suppressed"
        return out


# ============================================================
# Plot styling
# ============================================================

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    for d in (OUT_DIR, PLOT_DIR, RETAINED_DIR, CACHE_SUB):
        os.makedirs(d, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUT_DIR, "run_log.txt"))
    t_start = time.perf_counter()
    print(f"E2.1 beta diagnostic  started {time.strftime('%Y-%m-%d %H:%M:%S')}  SMOKE={SMOKE}")
    print(f"cache={os.path.abspath(CACHE_DIR)}\nout={os.path.abspath(OUT_DIR)}")
    print(f"beta grid={BETA_GRID}  budgets={BUDGET_FRACTIONS}  matched={MATCHED}  n_clusters={N_CLUSTERS}")

    with open(E2_SUMMARY) as f:
        e2 = json.load(f)
    e2_ref_nll = {bf: e2["results_grid"][str(bf)]["selective_sensitivity"]["mean_interpolated_nll"] for bf in BUDGET_FRACTIONS}
    e2_ref_repair = {bf: e2["partE_worst_query_recovery"][str(bf)]["selective_sensitivity"]["10"]["frac_positive_loss_repaired"]
                     for bf in BUDGET_FRACTIONS}
    e2_ref_n = {bf: e2["results_grid"][str(bf)]["selective_sensitivity"]["n_retained_vectors"] for bf in BUDGET_FRACTIONS}
    e2_base = e2["baselines"]
    print(f"E2 full-precision references (selective_sensitivity, beta=0.25): {e2_ref_nll}")

    # ============================================================
    # Cache, clusters, retrieval -- identical to E2
    # ============================================================
    print("\nLoading cache...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"], ds["values"]
    ct_keys, ct_values, ct_p_lm = ct["keys"], ct["values"], ct["p_lm_true"]
    val_keys, val_values, val_p_lm = val["keys"], val["values"], val["p_lm_true"]
    n_val, n_ct = len(val_values), len(ct_values)
    print(f"datastore={len(ds_values)}  controller_train={n_ct}  val={n_val}")
    for name, n in [("datastore", len(ds_values)), ("controller_train", n_ct), ("val", n_val)]:
        require(n % CHUNK_LEN == 0, f"{name} size {n} not a multiple of {CHUNK_LEN}: chunk IDs cannot be derived")

    print("Re-deriving DIME clusters (MiniBatchKMeans seed=42, identical to E2)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42)
    assignment = km.fit_predict(ds_keys)
    dime_centroids = km.cluster_centers_.astype(np.float32)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    compressed_dists = np.array(compressed_dists, dtype=object)
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    cluster_totals = np.array([sum(cd.values()) for cd in compressed_dists], dtype=np.float64)
    cluster_entropy = np.zeros(N_CLUSTERS)
    for c in range(N_CLUSTERS):
        if cluster_totals[c] > 0:
            p = np.array(list(compressed_dists[c].values()), dtype=np.float64) / cluster_totals[c]
            cluster_entropy[c] = float(-(p * np.log(p)).sum())

    raw_index, raw_values = build_datastore(ds_keys, ds_values)
    dime_index, dime_values = build_datastore(dime_centroids, compressed_dists)
    K, TAU, ALPHA = MATCHED["k"], MATCHED["tau"], MATCHED["alpha"]

    print("Retrieval + S1/S2 on val...")
    d_raw_v, idx_raw_v = query_knn_indices(raw_index, val_keys, k=K)
    d_dime_v, idx_dime_v = query_knn_indices(dime_index, val_keys, k=K)
    r_raw_v, r_dime_v = raw_values[idx_raw_v], dime_values[idx_dime_v]
    _, L_raw_v = mix_knn_and_lm(d_raw_v, r_raw_v, val_values, val_p_lm, tau=TAU, alpha=ALPHA)
    _, L_dime_v = mix_dime_and_lm(d_dime_v, r_dime_v, val_values, val_p_lm, tau=TAU, alpha=ALPHA)
    p_dime_v = dime_standalone_prob(d_dime_v, r_dime_v, val_values, tau=TAU)

    print("Retrieval + S1/S2 on controller_train (sensitivity ranking + beta tuning)...")
    d_raw_ct, idx_raw_ct = query_knn_indices(raw_index, ct_keys, k=K)
    d_dime_ct, idx_dime_ct = query_knn_indices(dime_index, ct_keys, k=K)
    _, L_raw_ct = mix_knn_and_lm(d_raw_ct, raw_values[idx_raw_ct], ct_values, ct_p_lm, tau=TAU, alpha=ALPHA)
    _, L_dime_ct = mix_dime_and_lm(d_dime_ct, dime_values[idx_dime_ct], ct_values, ct_p_lm, tau=TAU, alpha=ALPHA)

    s3_cache = os.path.join(CACHE_SUB, "p_s3_val.npz")
    idx_sig = hashlib.sha256(idx_dime_v.astype(np.int64).tobytes()).hexdigest()
    if os.path.exists(s3_cache) and str(np.load(s3_cache)["idx_sig"]) == idx_sig:
        print("S3 (full member-aware): loading this script's own cached per-query result...")
        p_s3_v = np.load(s3_cache)["p_s3"]
    else:
        print("S3 (full member-aware) on val via E2's hybrid_standalone_prob(beta=1, all members) -- ~30-40 min...")
        p_s3_v = hybrid_standalone_prob(val_keys, val_values, idx_dime_v, d_dime_v, TAU, 1.0, cluster_to_members,
                                        ds_keys, ds_values, compressed_dists)
        np.savez_compressed(s3_cache, p_s3=p_s3_v, idx_sig=idx_sig)
    L_s3_v = interp_nll(p_s3_v, val_p_lm, ALPHA)
    mL_raw, mL_dime, mL_s3 = float(L_raw_v.mean()), float(L_dime_v.mean()), float(L_s3_v.mean())
    print(f"  mean L_S1={mL_raw:.6f}  L_S2={mL_dime:.6f}  L_S3={mL_s3:.6f}")
    require(abs(mL_raw - e2_base["mean_L_raw_val"]) < REF_TOL, "S1 mean NLL differs from E2")
    require(abs(mL_dime - e2_base["mean_L_dime_val"]) < REF_TOL, "S2 mean NLL differs from E2")
    require(abs(mL_s3 - e2_base["mean_L_S3_val"]) < REF_TOL, "S3 mean NLL differs from E2")
    s3_denom = mL_dime - mL_s3

    # ============================================================
    # Worst-query groups -- E1C matched definition, reused by E1C.2/E2, never re-ranked
    # ============================================================
    delta_dime_raw = L_dime_v - L_raw_v
    order = np.argsort(-delta_dime_raw, kind="stable")
    masks = {}
    for pct in WORST_PCTS:
        m = np.zeros(n_val, dtype=bool)
        m[order[:max(1, round(n_val * pct / 100))]] = True
        masks[f"worst{pct}"] = m
    masks["rest90"] = ~masks["worst10"]
    masks["all"] = np.ones(n_val, dtype=bool)
    n10 = int(masks["worst10"].sum())
    severe_cutoff = float(delta_dime_raw[order[n10 - 1]])
    print(f"worst groups: " + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items()) +
          f"  | severe cutoff (worst-10% boundary L_DIME-L_raw) = {severe_cutoff:.4f}")
    try:
        with open(E1C2_SUMMARY) as f:
            e1c2_w10 = json.load(f)["partB_s3_recovery_on_worst_queries"]["matched"]["10"]
        for nm, arr in [("S1", L_raw_v), ("S2", L_dime_v), ("S3", L_s3_v)]:
            ok = abs(float(arr[masks["worst10"]].mean()) - e1c2_w10[f"{nm}_nll"]["mean"]) < REF_TOL
            print(f"  cross-check worst-10% mean {nm} vs E1C.2 persisted: {ok}")
            require(ok, f"worst-10% {nm} mean differs from E1C.2 -- group identities may differ")
    except FileNotFoundError:
        require(False, f"{E1C2_SUMMARY} missing -- cannot cross-check worst-query identities")

    val_chunk = np.arange(n_val) // CHUNK_LEN
    boot = ChunkBootstrap(val_chunk)
    print(f"bootstrap: {boot.G} val chunks of {CHUNK_LEN} positions, {N_BOOT} paired draws")

    # controller_train chunk split for the secondary, independent analysis
    ct_chunk = np.arange(n_ct) // CHUNK_LEN
    n_ct_chunks = int(ct_chunk.max()) + 1
    perm = np.random.default_rng(42).permutation(n_ct_chunks)
    ct_rank_mask = np.isin(ct_chunk, perm[: n_ct_chunks // 2])
    ct_tune_mask = ~ct_rank_mask
    ct_masks = {"ct_full": np.ones(n_ct, dtype=bool), "ct_halfR": ct_rank_mask, "ct_halfT": ct_tune_mask}
    print(f"controller_train split: {n_ct_chunks} chunks -> half R (ranking) {int(ct_rank_mask.sum())} queries, "
          f"half T (beta tuning) {int(ct_tune_mask.sum())} queries")

    priority_exact = sensitivity_priority(idx_dime_ct, L_dime_ct, L_raw_ct, N_CLUSTERS)
    priority_split = sensitivity_priority(idx_dime_ct, L_dime_ct, L_raw_ct, N_CLUSTERS, query_mask=ct_rank_mask)

    def nearest_centroid_fn(members, c, m):
        return select_members_nearest_centroid(members, dime_centroids[c], ds_keys, m)

    # Fixed stratified sample for full-distribution metrics
    rng_s = np.random.default_rng(42)
    w_idx, r_idx = np.where(masks["worst10"])[0], np.where(masks["rest90"])[0]
    samp = np.sort(np.concatenate([rng_s.choice(w_idx, size=min(N_DIST_WORST, len(w_idx)), replace=False),
                                   rng_s.choice(r_idx, size=min(N_DIST_REST, len(r_idx)), replace=False)]))
    samp_worst = masks["worst10"][samp]
    print(f"\nFull-distribution sample: {len(samp)} queries ({int(samp_worst.sum())} worst-10%, {int((~samp_worst).sum())} rest-90%)")
    D_samp = dime_full_distribution(d_dime_v[samp], r_dime_v[samp], tau=TAU)
    R_samp = raw_full_distribution(d_raw_v[samp], r_raw_v[samp], tau=TAU)
    aligned = align_sample(D_samp, R_samp)
    js_check = max(abs(distribution_metrics(aligned[:20], None, 0.0, val_values[samp][:20])["js_vs_raw"][i]
                       - js_divergence_dict(D_samp[i], R_samp[i])) for i in range(min(20, len(samp))))
    require(js_check < 1e-9, f"vectorized JS disagrees with e1_metrics.js_divergence_dict ({js_check:.2e})")
    delta_s3 = align_delta(aligned, member_delta_distribution(val_keys[samp], idx_dime_v[samp], d_dime_v[samp], TAU,
                                                              cluster_to_members, ds_keys, ds_values, compressed_dists, cluster_totals))

    def summarize_dist(dm):
        out = {}
        for sname, sm in [("worst10", samp_worst), ("rest90", ~samp_worst)]:
            out[sname] = {"n": int(sm.sum()), "mean_entropy": float(dm["entropy"][sm].mean()),
                          "mean_js_vs_raw": float(dm["js_vs_raw"][sm].mean()),
                          "top1_agreement_with_raw": float(dm["top1_agree_raw"][sm].mean())}
        frac_w = masks["worst10"].mean()
        out["all_reweighted"] = {k: float(frac_w * out["worst10"][k] + (1 - frac_w) * out["rest90"][k])
                                 for k in ("mean_entropy", "mean_js_vs_raw", "top1_agreement_with_raw")}
        return out

    dist_dime = summarize_dist(distribution_metrics(aligned, None, 0.0, val_values[samp]))
    dist_s3 = summarize_dist(distribution_metrics(aligned, delta_s3, 1.0, val_values[samp]))
    print(f"  standalone DIME: {dist_dime['all_reweighted']}\n  standalone S3:   {dist_s3['all_reweighted']}")

    report = {
        "experiment": "E2.1 beta diagnostic (information quantity vs utilization)", "smoke": SMOKE,
        "config": {"matched": MATCHED, "budgets": BUDGET_FRACTIONS, "beta_grid": BETA_GRID, "beta_fine": BETA_FINE,
                   "e2_beta": E2_BETA, "max_per_cluster": E2_MAX_PER_CLUSTER, "within_cluster_selection": "nearest_centroid",
                   "n_val": n_val, "n_controller_train": n_ct, "chunk_len": CHUNK_LEN, "n_boot": N_BOOT,
                   "decision_thresholds": {"min_rs3_gain_pp": OUTCOME_MIN_RS3_GAIN_PP, "new_severe_frac": OUTCOME_NEW_SEVERE_FRAC}},
        "baselines": {"mean_L_S1_raw": mL_raw, "mean_L_S2_dime": mL_dime, "mean_L_S3_full_member": mL_s3,
                      "s3_improvement_denominator": s3_denom,
                      "worst_groups": {k: {"n": int(v.sum()), "mean_S1": float(L_raw_v[v].mean()), "mean_S2": float(L_dime_v[v].mean()),
                                           "mean_S3": float(L_s3_v[v].mean())} for k, v in masks.items()},
                      "severe_cutoff_delta_L": severe_cutoff},
        "standalone_distribution_reference": {"dime": dist_dime, "s3": dist_s3},
        "data_reuse_limitations": [
            "PRIMARY beta selection uses controller_train, which E2 also used to build the sensitivity ranking that chose the "
            "retained members -- tuning and ranking share data; not independent validation.",
            "SECONDARY split analysis rebuilds retention from controller_train half R and tunes beta on half T; its retention "
            "sets are reconstructed and differ from E2's, so it is reported separately.",
            "Val-optimal beta is exploratory/oracle only.",
            "Worst-query groups are defined with held-out val labels (retrospective); never used to tune beta.",
            "Bootstrap resamples 127-position chunks; article identity is not persisted, so cross-chunk dependence within an "
            "article is not modeled.",
        ],
        "budgets": {},
    }

    beta_rows, ctrl_rows, worst_rows, token_rows, dist_rows, use_rows, boot_rows = [], [], [], [], [], [], []

    def ct_nll_curve(dec_ct, betas, mask):
        return {b: float(interp_nll(dec_ct["A"] + b * dec_ct["B"], ct_p_lm, ALPHA)[mask].mean()) for b in betas}

    def continuous_opt(A, B, p_lm, mask):
        res = minimize_scalar(lambda b: float(interp_nll(A[mask] + b * B[mask], p_lm[mask], ALPHA).mean()),
                              bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-4})
        return float(res.x), float(res.fun)

    def repair_frac(L_h, mask):
        pos_dime = np.clip(delta_dime_raw[mask], 0, None).sum()
        pos_h = np.clip((L_h - L_raw_v)[mask], 0, None).sum()
        return float((pos_dime - pos_h) / pos_dime) if pos_dime > 0 else None

    for bf in BUDGET_FRACTIONS:
        bk = bkey(bf)
        n_budget = int(bf * len(ds_values))
        print(f"\n################ BUDGET {bk} (~{n_budget} vectors) ################")
        breport = {}

        # ---------------- exact E2 retention sets ----------------
        retained, n_total = allocate_selective(cluster_to_members, priority_exact, n_budget, nearest_centroid_fn)
        info = verify_retained(retained, assignment, cluster_to_members, dime_centroids, ds_keys, E2_MAX_PER_CLUSTER)
        print(f"  exact E2 set: {info}")
        if not SMOKE:
            require(n_total == E2_EXPECTED_COUNTS[bf] == e2_ref_n[bf], f"retained count {n_total} != E2's {e2_ref_n[bf]}")
        ids = retained_ids(retained)
        np.savez_compressed(os.path.join(RETAINED_DIR, f"exact_e2_selective_sensitivity_{bk}.npz"),
                            member_ids=ids, cluster_ids=assignment[ids])
        h_before = info["sha256_sorted_ids"]

        print("  affine decomposition on val (records per-cluster pairs for Section 10)...")
        dv = affine_decomposition(val_keys, val_values, idx_dime_v, d_dime_v, TAU, retained, ds_keys, ds_values,
                                  compressed_dists, cluster_totals, record_pairs=True, label=f"val {bk}")
        print("  affine decomposition on controller_train...")
        dc = affine_decomposition(ct_keys, ct_values, idx_dime_ct, d_dime_ct, TAU, retained, ds_keys, ds_values,
                                  compressed_dists, cluster_totals, label=f"ct {bk}")

        # ---- Checks 1, 2, 5 and the decomposition-vs-E2 cross-check ----
        chk = {}
        chk["check1_beta0_p_max_abs_diff_vs_dime"] = float(np.abs(dv["A"] - p_dime_v).max())
        chk["check1_beta0_nll_max_abs_diff_vs_dime"] = float(np.abs(interp_nll(dv["A"], val_p_lm, ALPHA) - L_dime_v).max())
        require(chk["check1_beta0_p_max_abs_diff_vs_dime"] < 1e-12 and chk["check1_beta0_nll_max_abs_diff_vs_dime"] < REF_TOL,
                "Check 1 failed: beta=0 does not reproduce DIME")
        sub = np.unique(np.concatenate([np.arange(min(N_DIRECT_CHECK // 2, n_val)),
                                        np.where(masks["worst10"])[0][: N_DIRECT_CHECK // 2]]))
        max_direct = 0.0
        for b in BETA_GRID:
            p_direct = hybrid_standalone_prob(val_keys[sub], val_values[sub], idx_dime_v[sub], d_dime_v[sub], TAU, b, retained,
                                              ds_keys, ds_values, compressed_dists, verbose_every=0)
            max_direct = max(max_direct, float(np.abs(p_direct - (dv["A"][sub] + b * dv["B"][sub])).max()))
        chk["affine_vs_e2_direct_loop_max_abs_diff"] = max_direct
        chk["affine_vs_e2_direct_loop_n_queries"] = int(len(sub))
        require(max_direct < 1e-12, f"affine decomposition disagrees with E2's per-beta loop ({max_direct:.2e})")
        L025 = interp_nll(dv["A"] + E2_BETA * dv["B"], val_p_lm, ALPHA)
        chk["check2_beta025_mean_nll"] = float(L025.mean())
        chk["check2_e2_reference_mean_nll"] = e2_ref_nll[bf]
        chk["check2_abs_diff"] = abs(float(L025.mean()) - e2_ref_nll[bf])
        chk["check2_worst10_repair"] = repair_frac(L025, masks["worst10"])
        chk["check2_e2_reference_worst10_repair"] = e2_ref_repair[bf]
        require(chk["check2_abs_diff"] < REF_TOL, f"Check 2 failed: beta=0.25 NLL {L025.mean():.12f} != E2 {e2_ref_nll[bf]:.12f}")
        require(abs(chk["check2_worst10_repair"] - e2_ref_repair[bf]) < 1e-9, "Check 2 failed: worst-10% repair differs from E2")
        print(f"  Check 1 (beta=0 == DIME): max|dp|={chk['check1_beta0_p_max_abs_diff_vs_dime']:.1e}  "
              f"max|dL|={chk['check1_beta0_nll_max_abs_diff_vs_dime']:.1e}")
        print(f"  affine == E2 direct loop on {len(sub)} queries x {len(BETA_GRID)} betas: max|dp|={max_direct:.1e}")
        print(f"  Check 2 (beta=0.25 == E2): {L025.mean():.12f} vs {e2_ref_nll[bf]:.12f}  (|diff|={chk['check2_abs_diff']:.1e});  "
              f"worst-10% repair {chk['check2_worst10_repair']:.10f} vs {e2_ref_repair[bf]:.10f}")
        p_all = {b: dv["A"] + b * dv["B"] for b in BETA_GRID}
        chk["check5_min_p_true"] = float(min(p.min() for p in p_all.values()))
        chk["check5_max_p_true"] = float(max(p.max() for p in p_all.values()))
        chk["check5_all_nll_finite"] = bool(all(np.isfinite(interp_nll(p, val_p_lm, ALPHA)).all() for p in p_all.values()))
        nb = dv["w_ret"] == 0
        chk["check5_queries_without_retained_clusters"] = int(nb.sum())
        chk["check5_B_exactly_zero_when_no_retained"] = bool((dv["B"][nb] == 0).all())
        require(chk["check5_min_p_true"] > -1e-12 and chk["check5_max_p_true"] < 1 + 1e-9 and chk["check5_all_nll_finite"]
                and chk["check5_B_exactly_zero_when_no_retained"], "Check 5 failed: invalid probabilities or fallback")

        delta_b = align_delta(aligned, member_delta_distribution(val_keys[samp], idx_dime_v[samp], d_dime_v[samp], TAU,
                                                                 retained, ds_keys, ds_values, compressed_dists, cluster_totals))
        dist_by_beta = {}
        worst_mass, worst_minp, worst_ptrue = 0.0, 0.0, 0.0
        for b in BETA_GRID:
            dm = distribution_metrics(aligned, delta_b, b, val_values[samp])
            worst_mass = max(worst_mass, float(np.abs(dm["mass_dev"]).max()))
            worst_minp = min(worst_minp, float(dm["min_p"].min()))
            worst_ptrue = max(worst_ptrue, float(np.abs(dm["p_true"] - p_all[b][samp]).max()))
            dist_by_beta[b] = summarize_dist(dm)
        chk["check5_full_dist_max_mass_change"] = worst_mass
        chk["check5_full_dist_min_prob"] = worst_minp
        chk["check5_full_dist_true_token_vs_affine_max_abs_diff"] = worst_ptrue
        # 1e-5, not 1e-9: E2's member softmax runs on float32 distances (ds_keys are
        # float32), so P^member itself sums to 1 only to float32 precision (~1e-7).
        # That is E2's estimator as-is; the observed values are reported, not hidden.
        require(worst_mass < 1e-5 and worst_minp > -1e-7 and worst_ptrue < 1e-5,
                "Check 5 failed: full member-aware distribution not normalized / not consistent")
        chk["check4_beta1_is_partial"] = {"mean_retrieval_weight_on_retained_clusters": float(dv["w_ret"].mean()),
                                         "note": "beta=1 replaces P^DIME only in clusters with retained members; S3 replaces it "
                                                 "in every selected cluster with ALL members -- not the same representation"}
        print(f"  Check 5: p in [{chk['check5_min_p_true']:.2e}, {chk['check5_max_p_true']:.4f}], finite={chk['check5_all_nll_finite']}, "
              f"full-dist mass change {worst_mass:.1e}, no-retained fallback exact={chk['check5_B_exactly_zero_when_no_retained']}")
        print(f"  Check 4: beta=1 is partial -- mean retrieval weight on clusters with retained members = {dv['w_ret'].mean():.4f}")

        # ---------------- controller-based beta selection ----------------
        ct_curves = {mn: ct_nll_curve(dc, BETA_FINE, m) for mn, m in ct_masks.items()}
        sel = {mn: min(BETA_GRID, key=lambda b: ct_curves[mn][b]) for mn in ct_masks}
        cont = {mn: continuous_opt(dc["A"], dc["B"], ct_p_lm, m) for mn, m in ct_masks.items()}
        beta_sel = sel["ct_full"]
        val_opt = continuous_opt(dv["A"], dv["B"], val_p_lm, masks["all"])
        val_grid_best = min(BETA_GRID, key=lambda b: float(interp_nll(p_all[b], val_p_lm, ALPHA).mean()))
        for mn in ct_masks:
            for b in BETA_GRID:
                ctrl_rows.append({"retention_set": "exact_e2", "budget": bk, "tuning_data": mn, "beta": b,
                                  "controller_nll": ct_curves[mn][b], "selected": b == sel[mn]})
        print(f"  controller-selected beta (grid): full={sel['ct_full']}  halfR={sel['ct_halfR']}  halfT={sel['ct_halfT']}  "
              f"| continuous ct optimum={cont['ct_full'][0]:.3f}  | EXPLORATORY val optimum={val_opt[0]:.3f} (grid {val_grid_best})")

        # ---------------- Part D/E metrics over the beta grid ----------------
        L_by_beta = {b: interp_nll(p_all[b], val_p_lm, ALPHA) for b in BETA_GRID}
        for b in BETA_FINE:
            if b in L_by_beta:
                continue
            L_by_beta[b] = interp_nll(dv["A"] + b * dv["B"], val_p_lm, ALPHA)
        fine_curve = []
        for b in BETA_FINE:
            L = L_by_beta[b]
            r_pct, _ = ratio_with_stability_guard(mL_dime - L.mean(), s3_denom)
            fine_curve.append({"beta": b, "val_nll": float(L.mean()), "r_s3_pct": r_pct,
                               "worst10_repair": repair_frac(L, masks["worst10"]), "ct_full_nll": ct_curves["ct_full"][b]})

        beta_table = []
        for b in BETA_GRID:
            L, p = L_by_beta[b], p_all[b]
            r_pct, r_interp = ratio_with_stability_guard(mL_dime - L.mean(), s3_denom)
            row = {"retention_set": "exact_e2", "budget": bk, "beta": b, "n_retained": n_total, "retained_sha256": h_before[:16],
                   "val_mean_nll": float(L.mean()), "d_nll_vs_dime": float(L.mean() - mL_dime),
                   "d_nll_vs_beta025": float(L.mean() - L025.mean()), "r_s3_pct": r_pct, "r_s3_interpretability": r_interp,
                   "worst10_repair": repair_frac(L, masks["worst10"]),
                   "ct_full_nll": ct_curves["ct_full"][b], "ct_halfR_nll": ct_curves["ct_halfR"][b], "ct_halfT_nll": ct_curves["ct_halfT"][b],
                   "is_controller_selected": b == beta_sel, "is_e2_reference": b == E2_BETA}
            beta_table.append(row)
            beta_rows.append(row)

            for gname in ("worst1", "worst5", "worst10", "rest90", "all"):
                m = masks[gname]
                Lg, Ld, Lr = L[m], L_dime_v[m], L_raw_v[m]
                pos_d = np.clip(Ld - Lr, 0, None).sum()
                pos_h = np.clip(Lg - Lr, 0, None).sum()
                harmed = Lg > Ld
                worst_rows.append({
                    "budget": bk, "beta": b, "group": gname, "n": int(m.sum()), "mean_nll": float(Lg.mean()),
                    "mean_nll_dime": float(Ld.mean()), "mean_nll_s3": float(L_s3_v[m].mean()), "mean_nll_raw": float(Lr.mean()),
                    "frac_positive_loss_repaired": float((pos_d - pos_h) / pos_d) if pos_d > 0 else None,
                    "s3_frac_positive_loss_repaired": float((pos_d - np.clip(L_s3_v[m] - Lr, 0, None).sum()) / pos_d) if pos_d > 0 else None,
                    "frac_improved_vs_dime": float((Lg < Ld).mean()), "frac_harmed_vs_dime": float(harmed.mean()),
                    "mean_nll_reduction_vs_dime": float((Ld - Lg).mean()),
                    "mean_harm_given_harmed": float((Lg - Ld)[harmed].mean()) if harmed.any() else 0.0,
                    "frac_harmed_vs_beta025": float((Lg > L025[m]).mean()),
                    "residual_positive_loss_vs_raw_sum": float(pos_h), "residual_positive_loss_vs_raw_mean": float(pos_h / m.sum()),
                    "n_severe_vs_raw": int(((Lg - Lr) >= severe_cutoff).sum()),
                    "n_severe_vs_raw_dime": int(((Ld - Lr) >= severe_cutoff).sum()),
                })
                pg, pd_ = p[m], p_dime_v[m]
                token_rows.append({
                    "budget": bk, "beta": b, "group": gname, "n": int(m.sum()),
                    "mean_p_true_standalone": float(pg.mean()), "median_p_true_standalone": float(np.median(pg)),
                    "mean_p_true_interpolated": float((ALPHA * pg + (1 - ALPHA) * val_p_lm[m]).mean()),
                    "frac_p_true_increases_vs_dime": float((pg - pd_ > 1e-14).mean()),
                    "frac_p_true_decreases_vs_dime": float((pg - pd_ < -1e-14).mean()),
                    "correct_token_coverage": float((pg > 0).mean()), "dime_coverage": float((pd_ > 0).mean()),
                })
            for sname, vals in dist_by_beta[b].items():
                dist_rows.append({"budget": bk, "beta": b, "stratum": sname, **{k: v for k, v in vals.items() if k != "n"}})

        # ---------------- Section 10: effective use of retained information ----------------
        betas_use = sorted({E2_BETA, beta_sel, 1.0})
        w_ret = dv["w_ret"]
        use = {"frac_queries_with_any_retained_cluster": float((w_ret > 0).mean()),
               "mean_retrieval_weight_on_retained_clusters": {g: float(w_ret[masks[g]].mean()) for g in ("worst10", "rest90", "all")},
               "mean_weight_given_any": float(w_ret[w_ret > 0].mean()) if (w_ret > 0).any() else None, "weight_bins": [], "cluster_categories": []}
        edges = [(0.0, 0.0, "w=0"), (0.0, 0.1, "0<w<=0.1"), (0.1, 0.3, "0.1<w<=0.3"), (0.3, 0.6, "0.3<w<=0.6"), (0.6, 1.0 + 1e-9, "w>0.6")]
        for lo, hi, lab in edges:
            m = (w_ret == 0) if hi == 0.0 else ((w_ret > lo) & (w_ret <= hi))
            row = {"budget": bk, "kind": "query_weight_bin", "category": lab, "n_queries": int(m.sum()), "frac_queries": float(m.mean()),
                   "frac_of_bin_in_worst10": float(masks["worst10"][m].mean()) if m.any() else None}
            for b in betas_use:
                row[f"mean_dnll_vs_dime_beta{b:g}"] = float((L_by_beta[b] - L_dime_v)[m].mean()) if m.any() else None
            use["weight_bins"].append(row)
            use_rows.append(row)
        pr = dv["pairs"]
        n_ret_c = np.zeros(N_CLUSTERS)
        for c, r in retained.items():
            n_ret_c[c] = len(r)
        frac_ret_c = np.where(cluster_totals > 0, n_ret_c / np.maximum(cluster_totals, 1), 0.0)
        ret_clusters = np.where(n_ret_c > 0)[0]
        ent_med = float(np.median(cluster_entropy[ret_clusters])) if len(ret_clusters) else 0.0
        cats = {
            "n_retained=50 (capped)": n_ret_c[pr["c"]] >= E2_MAX_PER_CLUSTER,
            "n_retained<50": n_ret_c[pr["c"]] < E2_MAX_PER_CLUSTER,
            "cluster fully retained": frac_ret_c[pr["c"]] >= 1.0,
            "retained frac in [0.5,1)": (frac_ret_c[pr["c"]] >= 0.5) & (frac_ret_c[pr["c"]] < 1.0),
            "retained frac < 0.5": frac_ret_c[pr["c"]] < 0.5,
            f"token entropy <= median ({ent_med:.2f})": cluster_entropy[pr["c"]] <= ent_med,
            f"token entropy > median ({ent_med:.2f})": cluster_entropy[pr["c"]] > ent_med,
        }
        tot_w = pr["w"].sum()
        for lab, m in cats.items():
            wsum = pr["w"][m].sum()
            row = {"budget": bk, "kind": "retrieved_cluster_category", "category": lab,
                   "n_clusters": int(len(np.unique(pr["c"][m]))), "n_query_cluster_pairs": int(m.sum()),
                   "share_of_retained_weight": float(wsum / tot_w) if tot_w > 0 else None,
                   "weighted_mean_member_minus_dime_p_true": float((pr["w"][m] * pr["diff"][m]).sum() / wsum) if wsum > 0 else None,
                   "contribution_to_mean_B": float((pr["w"][m] * pr["diff"][m]).sum() / n_val),
                   "frac_pairs_member_higher": float((pr["diff"][m] > 0).mean()) if m.any() else None,
                   "frac_pairs_member_lower": float((pr["diff"][m] < 0).mean()) if m.any() else None}
            use["cluster_categories"].append(row)
            use_rows.append(row)
        print(f"  Section 10: {use['frac_queries_with_any_retained_cluster']*100:.2f}% of queries retrieve >=1 cluster with retained "
              f"members; mean retained weight all={use['mean_retrieval_weight_on_retained_clusters']['all']:.4f} "
              f"worst10={use['mean_retrieval_weight_on_retained_clusters']['worst10']:.4f}")

        # ---------------- Section 12: uncertainty ----------------
        L_sel = L_by_beta[beta_sel]
        L_vopt = interp_nll(dv["A"] + val_opt[0] * dv["B"], val_p_lm, ALPHA)
        w10 = masks["worst10"]
        pos_d_q = np.clip(delta_dime_raw, 0, None)
        bs = {
            "selected_minus_beta025": boot.mean(L_sel - L025),
            "selected_minus_dime": boot.mean(L_sel - L_dime_v),
            "selected_minus_s3": boot.mean(L_sel - L_s3_v),
            "beta025_minus_dime": boot.mean(L025 - L_dime_v),
            "valopt_minus_beta025_EXPLORATORY": boot.mean(L_vopt - L025),
            "r_s3_selected": boot.ratio(L_dime_v - L_sel, L_dime_v - L_s3_v, min_mean_den=0.01),
            "r_s3_beta025": boot.ratio(L_dime_v - L025, L_dime_v - L_s3_v, min_mean_den=0.01),
            "r_s3_valopt_EXPLORATORY": boot.ratio(L_dime_v - L_vopt, L_dime_v - L_s3_v, min_mean_den=0.01),
            "worst10_mean_reduction_selected_vs_dime": boot.mean(L_dime_v - L_sel, mask=w10),
            "worst10_mean_reduction_selected_vs_beta025": boot.mean(L025 - L_sel, mask=w10),
            "worst10_repair_selected": boot.ratio(pos_d_q - np.clip(L_sel - L_raw_v, 0, None), pos_d_q, mask=w10),
            "worst10_repair_beta025": boot.ratio(pos_d_q - np.clip(L025 - L_raw_v, 0, None), pos_d_q, mask=w10),
            "worst10_repair_selected_minus_beta025": boot.ratio(np.clip(L025 - L_raw_v, 0, None) - np.clip(L_sel - L_raw_v, 0, None),
                                                              pos_d_q, mask=w10),
            "rest90_selected_minus_beta025": boot.mean(L_sel - L025, mask=masks["rest90"]),
        }
        for nm, v in bs.items():
            boot_rows.append({"retention_set": "exact_e2", "budget": bk, "beta_selected": beta_sel, "statistic": nm, **v})
        print(f"  selected beta={beta_sel}: NLL={L_sel.mean():.6f}  vs beta=0.25 diff={bs['selected_minus_beta025']['point']:+.6f} "
              f"[{bs['selected_minus_beta025']['ci_95_low']:+.6f}, {bs['selected_minus_beta025']['ci_95_high']:+.6f}]  "
              f"R_S3={fmt_pct(bs['r_s3_selected']['point'], 100)}")

        # ---------------- secondary: split-independent reconstructed retention ----------------
        retained_s, n_total_s = allocate_selective(cluster_to_members, priority_split, n_budget, nearest_centroid_fn)
        info_s = verify_retained(retained_s, assignment, cluster_to_members, dime_centroids, ds_keys, E2_MAX_PER_CLUSTER)
        ids_s = retained_ids(retained_s)
        np.savez_compressed(os.path.join(RETAINED_DIR, f"split_halfR_selective_sensitivity_{bk}.npz"),
                            member_ids=ids_s, cluster_ids=assignment[ids_s])
        inter = len(np.intersect1d(ids, ids_s))
        overlap = {"jaccard_member_ids": float(inter / len(np.union1d(ids, ids_s))) if len(ids) else None,
                   "frac_of_exact_set_shared": float(inter / len(ids)) if len(ids) else None,
                   "jaccard_clusters": float(len(set(retained) & set(retained_s)) / len(set(retained) | set(retained_s)))}
        dv_s = affine_decomposition(val_keys, val_values, idx_dime_v, d_dime_v, TAU, retained_s, ds_keys, ds_values,
                                    compressed_dists, cluster_totals, label=f"val split {bk}")
        dc_s = affine_decomposition(ct_keys, ct_values, idx_dime_ct, d_dime_ct, TAU, retained_s, ds_keys, ds_values,
                                    compressed_dists, cluster_totals, label=f"ct split {bk}")
        ct_s = ct_nll_curve(dc_s, BETA_FINE, ct_tune_mask)
        sel_s = min(BETA_GRID, key=lambda b: ct_s[b])
        cont_s = continuous_opt(dc_s["A"], dc_s["B"], ct_p_lm, ct_tune_mask)
        L025_s = interp_nll(dv_s["A"] + E2_BETA * dv_s["B"], val_p_lm, ALPHA)
        Lsel_s = interp_nll(dv_s["A"] + sel_s * dv_s["B"], val_p_lm, ALPHA)
        split_curve = []
        for b in BETA_GRID:
            L = interp_nll(dv_s["A"] + b * dv_s["B"], val_p_lm, ALPHA)
            r_pct, r_interp = ratio_with_stability_guard(mL_dime - L.mean(), s3_denom)
            row = {"retention_set": "split_reconstructed", "budget": bk, "beta": b, "n_retained": n_total_s,
                   "retained_sha256": info_s["sha256_sorted_ids"][:16], "val_mean_nll": float(L.mean()),
                   "d_nll_vs_dime": float(L.mean() - mL_dime), "d_nll_vs_beta025": float(L.mean() - L025_s.mean()),
                   "r_s3_pct": r_pct, "r_s3_interpretability": r_interp, "worst10_repair": repair_frac(L, w10),
                   "ct_halfT_nll": ct_s[b], "is_controller_selected": b == sel_s, "is_e2_reference": b == E2_BETA}
            split_curve.append(row)
            beta_rows.append(row)
            ctrl_rows.append({"retention_set": "split_reconstructed", "budget": bk, "tuning_data": "ct_halfT", "beta": b,
                              "controller_nll": ct_s[b], "selected": b == sel_s})
        bs_split = {"selected_minus_beta025": boot.mean(Lsel_s - L025_s), "selected_minus_dime": boot.mean(Lsel_s - L_dime_v),
                    "r_s3_selected": boot.ratio(L_dime_v - Lsel_s, L_dime_v - L_s3_v, min_mean_den=0.01)}
        for nm, v in bs_split.items():
            boot_rows.append({"retention_set": "split_reconstructed", "budget": bk, "beta_selected": sel_s, "statistic": nm, **v})
        print(f"  SPLIT (reconstructed, ranking on half R, beta tuned on half T): selected beta={sel_s} (continuous {cont_s[0]:.3f})  "
              f"val NLL={Lsel_s.mean():.6f} (beta=0.25: {L025_s.mean():.6f})  member-ID overlap with E2 set={overlap['frac_of_exact_set_shared']}")
        h_after = retained_hash(retained)
        chk["check3_same_retained_ids_for_every_beta"] = (h_after == h_before)
        chk["check3_note"] = "every beta row is computed from one (A,B) pair built from this single retention set; hash re-checked after all rows"
        require(h_after == h_before, "Check 3 failed: retention set changed during the beta sweep")

        # ---------------- per-budget record ----------------
        def rs3(L):
            return ratio_with_stability_guard(mL_dime - L.mean(), s3_denom)[0]
        rest = masks["rest90"]
        breport = {
            "exact_e2": {
                "retention": info, "checks": chk,
                "controller_selection": {"grid_selected": sel, "continuous_optimum": {k: {"beta": v[0], "nll": v[1]} for k, v in cont.items()},
                                         "primary_selected_beta": beta_sel,
                                         "limitation": "ct_full was also used for E2's sensitivity ranking -- not independent"},
                "exploratory_val_optimum": {"continuous_beta": val_opt[0], "val_nll": val_opt[1], "grid_best_beta": val_grid_best,
                                            "r_s3_pct": rs3(L_vopt), "label": "EXPLORATORY / oracle -- chosen on evaluation labels"},
                "beta_table": beta_table, "fine_curve": fine_curve,
                "required_comparison": {"beta025_nll": float(L025.mean()), "controller_selected_beta": beta_sel,
                                        "selected_nll": float(L_sel.mean()), "r_s3_beta025_pct": rs3(L025), "r_s3_selected_pct": rs3(L_sel)},
                "effective_use": use, "bootstrap": bs,
                "new_severe_rest90": {"beta025": int(((L025 - L_raw_v)[rest] >= severe_cutoff).sum()),
                                      "selected": int(((L_sel - L_raw_v)[rest] >= severe_cutoff).sum()),
                                      "beta1": int(((L_by_beta[1.0] - L_raw_v)[rest] >= severe_cutoff).sum()), "n_rest90": int(rest.sum())},
                "fig5_per_query_delta": {"beta025": L025 - L_dime_v, "selected": L_sel - L_dime_v, "valopt": L_vopt - L_dime_v},
            },
            "split_reconstructed": {"retention": info_s, "overlap_with_exact_e2": overlap, "selected_beta_halfT": sel_s,
                                    "continuous_optimum_halfT": {"beta": cont_s[0], "nll": cont_s[1]}, "beta_table": split_curve,
                                    "bootstrap": bs_split, "r_s3_selected_pct": rs3(Lsel_s), "r_s3_beta025_pct": rs3(L025_s)},
        }
        report["budgets"][bk] = breport
        to_save = json.loads(json.dumps(report, default=lambda o: None if isinstance(o, np.ndarray) else (o.item() if hasattr(o, "item") else str(o))))
        dump_json(os.path.join(OUT_DIR, "e2_1_summary.json"), to_save)

    # ============================================================
    # Mechanical outcome classification + research-question evidence
    # ============================================================
    decision = {}
    for bf in BUDGET_FRACTIONS:
        ex = report["budgets"][bkey(bf)]["exact_e2"]
        d = ex["bootstrap"]["selected_minus_beta025"]
        rc = ex["required_comparison"]
        gain_pp = (rc["r_s3_selected_pct"] - rc["r_s3_beta025_pct"]) if rc["r_s3_selected_pct"] is not None else None
        sig = d["ci_95_high"] < 0
        substantial = bool(sig and gain_pp is not None and gain_pp >= OUTCOME_MIN_RS3_GAIN_PP)
        ns = ex["new_severe_rest90"]
        new_severe_excess = ns["selected"] - ns["beta025"]
        decision[bkey(bf)] = {"selected_beta": rc["controller_selected_beta"], "nll_diff_vs_beta025": d["point"],
                              "ci_excludes_zero_improvement": bool(sig), "r_s3_gain_pp": gain_pp, "substantial_utilization_gain": substantial,
                              "new_severe_rest90_excess": int(new_severe_excess),
                              "new_severe_flag": bool(new_severe_excess > OUTCOME_NEW_SEVERE_FRAC * ns["n_rest90"])}
    any_sub = [k for k, v in decision.items() if v["substantial_utilization_gain"]]
    if any_sub and any(decision[k]["new_severe_flag"] for k in any_sub):
        outcome = "C"
    elif any_sub:
        outcome = "A"
    else:
        outcome = "B"
    sel_seq = [decision[bkey(bf)]["selected_beta"] for bf in BUDGET_FRACTIONS]
    report["decision"] = {"per_budget": decision, "mechanical_outcome": outcome,
                          "selected_beta_by_budget": sel_seq,
                          "selected_beta_nondecreasing_with_budget": bool(all(a <= b for a, b in zip(sel_seq, sel_seq[1:]))),
                          "note": "pre-registered heuristic thresholds; the written report, not this flag, is the conclusion"}
    print(f"\nMechanical outcome: {outcome}   selected beta by budget: {sel_seq}")

    # ============================================================
    # Machine-readable outputs
    # ============================================================
    write_csv(os.path.join(OUT_DIR, "beta_sensitivity.csv"), beta_rows)
    write_csv(os.path.join(OUT_DIR, "controller_selection.csv"), ctrl_rows)
    write_csv(os.path.join(OUT_DIR, "worst_query_analysis.csv"), worst_rows)
    write_csv(os.path.join(OUT_DIR, "correct_token_probability.csv"), token_rows)
    write_csv(os.path.join(OUT_DIR, "distribution_properties_standalone.csv"), dist_rows)
    write_csv(os.path.join(OUT_DIR, "effective_use.csv"), use_rows)
    write_csv(os.path.join(OUT_DIR, "bootstrap_uncertainty.csv"), boot_rows)
    fig5_data = {bk: report["budgets"][bk]["exact_e2"].pop("fig5_per_query_delta") for bk in report["budgets"]}
    np.savez_compressed(os.path.join(OUT_DIR, "per_query_delta_nll_vs_dime.npz"),
                        **{f"{bk}_{k}": v for bk, dd in fig5_data.items() for k, v in dd.items()})
    dump_json(os.path.join(OUT_DIR, "e2_1_summary.json"), report)

    # ============================================================
    # Figures
    # ============================================================
    print("\nGenerating figures...")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelcolor": INK, "text.color": INK})
    ex_all = {bf: report["budgets"][bkey(bf)]["exact_e2"] for bf in BUDGET_FRACTIONS}

    def curve_fig(ykey, ylabel, title, fname, refs):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for bf in BUDGET_FRACTIONS:
            fc = ex_all[bf]["fine_curve"]
            xs = [r["beta"] for r in fc]
            ys = [np.nan if r[ykey] is None else r[ykey] for r in fc]
            ax.plot(xs, ys, color=COLORS[bf], linewidth=2, label=f"{bf*100:g}% budget")
            gx = [x for x in xs if x in BETA_GRID]
            gy = [y for x, y in zip(xs, ys) if x in BETA_GRID]
            ax.plot(gx, gy, "o", color=COLORS[bf], markersize=5)
            bsel = ex_all[bf]["controller_selection"]["primary_selected_beta"]
            ysel = [y for x, y in zip(xs, ys) if abs(x - bsel) < 1e-9][0]
            ax.plot([bsel], [ysel], "*", color=COLORS[bf], markersize=15, markeredgecolor="white", markeredgewidth=1.2, zorder=5)
            ax.annotate(f"{bf*100:g}%", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        for val_, lab, col, ls in refs:
            ax.axhline(val_, color=col, linestyle=ls, linewidth=1.2)
            ax.annotate(lab, (0.0, val_), xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK_MUTED)
        ax.axvline(E2_BETA, color="#d6d5cf", linewidth=1)
        ax.set_xlabel("β (weight on retained-member distribution)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.plot([], [], "*", color=INK_MUTED, markersize=12, label="controller-selected β")
        ax.legend(frameon=False, loc="best")
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
        plt.close(fig)

    curve_fig("val_nll", "mean interpolated NLL (held-out val)", "Fig 1 — NLL vs β (exact E2 retention sets; vertical line = E2's β=0.25)",
              "fig1_nll_vs_beta.png", [(mL_dime, "DIME (S2)", REF_DIME, "--"), (mL_s3, "full S3 (all members)", REF_S3, ":")])
    curve_fig("r_s3_pct", "% of S3's NLL improvement recovered", "Fig 2 — S3 recovery vs β (held-out val)",
              "fig2_s3_recovery_vs_beta.png", [(0.0, "DIME", REF_DIME, "--"), (100.0, "full S3", REF_S3, ":")])
    s3_rep = [r for r in worst_rows if r["group"] == "worst10"][0]["s3_frac_positive_loss_repaired"]
    curve_fig("worst10_repair", "fraction of worst-10% positive loss repaired", "Fig 3 — worst-10% repair vs β (retrospective groups)",
              "fig3_worst10_repair_vs_beta.png", [(0.0, "DIME", REF_DIME, "--"), (s3_rep, "full S3", REF_S3, ":")])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs = np.arange(len(BUDGET_FRACTIONS))
    series = [("controller-selected (grid, ct_full) — primary", [ex_all[bf]["controller_selection"]["primary_selected_beta"] for bf in BUDGET_FRACTIONS], "o", "-", COLORS[0.01], True),
              ("controller continuous optimum (ct_full)", [ex_all[bf]["controller_selection"]["continuous_optimum"]["ct_full"]["beta"] for bf in BUDGET_FRACTIONS], "s", "--", COLORS[0.05], True),
              ("split-independent continuous opt. (rank half R, tune half T)",[report["budgets"][bkey(bf)]["split_reconstructed"]["continuous_optimum_halfT"]["beta"] for bf in BUDGET_FRACTIONS], "^", "-.", COLORS[0.10], True),
              ("val optimum — EXPLORATORY oracle", [ex_all[bf]["exploratory_val_optimum"]["continuous_beta"] for bf in BUDGET_FRACTIONS], "D", ":", INK_MUTED, False)]
    for lab, ys, mk, ls, col, filled in series:
        ax.plot(xs, ys, marker=mk, linestyle=ls, color=col, linewidth=2, markersize=8,
                markerfacecolor=col if filled else "white", markeredgecolor=col, label=lab)
    ax.axhline(E2_BETA, color=REF_DIME, linestyle="--", linewidth=1)
    ax.annotate("E2 fixed β=0.25", (xs[0], E2_BETA), xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{bf*100:g}%\n({ex_all[bf]['retention']['n_retained_vectors']:,} vectors)" for bf in BUDGET_FRACTIONS])
    ax.set_xlabel("extra raw storage budget")
    ax.set_ylabel("preferred β")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Fig 4 — preferred β vs storage budget")
    ax.legend(frameon=False, fontsize=8, loc="best")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig4_best_beta_vs_budget.png"), dpi=150)
    plt.close(fig)

    bk10 = bkey(0.10)
    d10 = fig5_data[bk10]
    bsel10 = ex_all[0.10]["controller_selection"]["primary_selected_beta"]
    vopt10 = ex_all[0.10]["exploratory_val_optimum"]["continuous_beta"]
    allv = np.concatenate([np.abs(v[v != 0]) for v in d10.values()])
    vmax = float(allv.max()) if len(allv) else 1.0
    min_edge = 1e-6
    pos_edges = np.logspace(np.log10(min_edge), np.log10(max(vmax, 1e-4)) + 0.01, 40)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_sets = [("beta025", f"β=0.25 (E2)", COLORS[0.01], "-"), ("selected", f"β={bsel10:g} (controller-selected)", COLORS[0.05], "-")]
    if abs(vopt10 - bsel10) > 1e-3:
        plot_sets.append(("valopt", f"β={vopt10:.2f} (val optimum, EXPLORATORY)", INK_MUTED, ":"))
    for key_, lab, col, ls in plot_sets:
        v = d10[key_]
        nz = v[v != 0]
        tiny = int(((nz != 0) & (np.abs(nz) < min_edge)).sum())
        lab_full = (f"{lab}: improved {(v < 0).mean()*100:.1f}%, harmed {(v > 0).mean()*100:.1f}%"
                    f" ({tiny} with |Δ|<{min_edge:g} not drawn)")
        # two one-sided histograms, so no bin straddles zero and lumps tiny changes together
        ax.hist(nz[nz <= -min_edge], bins=-pos_edges[::-1], histtype="step", linewidth=2, color=col, linestyle=ls, label=lab_full)
        ax.hist(nz[nz >= min_edge], bins=pos_edges, histtype="step", linewidth=2, color=col, linestyle=ls)
    ax.set_xscale("symlog", linthresh=min_edge)
    ax.set_yscale("log")
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.annotate("← improved vs DIME", (0.02, 0.96), xycoords="axes fraction", fontsize=9, color=INK_MUTED, va="top")
    ax.annotate("harmed vs DIME →", (0.98, 0.96), xycoords="axes fraction", fontsize=9, color=INK_MUTED, va="top", ha="right")
    zero_frac = float((d10["beta025"] == 0).mean())
    ax.set_xlabel("per-query interpolated NLL change vs DIME (symlog)")
    ax.set_ylabel("queries (log)")
    ax.set_title(f"Fig 5 — per-query NLL change at 10% budget (held-out val)\n"
                 f"{zero_frac*100:.1f}% of queries unchanged (no retrieved cluster has retained members), not shown")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig5_per_query_delta_10pct.png"), dpi=150)
    plt.close(fig)

    # ============================================================
    # Auto-generated numeric report (numbers only; interpretation written separately)
    # ============================================================
    lines = ["# E2.1 — auto-generated numeric summary", "",
             f"Mechanical outcome flag: **{outcome}** (pre-registered heuristic; not the conclusion).", "",
             f"Baselines (val, matched k=20 τ=2.0 α=0.25): S1 raw {mL_raw:.4f} · S2 DIME {mL_dime:.4f} · S3 {mL_s3:.4f}", "",
             "## Required comparison (exact E2 retention sets)", "",
             "| Budget | β=0.25 NLL | Controller-selected β | Selected-β NLL | ΔNLL vs β=0.25 [95% chunk CI] | S3 recovery (0.25 → selected) | Val-optimal β (exploratory) |",
             "|---|---:|---:|---:|---|---|---:|"]
    for bf in BUDGET_FRACTIONS:
        ex = ex_all[bf]
        rc, d = ex["required_comparison"], ex["bootstrap"]["selected_minus_beta025"]
        lines.append(f"| {bf*100:g}% | {rc['beta025_nll']:.4f} | {rc['controller_selected_beta']:g} | {rc['selected_nll']:.4f} | "
                     f"{d['point']:+.5f} [{d['ci_95_low']:+.5f}, {d['ci_95_high']:+.5f}] | "
                     f"{fmt_pct(rc['r_s3_beta025_pct'])} → {fmt_pct(rc['r_s3_selected_pct'])} | "
                     f"{ex['exploratory_val_optimum']['continuous_beta']:.3f} ({fmt_pct(ex['exploratory_val_optimum']['r_s3_pct'])}) |")
    lines += ["", "## Split-independent selection (reconstructed sets; ranking half R, β tuned on half T)", "",
              "| Budget | Selected β | Continuous opt. | S3 recovery (0.25 → selected) | Member-ID overlap with E2 set |", "|---|---:|---:|---|---:|"]
    for bf in BUDGET_FRACTIONS:
        sp = report["budgets"][bkey(bf)]["split_reconstructed"]
        lines.append(f"| {bf*100:g}% | {sp['selected_beta_halfT']:g} | {sp['continuous_optimum_halfT']['beta']:.3f} | "
                     f"{fmt_pct(sp['r_s3_beta025_pct'])} → {fmt_pct(sp['r_s3_selected_pct'])} | {sp['overlap_with_exact_e2']['frac_of_exact_set_shared']:.3f} |")
    lines += ["", "## Worst-10% (retrospective) at β = 0.25 / selected / 1.0", "",
              "| Budget | β | Repaired | Improved vs DIME | Harmed vs DIME | New severe in rest-90% |", "|---|---:|---:|---:|---:|---:|"]
    for bf in BUDGET_FRACTIONS:
        ex = ex_all[bf]
        for b in sorted({E2_BETA, ex["controller_selection"]["primary_selected_beta"], 1.0}):
            w = [r for r in worst_rows if r["budget"] == bkey(bf) and r["beta"] == b and r["group"] == "worst10"][0]
            rr = [r for r in worst_rows if r["budget"] == bkey(bf) and r["beta"] == b and r["group"] == "rest90"][0]
            lines.append(f"| {bf*100:g}% | {b:g} | {w['frac_positive_loss_repaired']:.4f} | {w['frac_improved_vs_dime']:.3f} | "
                         f"{w['frac_harmed_vs_dime']:.3f} | {rr['n_severe_vs_raw']} |")
    lines += ["", f"S3 reference on worst-10%: repaired {s3_rep:.4f}.", "",
              "Limitations: " + " ".join(report["data_reuse_limitations"])]
    with open(os.path.join(OUT_DIR, "e2_1_numeric_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nAll E2.1 outputs saved under {os.path.abspath(OUT_DIR)}  ({(time.perf_counter()-t_start)/60:.1f} min)")
    print("No E1-series or E2 output was modified.")


if __name__ == "__main__":
    main()
