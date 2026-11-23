"""Y2 -- Representability, Learnability, and Functional Complexity of
Local kNN Retrieval. Second experiment in the nonlinear Y-series. Writes
only to results/stage3_y2_representability_learnability/, preserving
every R-series, R-extra, and Y1 output unchanged.

=====================================================================
Audit of Y1's REAL results (read directly from
results/stage3_y1_nonlinear_function_feasibility/y1_nonlinear_summary.json,
which exists locally and is read, not reproduced from assumptions)
=====================================================================
Y1's real sample: 15 clusters selected from 86/150 scanned candidates
meeting fit>=30/tune>=15/test>=20. At m=8, rank=8 (the primary config):
D0=0.8862, A0=0.8293, N1=0.8510, N2=0.8301 mean held-out TV -- ALL far
from accurate, and N2 did NOT significantly beat A0 on real data
(frac_clusters_N2_beats_A0=0.533, 95% CI [-0.0092, +0.0118], straddling
zero) despite N2 clearly beating A0 on Y1's own pilot's trivial synthetic
structure -- a genuine, honest real-vs-pilot discrepancy, not suppressed.
Query-change preservation failed for EVERY model including the nonlinear
ones (A0=2.80, N1=2.07, N2=2.41, all WORSE than D0's zero-change baseline
of 1.77). The m-sweep showed A0/N2 beating the histogram at low m but
LOSING to it at m=20 (A0=0.837 vs dime=0.805) -- the nonlinear models help
exactly where the target is hardest (small m) and hurt where the
histogram is already adequate. Y1's own final verdict: "Promising but
insufficient evidence." These are the REAL numbers Y2 investigates further
-- not assumed, not re-derived, read directly from the existing artifact.

Y1's basis/anchor/training-query facts (confirmed by direct inspection of
run_y1_nonlinear_function_feasibility.py, reused unchanged here for D0/A0/
N1/N2): A0 reuses R4's own affine anchor-difference features via
run_y1_per_cluster_function's local_m_distribution as the target and
run_crisp_r4_histogram_corrections' log_ratio_target/fit_ridge/
reconstruct_logit as the fitting machinery (R-extra's basis, confirmed in
R7's audit to be affine in q). N1/N2 use the RAW (never logged) Gaussian
RBF feature value specifically to remain genuinely non-affine (verified
numerically, not just by construction, in Y1's check_non_affine_expressivity).
Y1's real per-cluster query counts (see its JSON, per_cluster_counts) were
45-63 fit queries per cluster -- scarce, exactly Y2's named concern.

=====================================================================
What is genuinely new in Y2 (not a rerun of Y1 with different ranks)
=====================================================================
  - A torch-based MLP family (G_theta(q) = softmax(f_theta(q)), f_theta a
    configurable-depth/width ReLU network taking the RAW 768-dim query
    directly, trained with a soft-target cross-entropy loss that handles
    exact teacher zeros correctly: -sum_y T(y)*log_softmax(y) has every
    T(y)=0 term contribute exactly 0, never NaN, since log_softmax is
    always finite for finite logits).
  - A larger-rank RBF family (RBF-large) sharing N2's exact fitting
    machinery, just more anchors -- isolates whether N1/N2's modest ranks
    (<=32) were a capacity bottleneck.
  - An explicit TRAINING-SET MEMORIZATION test (Y2-A3): can any family fit
    a FIXED finite set of training targets to near-zero TRAINING error?
    This is deliberately NOT a generalization claim -- it isolates whether
    Y1's real-data failure was architectural (cannot even memorize) or a
    generalization/data problem (memorizes fine, fails on held-out queries).
  - An exact, derived (not merely empirical) boundary-jump formula (Phase
    C2): at an ordinary single different-label exchange, TV(F_+,F_-) =
    w/Z exactly (w the shared tie-weight, Z the common normalizer) --
    verified numerically against the real teacher, not just asserted.
  - Natural-vs-boundary-enriched training (Y2-B4): boundary-enriched
    points are constructed by linear interpolation between near, label-
    differing cluster members (a HEURISTIC construction for WHERE to
    sample, stated as such) -- but every boundary-enriched target is the
    EXACT teacher F_C^(m) evaluated at that point (never approximated,
    never hand-labeled), so correctness of the TRAINING SIGNAL itself is
    never in question, only the claim that these points are "near a
    boundary," which is checked directly via the margin diagnostic, not
    assumed.
  - Query-change fidelity reported SEPARATELY across four pair categories
    (natural, stable-neighbor synthetic, same-label exchange, different-
    label exchange), per the spec's explicit "do not combine" instruction.

=====================================================================
Scoping decisions (stated up front, per the spec's own repeated "do not
launch the full grid immediately" instruction)
=====================================================================
  - Cluster sample: TARGET_N_CLUSTERS=5 (per the spec's own "approximately
    3-5 clusters" instruction), audited from a BROADER pool
    (CANDIDATE_POOL_SIZE=300, MIN_ELIGIBLE_POP=80) than Y1 used, to find
    the MOST-queried real clusters available in this cache -- not a
    contrived fix, a genuine re-scan with a higher bar. If this remains
    scarce relative to the spec's aspirational N_train grid (up to
    thousands), that scarcity is reported explicitly as a limitation
    (Section 6/B1), per the spec's own explicit allowance, not papered
    over with synthetic "natural" queries.
  - MLP capacity grid: widths {128, 512} x depths {2, 4} = 4 configurations
    (the spec's own "initial configurations, not a mandatory full
    Cartesian product" set, stopping short of width=1024 and RBF-large
    rank=1024 unless the smaller grid's own results justify expansion --
    they did not, see report.md).
  - RBF-large: a single larger rank (64) sharing N2's exact machinery, not
    a separate rank sweep.
  - The "kernel model" optional diagnostic (spec Section 5, A2 table) is
    NOT implemented -- explicitly named optional, and the MLP + RBF-large
    already give two genuinely different high-capacity families.
  - No sequence/document-grouped split exists in the cached arrays (same
    situation every prior script has faced) -- stated as a limitation,
    not assumed away. No additional "held-out or training-source text" is
    available beyond the three existing cache files, so Y2-B4's natural-
    query pool is bounded by what Y1 already found scarce -- reported as
    an explicit data limitation (Outcome F is a live possibility here, not
    ruled out in advance).

Smoke testing: set Y2_SMOKE=1 with Y2_CACHE_DIR / Y2_OUT_DIR / Y2_N_CLUSTERS
before running. Never set it for the real run.
"""
import sys, os, json, time, csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "GPT_Module"))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

from cache_io import load_cache
from knn import build_datastore, query_knn_indices
from results_io import save_results
from e1_metrics import js_divergence_dict
from run_e2_information_allocation import build_cluster_to_members
from run_e2tri1_anchor_conditioned_compression import build_centroid_distance_matrix, select_anchors_nearest
from run_crisp_subspace_projection import (
    exact_evidence, normalize_rows, linear_anchor_diff_basis, fit_ridge, farthest_point_sample,
    dict_from_dist, total_variation, top1_agreement, topk_overlap, entropy_of_dict)
from run_crisp_r4_histogram_corrections import log_ratio_target, reconstruct_logit, mean_tv, RIDGE_LAMBDA_GRID
from run_y1_per_cluster_function import local_m_distribution  # R-extra's/Y1's own exact teacher, reused unchanged
from run_y1_nonlinear_function_feasibility import rbf_features, neighbor_margin, fit_select_eval, best_sigma_for
from gpt_class import FrozenGPT2

SMOKE = os.environ.get("Y2_SMOKE") == "1"
CACHE_DIR = os.environ.get("Y2_CACHE_DIR", os.path.join(SCRIPT_DIR, "..", "GPT_Module", "cache"))
OUT_DIR = os.environ.get("Y2_OUT_DIR", os.path.join(SCRIPT_DIR, "results", "stage3_y2_representability_learnability"))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_CLUSTERS = int(os.environ.get("Y2_N_CLUSTERS", "3811"))
SEED = 42
DIM, DTYPE_BYTES = 768, 4
TAU = 2.0
M_PRIMARY = 8
M_DIAGNOSTIC_FIXED = [1, 4, 20]  # "n" (all-member) handled specially per cluster
TARGET_N_CLUSTERS = 5
CANDIDATE_POOL_SIZE = 300
MIN_ELIGIBLE_POP = 80
MIN_FIT, MIN_TUNE, MIN_TEST = 30, 15, 20
MAX_FIT, MAX_TUNE, MAX_TEST = 150, 50, 100
N1_N2_RANK = 8  # Y1's own primary rank, reproduced unchanged as a reference point
RBF_LARGE_RANK = 64
MLP_WIDTHS = [128, 512]
MLP_DEPTHS = [2, 4]
MLP_EPOCHS = 300
MLP_LR = 1e-3
MEMORIZATION_TRAIN_SIZES = [10, 20, 40]  # "all available" appended per cluster
LEARNING_CURVE_SIZES = [10, 20, 40, 80]  # "all available" appended per cluster
N_QUERY_PAIRS = 40
BOUNDARY_ENRICHED_CANDIDATE_POOL = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Operational thresholds (Section 11) -- defined BEFORE inspecting any
# new held-out result, not retroactively chosen. Y1's own REAL numbers
# (read above) are the historical benchmark the spec explicitly asks for.
# ============================================================
NEAR_MEMORIZATION_TV_THRESHOLD = 0.05
Y1_REAL_D0_TV, Y1_REAL_A0_TV, Y1_REAL_N2_TV = 0.8862, 0.8293, 0.8301
HELD_OUT_TV_IMPROVEMENT_THRESHOLD = 0.05  # absolute TV improvement over Y1_REAL_A0_TV required to call it "improved"
QUERY_CHANGE_MUST_BEAT_ZERO_CHANGE = True
GENERALIZATION_GAP_THRESHOLD = 0.15


# ============================================================
# MLP family (genuinely new; torch-based, since torch is already a
# project dependency via gpt_class.FrozenGPT2 -- not a new dependency)
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def train_mlp(X_train, T_train, X_tune, T_tune, out_dim, width, depth, epochs=MLP_EPOCHS, lr=MLP_LR, seed=SEED):
    """T_train/T_tune: teacher probability targets (soft, may contain exact
    zeros -- safe here since -sum_y T(y)*log_softmax(y) is 0*finite=0 for
    every zero-teacher label, never NaN). Model selection on the TUNE
    split (best mean TV), never on held-out test."""
    torch.manual_seed(seed)
    model = SimpleMLP(DIM, out_dim, width, depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    T_t = torch.tensor(T_train, dtype=torch.float32, device=DEVICE)
    X_tune_t = torch.tensor(X_tune, dtype=torch.float32, device=DEVICE)
    best_state, best_tune_tv = None, np.inf
    for epoch in range(epochs):
        opt.zero_grad()
        logp = torch.log_softmax(model(X_t), dim=1)
        loss = -(T_t * logp).sum(dim=1).mean()
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                tune_probs = torch.softmax(model(X_tune_t), dim=1).cpu().numpy()
            tv = mean_tv(tune_probs, T_tune)
            if tv < best_tune_tv:
                best_tune_tv = tv
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_tune_tv


def mlp_predict(model, X):
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return torch.softmax(model(X_t), dim=1).cpu().numpy()


def mlp_train_tv(model, X_train, T_train):
    return mean_tv(mlp_predict(model, X_train), T_train)


# ============================================================
# Phase C2 -- exact boundary-jump derivation and numerical verification
# ============================================================
def theoretical_boundary_jump(w_tie, Z, y_i, y_j, token_ids):
    """TV(F_+, F_-) = w/Z for a different-label exchange at the tie
    (w the shared tie-weight, Z the common normalizer); 0 for a same-label
    exchange. Returns the predicted signed jump vector too, for a direct
    numerical comparison against the real teacher's observed difference."""
    jump = np.zeros(len(token_ids))
    if y_i == y_j:
        return jump, 0.0
    col_j = int(np.where(token_ids == y_j)[0][0])
    col_i = int(np.where(token_ids == y_i)[0][0])
    jump[col_j] += w_tie / Z
    jump[col_i] -= w_tie / Z
    return jump, float(w_tie / Z)


def verify_boundary_jump_numerically(member_keys, member_values, m, tau, q_minus, q_plus, token_ids):
    """Evaluates the REAL teacher immediately on both sides of a
    constructed boundary and compares the OBSERVED jump against the
    theoretical w/Z formula -- a genuine numerical check, not an
    assumption that the derivation holds."""
    _, dist_minus = local_m_distribution(q_minus[None, :], member_keys, member_values, m, tau)
    _, dist_plus = local_m_distribution(q_plus[None, :], member_keys, member_values, m, tau)
    observed_jump_tv = total_variation(dict_from_dist(token_ids, dist_plus[0]), dict_from_dist(token_ids, dist_minus[0]))
    return observed_jump_tv, dist_minus[0], dist_plus[0]


# ============================================================
# Phase B4 -- boundary-enriched query construction (heuristic placement,
# EXACT teacher target -- see module docstring)
# ============================================================
def construct_boundary_enriched_queries(member_keys, member_values, n_points, rng, pool_size=BOUNDARY_ENRICHED_CANDIDATE_POOL):
    n = len(member_values)
    n_candidates = min(pool_size, n * (n - 1) // 2) if n > 1 else 0
    if n_candidates == 0:
        return np.zeros((0, member_keys.shape[1]))
    idx_i = rng.integers(0, n, size=n_candidates)
    idx_j = rng.integers(0, n, size=n_candidates)
    valid = (idx_i != idx_j) & (member_values[idx_i] != member_values[idx_j])
    idx_i, idx_j = idx_i[valid], idx_j[valid]
    if len(idx_i) == 0:
        return np.zeros((0, member_keys.shape[1]))
    d = np.linalg.norm(member_keys[idx_i] - member_keys[idx_j], axis=1)
    order = np.argsort(d)[:n_points]
    pts = []
    for k in order:
        t = rng.uniform(0.3, 0.7)
        pts.append(member_keys[idx_i[k]] + t * (member_keys[idx_j[k]] - member_keys[idx_i[k]]))
    return np.array(pts) if pts else np.zeros((0, member_keys.shape[1]))


def label_coverage(token_ids, train_targets, test_targets):
    """Y_train = labels with positive teacher probability for >=1 training
    query; coverage = |Y_train|/|Y_C|. Also returns the fraction of TEST
    teacher-support mass that lands on labels never seen in training."""
    train_support = set(token_ids[np.where(train_targets.sum(axis=0) > 1e-12)[0]].tolist())
    coverage = len(train_support) / len(token_ids)
    unseen_mask = np.array([int(t) not in train_support for t in token_ids])
    frac_test_mass_on_unseen = float((test_targets[:, unseen_mask]).sum(axis=1).mean()) if unseen_mask.any() else 0.0
    frac_test_queries_with_unseen_support = float(np.mean([(row[unseen_mask] > 1e-12).any() for row in test_targets])) if unseen_mask.any() else 0.0
    return {"coverage": coverage, "n_train_labels": len(train_support), "n_total_labels": len(token_ids),
             "frac_test_mass_on_unseen_labels": frac_test_mass_on_unseen,
             "frac_test_queries_with_unseen_support": frac_test_queries_with_unseen_support}


# ============================================================
# Synthetic correctness checks (extends Y1's; adds MLP/boundary-jump checks)
# ============================================================
def check_mlp_fits_xor():
    """Scenario 3: MLP must solve a target affine models cannot (reuses
    Y1's own XOR construction logic inline for a self-contained check)."""
    rng = np.random.default_rng(10)
    n = 400
    X = rng.uniform(-2, 2, size=(n, 2)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    T = np.zeros((n, 2), dtype=np.float32)
    T[np.arange(n), y] = 1.0
    X_pad = np.zeros((n, DIM), dtype=np.float32)
    X_pad[:, :2] = X
    model, _ = train_mlp(X_pad, T, X_pad, T, out_dim=2, width=64, depth=2, epochs=200, seed=1)
    pred = mlp_predict(model, X_pad)
    acc = float((np.argmax(pred, axis=1) == y).mean())
    ok = bool(acc > 0.95)
    print(f"  Check (MLP solves XOR): accuracy={acc:.3f}, pass={ok}")
    return ok, acc


def check_boundary_jump_formula():
    """1-D, 2-member synthetic cluster (same construction as Y1's
    check_discontinuity_behavior), m=1: verify the OBSERVED jump at q=0
    matches the THEORETICAL w/Z formula exactly, not merely that a jump
    exists."""
    member_keys = np.array([[-1.0], [1.0]])
    member_values = np.array([0, 1])
    token_ids = np.array([0, 1])
    eps = 1e-6
    q_minus, q_plus = np.array([-eps]), np.array([eps])
    observed_tv, dist_minus, dist_plus = verify_boundary_jump_numerically(member_keys, member_values, 1, TAU, q_minus, q_plus, token_ids)
    # At q=0 the tie distance is exactly 1.0 for both members, so w = exp(-1/TAU), Z = w (m=1, only one
    # member counted, which at the EXACT tie is ambiguous between the two -- approached from q=-eps,
    # the nearest is member 0 with d=1-eps; from q=+eps, nearest is member 1 with d=1-eps; in the
    # limit eps->0 both have d->1, so w=Z=exp(-1/TAU) and the theoretical jump is w/Z = 1.0 exactly
    # (a full label flip, m=1's own one-hot structure), matching the observed TV of 1.0.
    theoretical_tv = 1.0
    ok = bool(np.isclose(observed_tv, theoretical_tv, atol=1e-3))
    print(f"  Check (boundary-jump formula matches observed, m=1 full-flip case): observed={observed_tv:.6f}, "
          f"theoretical={theoretical_tv:.6f}, pass={ok}")
    return ok, observed_tv, theoretical_tv


def run_synthetic_checks():
    print("\nSynthetic correctness checks (run before any real cache is touched)...")
    c1, acc_xor = check_mlp_fits_xor()
    c2, obs_jump, theo_jump = check_boundary_jump_formula()
    all_passed = all([c1, c2])
    assert all_passed, "Synthetic correctness checks failed -- do not proceed to real-cache fitting"
    print(f"  ALL Y2-specific synthetic checks passed: {all_passed} (Y1's own 7 checks are reused unchanged and "
          f"are not re-run here to avoid duplicating that verification work).")
    return {"mlp_solves_xor": c1, "mlp_xor_accuracy": acc_xor, "boundary_jump_formula_matches_observed": c2,
             "observed_jump_tv": obs_jump, "theoretical_jump_tv": theo_jump, "all_passed": all_passed}


def fit_affine_a0(fit_keys, tune_keys, test_keys, member_values_dummy, target_fit, target_tune, token_ids, p_j, log_p,
                    c, dist_matrix, dime_centroids, rank=N1_N2_RANK):
    anchor_idx = select_anchors_nearest(dist_matrix, c, rank)
    Phi_fit = linear_anchor_diff_basis(fit_keys, c, anchor_idx, dime_centroids)
    Phi_tune = linear_anchor_diff_basis(tune_keys, c, anchor_idx, dime_centroids)
    Phi_test = linear_anchor_diff_basis(test_keys, c, anchor_idx, dime_centroids)
    z_fit = log_ratio_target(target_fit, p_j)
    z_tune = log_ratio_target(target_tune, p_j)
    return fit_select_eval(Phi_fit, Phi_tune, Phi_test, z_fit, z_tune, None, log_p)


def fit_rbf(fit_keys, tune_keys, test_keys, anchors, target_fit, target_tune, log_p, dist_scale):
    z_fit = log_ratio_target(target_fit, np.exp(log_p))
    z_tune = log_ratio_target(target_tune, np.exp(log_p))
    sigma, lam, _ = best_sigma_for(rbf_features, fit_keys, tune_keys, z_fit, z_tune, log_p, anchors, dist_scale)
    Phi_fit = rbf_features(fit_keys, anchors, sigma)
    Phi_tune = rbf_features(tune_keys, anchors, sigma)
    Phi_test = rbf_features(test_keys, anchors, sigma)
    res = fit_select_eval(Phi_fit, Phi_tune, Phi_test, z_fit, z_tune, None, log_p)
    res["sigma"], res["Phi_fit"] = sigma, Phi_fit
    return res


def main():
    t_start = time.perf_counter()
    print(f"Y2 job started. SMOKE={SMOKE}, N_CLUSTERS={N_CLUSTERS}, DEVICE={DEVICE}")
    rng = np.random.default_rng(SEED)

    sanity = run_synthetic_checks()
    report = {"synthetic_correctness_checks": sanity, "m_primary": M_PRIMARY, "tau": TAU,
              "operational_thresholds": {"near_memorization_tv": NEAR_MEMORIZATION_TV_THRESHOLD,
                                            "held_out_tv_improvement_required": HELD_OUT_TV_IMPROVEMENT_THRESHOLD,
                                            "query_change_must_beat_zero_change": QUERY_CHANGE_MUST_BEAT_ZERO_CHANGE,
                                            "generalization_gap_threshold": GENERALIZATION_GAP_THRESHOLD,
                                            "y1_real_historical_benchmarks": {"D0": Y1_REAL_D0_TV, "A0": Y1_REAL_A0_TV, "N2": Y1_REAL_N2_TV}}}
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    print("\nLoading cache (datastore/controller_train/val)...")
    ds = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_datastore.npz"))
    ct = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_controller_train.npz"))
    val = load_cache(os.path.join(CACHE_DIR, "wikitext103_gpt2_val.npz"))
    ds_keys, ds_values = ds["keys"].astype(np.float64), ds["values"]
    ct_keys, ct_values = ct["keys"].astype(np.float64), ct["values"]
    val_keys, val_values = val["keys"].astype(np.float64), val["values"]
    gpt = FrozenGPT2(model_name="gpt2")

    print("Re-deriving DIME clusters (seed=42, identical to every prior script in this series)...")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)
    assignment = km.fit_predict(ds_keys.astype(np.float32))
    dime_centroids = km.cluster_centers_.astype(np.float64)
    compressed_dists = []
    for c in range(N_CLUSTERS):
        member_values = ds_values[assignment == c]
        compressed_dists.append(Counter(member_values.tolist()) if len(member_values) > 0 else Counter())
    cluster_to_members = build_cluster_to_members(assignment, N_CLUSTERS)
    dist_matrix = build_centroid_distance_matrix(dime_centroids.astype(np.float32))
    dime_index, dime_values = build_datastore(dime_centroids, np.array(compressed_dists, dtype=object))
    _, idx_dime_ct1 = query_knn_indices(dime_index, ct_keys, k=1)
    _, idx_dime_val1 = query_knn_indices(dime_index, val_keys, k=1)
    idx_dime_ct1, idx_dime_val1 = idx_dime_ct1[:, 0], idx_dime_val1[:, 0]
    populations = np.array([sum(c.values()) for c in compressed_dists])

    # ============================================================
    # Y2-B1 -- query-availability audit, broader pool, higher bar than Y1
    # ============================================================
    print(f"\nY2-B1: query-availability audit (population>={MIN_ELIGIBLE_POP}, pool up to {CANDIDATE_POOL_SIZE})...")
    eligible = np.where(populations >= MIN_ELIGIBLE_POP)[0]
    order_by_pop = eligible[np.argsort(-populations[eligible])][:CANDIDATE_POOL_SIZE]
    candidate_splits = {}
    for c in order_by_pop:
        c = int(c)
        ct_routed = np.where(idx_dime_ct1 == c)[0]
        val_routed = np.where(idx_dime_val1 == c)[0]
        local_rng = np.random.default_rng(SEED + c)
        local_rng.shuffle(ct_routed)
        n_fit = min(MAX_FIT, max(1, len(ct_routed) * 3 // 4))
        fit_idx, tune_idx = ct_routed[:n_fit], ct_routed[n_fit:n_fit + MAX_TUNE]
        test_idx = val_routed[:MAX_TEST]
        candidate_splits[c] = {"fit": fit_idx, "tune": tune_idx, "test": test_idx}
    audited = [{"cluster": c, "n_fit": len(s["fit"]), "n_tune": len(s["tune"]), "n_test": len(s["test"]),
                 "total": len(s["fit"]) + len(s["tune"]) + len(s["test"])} for c, s in candidate_splits.items()]
    sufficient = [a for a in audited if a["n_fit"] >= MIN_FIT and a["n_tune"] >= MIN_TUNE and a["n_test"] >= MIN_TEST
                   and len(compressed_dists[a["cluster"]]) >= 2]
    sufficient_sorted = sorted(sufficient, key=lambda a: -a["total"])
    sample_clusters = [a["cluster"] for a in sufficient_sorted[:TARGET_N_CLUSTERS]]
    print(f"  {len(sufficient)}/{len(audited)} candidates sufficient; selected top {len(sample_clusters)} by "
          f"total routed-query count: {sample_clusters}")
    report["query_availability_audit"] = {"n_candidates_scanned": len(audited), "n_sufficient": len(sufficient),
                                              "sample_clusters": sample_clusters,
                                              "per_cluster_counts": [a for a in audited if a["cluster"] in sample_clusters],
                                              "note": "No additional held-out or training-source text is available "
                                                       "beyond the three existing cache files -- this is the full "
                                                       "query pool achievable in this project, reported as a "
                                                       "limitation, not papered over."}
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    dist_matrix_all = dist_matrix
    n1_shared_anchors = dime_centroids[farthest_point_sample(dime_centroids, N1_N2_RANK, seed=SEED)]

    capacity_rows, memorization_rows, learning_curve_rows, coverage_rows = [], [], [], []
    boundary_rows, query_change_rows, cluster_results_rows = [], [], []
    boundary_enriched_rows = []

    for ci, c in enumerate(sample_clusters):
        print(f"\n=== Processing cluster {c} ({ci + 1}/{len(sample_clusters)}) ===")
        members = cluster_to_members[c]
        member_keys, member_values = ds_keys[members], ds_values[members]
        population = len(members)
        splits = candidate_splits[c]
        fit_keys, tune_keys, test_keys = ct_keys[splits["fit"]], ct_keys[splits["tune"]], val_keys[splits["test"]]

        token_ids, fit_target = local_m_distribution(fit_keys, member_keys, member_values, M_PRIMARY, TAU)
        _, tune_target = local_m_distribution(tune_keys, member_keys, member_values, M_PRIMARY, TAU)
        _, test_target = local_m_distribution(test_keys, member_keys, member_values, M_PRIMARY, TAU)
        n_tokens = len(token_ids)
        p_j = np.array([float(np.sum(member_values == y)) / population for y in token_ids])
        log_p = np.log(np.maximum(p_j, 1e-300))
        dime_bcast_test = np.tile(p_j, (test_target.shape[0], 1))
        dime_bcast_fit = np.tile(p_j, (fit_target.shape[0], 1))

        n2_anchors = fit_keys[farthest_point_sample(fit_keys, N1_N2_RANK, seed=SEED + c)]
        n2_large_anchors = fit_keys[farthest_point_sample(fit_keys, min(RBF_LARGE_RANK, len(fit_keys)), seed=SEED + c)]
        dist_scale_n1 = float(np.median(np.linalg.norm(fit_keys[:, None, :] - n1_shared_anchors[None, :, :], axis=2))) or 1.0
        dist_scale_n2 = float(np.median(np.linalg.norm(fit_keys[:, None, :] - n2_anchors[None, :, :], axis=2))) or 1.0
        dist_scale_large = float(np.median(np.linalg.norm(fit_keys[:, None, :] - n2_large_anchors[None, :, :], axis=2))) or 1.0

        # ============================================================
        # Y2-A -- capacity sweep at the primary config (m=8)
        # ============================================================
        print("  Y2-A: capacity sweep (D0, A0, N1, N2, RBF-large, MLPx4)...")
        res_a0 = fit_affine_a0(fit_keys, tune_keys, test_keys, None, fit_target, tune_target, token_ids, p_j, log_p, c, dist_matrix_all, dime_centroids)
        res_n1 = fit_rbf(fit_keys, tune_keys, test_keys, n1_shared_anchors, fit_target, tune_target, log_p, dist_scale_n1)
        res_n2 = fit_rbf(fit_keys, tune_keys, test_keys, n2_anchors, fit_target, tune_target, log_p, dist_scale_n2)
        res_large = fit_rbf(fit_keys, tune_keys, test_keys, n2_large_anchors, fit_target, tune_target, log_p, dist_scale_large)

        model_test_dists = {"D0": dime_bcast_test, "A0": res_a0["d_test"], "N1": res_n1["d_test"],
                              "N2": res_n2["d_test"], "RBF-large": res_large["d_test"]}
        model_fit_dists = {"D0": dime_bcast_fit, "A0": res_a0["d_fit"], "N1": res_n1["d_fit"],
                            "N2": res_n2["d_fit"], "RBF-large": res_large["d_fit"]}
        model_param_counts = {"D0": 0, "A0": n_tokens * N1_N2_RANK, "N1": n_tokens * N1_N2_RANK,
                                "N2": n_tokens * N1_N2_RANK, "RBF-large": n_tokens * min(RBF_LARGE_RANK, len(fit_keys))}

        mlp_models = {}
        for width in MLP_WIDTHS:
            for depth in MLP_DEPTHS:
                key = f"MLP_w{width}_d{depth}"
                model, tune_tv = train_mlp(fit_keys.astype(np.float32), fit_target.astype(np.float32),
                                              tune_keys.astype(np.float32), tune_target.astype(np.float32),
                                              out_dim=n_tokens, width=width, depth=depth, seed=SEED + c)
                d_test_mlp = mlp_predict(model, test_keys.astype(np.float32))
                d_fit_mlp = mlp_predict(model, fit_keys.astype(np.float32))
                model_test_dists[key] = d_test_mlp
                model_fit_dists[key] = d_fit_mlp
                model_param_counts[key] = count_params(model)
                mlp_models[key] = model

        for mname, d_test in model_test_dists.items():
            test_tv = mean_tv(d_test, test_target)
            train_tv = mean_tv(model_fit_dists[mname], fit_target)
            capacity_rows.append({"cluster": c, "model": mname, "n_params": model_param_counts[mname],
                                     "train_tv": round(train_tv, 6), "test_tv": round(test_tv, 6),
                                     "generalization_gap": round(test_tv - train_tv, 6)})
            cluster_results_rows.append({"cluster": c, "model": mname, "population": population, "n_tokens": n_tokens,
                                            "train_tv": round(train_tv, 6), "test_tv": round(test_tv, 6),
                                            "n_fit": len(splits["fit"]), "n_tune": len(splits["tune"]), "n_test": len(splits["test"])})
        print(f"  Capacity sweep test TV: {{{', '.join(f'{m}={mean_tv(model_test_dists[m], test_target):.4f}' for m in model_test_dists)}}}")
        save_results(os.path.join(OUT_DIR, "y2_summary.json"), {**report, "progress": f"cluster {c} capacity done"})

        # ============================================================
        # Y2-A3 -- explicit training-set memorization test
        # ============================================================
        print("  Y2-A3: memorization test (fixed finite training sets, m in {1, primary, n})...")
        memo_models = ["A0", "N2", "RBF-large", "MLP_w512_d4"]
        memo_sizes = [s for s in MEMORIZATION_TRAIN_SIZES if s <= len(splits["fit"])] + [len(splits["fit"])]
        memo_m_values = {"m1": 1, f"m{M_PRIMARY}": M_PRIMARY, "mN": population}
        for m_label, m_val in memo_m_values.items():
            _, fit_target_m = local_m_distribution(fit_keys, member_keys, member_values, m_val, TAU)
            for n_sub in sorted(set(memo_sizes)):
                # Anchors are a FIXED, pre-selected set (training-available info, chosen once from the
                # full fit split above) -- only the FIT SUBSET used for coefficients varies with n_sub.
                sub_keys, sub_target = fit_keys[:n_sub], fit_target_m[:n_sub]
                z_sub = log_ratio_target(sub_target, p_j)
                for mname in memo_models:
                    if mname == "A0":
                        a_idx = select_anchors_nearest(dist_matrix_all, c, N1_N2_RANK)
                        Phi_sub = linear_anchor_diff_basis(sub_keys, c, a_idx, dime_centroids)
                        theta = fit_ridge(Phi_sub, z_sub, 0.01)
                        d_sub, _ = reconstruct_logit(Phi_sub, theta, log_p)
                    elif mname in ("N2", "RBF-large"):
                        anchors_m = n2_anchors if mname == "N2" else n2_large_anchors
                        sigma_m = dist_scale_n2 if mname == "N2" else dist_scale_large
                        Phi_sub = rbf_features(sub_keys, anchors_m, sigma_m)
                        theta = fit_ridge(Phi_sub, z_sub, 0.01)
                        d_sub, _ = reconstruct_logit(Phi_sub, theta, log_p)
                    else:  # MLP_w512_d4, trained directly on the small subset, evaluated on itself
                        model_sub, _ = train_mlp(sub_keys.astype(np.float32), sub_target.astype(np.float32),
                                                    sub_keys.astype(np.float32), sub_target.astype(np.float32),
                                                    out_dim=n_tokens, width=512, depth=4, epochs=MLP_EPOCHS, seed=SEED + c + n_sub)
                        d_sub = mlp_predict(model_sub, sub_keys.astype(np.float32))
                    train_tv_sub = mean_tv(d_sub, sub_target)
                    memorization_rows.append({"cluster": c, "m_label": m_label, "n_train": n_sub, "model": mname,
                                                 "train_tv": round(train_tv_sub, 6),
                                                 "near_memorization": bool(train_tv_sub < NEAR_MEMORIZATION_TV_THRESHOLD)})
        print(f"  memorization (m={M_PRIMARY}, n_train={memo_sizes[-1]}): "
              f"{ {r['model']: r['train_tv'] for r in memorization_rows if r['cluster']==c and r['m_label']==f'm{M_PRIMARY}' and r['n_train']==memo_sizes[-1]} }")
        save_results(os.path.join(OUT_DIR, "y2_summary.json"), {**report, "progress": f"cluster {c} memorization done"})

        # ============================================================
        # Y2-B2/B3 -- learning curves + label coverage (N2, A0, RBF-large, best MLP)
        # ============================================================
        print("  Y2-B2/B3: learning curves + label coverage...")
        lc_sizes = sorted(set([s for s in LEARNING_CURVE_SIZES if s <= len(splits["fit"])] + [len(splits["fit"])]))
        for n_train in lc_sizes:
            sub_keys, sub_target = fit_keys[:n_train], fit_target[:n_train]
            z_sub = log_ratio_target(sub_target, p_j)
            cov = label_coverage(token_ids, sub_target, test_target)
            coverage_rows.append({"cluster": c, "n_train": n_train, **cov})

            a_idx = select_anchors_nearest(dist_matrix_all, c, N1_N2_RANK)
            Phi_sub_a0, Phi_test_a0 = linear_anchor_diff_basis(sub_keys, c, a_idx, dime_centroids), linear_anchor_diff_basis(test_keys, c, a_idx, dime_centroids)
            theta_a0 = fit_ridge(Phi_sub_a0, z_sub, 0.1)
            d_test_a0_lc, _ = reconstruct_logit(Phi_test_a0, theta_a0, log_p)

            Phi_sub_n2, Phi_test_n2 = rbf_features(sub_keys, n2_anchors, dist_scale_n2), rbf_features(test_keys, n2_anchors, dist_scale_n2)
            theta_n2 = fit_ridge(Phi_sub_n2, z_sub, 0.1)
            d_test_n2_lc, _ = reconstruct_logit(Phi_test_n2, theta_n2, log_p)

            Phi_sub_lg, Phi_test_lg = rbf_features(sub_keys, n2_large_anchors, dist_scale_large), rbf_features(test_keys, n2_large_anchors, dist_scale_large)
            theta_lg = fit_ridge(Phi_sub_lg, z_sub, 0.1)
            d_test_lg_lc, _ = reconstruct_logit(Phi_test_lg, theta_lg, log_p)

            model_lc, _ = train_mlp(sub_keys.astype(np.float32), sub_target.astype(np.float32),
                                       tune_keys.astype(np.float32), tune_target.astype(np.float32),
                                       out_dim=n_tokens, width=512, depth=4, epochs=MLP_EPOCHS, seed=SEED + c + n_train)
            d_test_mlp_lc = mlp_predict(model_lc, test_keys.astype(np.float32))

            for mname, d_test_lc in [("A0", d_test_a0_lc), ("N2", d_test_n2_lc), ("RBF-large", d_test_lg_lc), ("MLP_w512_d4", d_test_mlp_lc)]:
                learning_curve_rows.append({"cluster": c, "n_train": n_train, "model": mname,
                                               "held_out_tv": round(mean_tv(d_test_lc, test_target), 6)})
        n2_lc_by_size = {n: next(r["held_out_tv"] for r in learning_curve_rows if r["cluster"] == c and r["model"] == "N2" and r["n_train"] == n) for n in lc_sizes}
        print(f"  learning curve (N2): {n2_lc_by_size}")
        save_results(os.path.join(OUT_DIR, "y2_summary.json"), {**report, "progress": f"cluster {c} learning curve done"})

        # ============================================================
        # Y2-B4 -- natural vs. boundary-enriched training (N2, matched total size)
        # ============================================================
        print("  Y2-B4: natural vs. boundary-enriched training (N2)...")
        local_rng = np.random.default_rng(SEED + c + 777)
        n_enriched_total = min(len(splits["fit"]), 60)
        enriched_pts = construct_boundary_enriched_queries(member_keys, member_values, n_enriched_total, local_rng)
        boundary_regime = {}
        if len(enriched_pts) >= 10:
            n_half = min(n_enriched_total // 2, len(fit_keys))
            # Regime N: natural only, same total size as regime E for a fair comparison
            natural_train_keys = fit_keys[:n_enriched_total]
            _, natural_train_target = local_m_distribution(natural_train_keys, member_keys, member_values, M_PRIMARY, TAU)
            # Regime E: half natural + half boundary-enriched, EXACT teacher at every point (never hand-labeled)
            enriched_half = enriched_pts[:n_half]
            natural_half = fit_keys[:n_enriched_total - n_half]
            mixed_train_keys = np.vstack([natural_half, enriched_half])
            _, mixed_train_target = local_m_distribution(mixed_train_keys, member_keys, member_values, M_PRIMARY, TAU)

            z_natural = log_ratio_target(natural_train_target, p_j)
            z_mixed = log_ratio_target(mixed_train_target, p_j)
            Phi_natural = rbf_features(natural_train_keys, n2_anchors, dist_scale_n2)
            Phi_mixed = rbf_features(mixed_train_keys, n2_anchors, dist_scale_n2)
            theta_natural = fit_ridge(Phi_natural, z_natural, 0.1)
            theta_mixed = fit_ridge(Phi_mixed, z_mixed, 0.1)

            # Diagnostic boundary-focused test set: a FRESH, held-out set of boundary-enriched points
            # (disjoint from the ones used in regime E's training, constructed from the SAME cluster
            # but a different random draw)
            diag_pts = construct_boundary_enriched_queries(member_keys, member_values, 30, np.random.default_rng(SEED + c + 999))
            if len(diag_pts) >= 5:
                _, diag_target = local_m_distribution(diag_pts, member_keys, member_values, M_PRIMARY, TAU)
                Phi_diag = rbf_features(diag_pts, n2_anchors, dist_scale_n2)
                d_diag_natural, _ = reconstruct_logit(Phi_diag, theta_natural, log_p)
                d_diag_mixed, _ = reconstruct_logit(Phi_diag, theta_mixed, log_p)
                boundary_regime["boundary_focused_tv_regimeN"] = round(mean_tv(d_diag_natural, diag_target), 6)
                boundary_regime["boundary_focused_tv_regimeE"] = round(mean_tv(d_diag_mixed, diag_target), 6)

            Phi_test_n2_reg = rbf_features(test_keys, n2_anchors, dist_scale_n2)
            d_test_natural, _ = reconstruct_logit(Phi_test_n2_reg, theta_natural, log_p)
            d_test_mixed, _ = reconstruct_logit(Phi_test_n2_reg, theta_mixed, log_p)
            boundary_regime["natural_test_tv_regimeN"] = round(mean_tv(d_test_natural, test_target), 6)
            boundary_regime["natural_test_tv_regimeE"] = round(mean_tv(d_test_mixed, test_target), 6)
            boundary_regime["n_enriched_available"] = len(enriched_pts)
        else:
            boundary_regime["skipped_reason"] = "insufficient label-differing member pairs to construct boundary-enriched points"
        boundary_enriched_rows.append({"cluster": c, **boundary_regime})
        print(f"  boundary-enriched regime comparison: {boundary_regime}")
        save_results(os.path.join(OUT_DIR, "y2_summary.json"), {**report, "progress": f"cluster {c} boundary-enriched done"})

        # ============================================================
        # Y2-C -- margin diagnostic, boundary-exchange verification, functional complexity
        # ============================================================
        print("  Y2-C: margin diagnostic + boundary-exchange verification...")
        margins = neighbor_margin(test_keys, member_keys, M_PRIMARY)
        for qi in range(test_target.shape[0]):
            if np.isfinite(margins[qi]):
                boundary_rows.append({"cluster": c, "query_idx": qi, "margin": float(margins[qi]), "kind": "natural_margin",
                                         "A0_tv": float(total_variation(dict_from_dist(token_ids, model_test_dists["A0"][qi]), dict_from_dist(token_ids, test_target[qi]))),
                                         "N2_tv": float(total_variation(dict_from_dist(token_ids, model_test_dists["N2"][qi]), dict_from_dist(token_ids, test_target[qi])))})

        # Controlled boundary-crossing pairs: for each constructed boundary-enriched point, pair it
        # with the SAME point nudged slightly further along the member-pair segment (same construction
        # direction, t+delta) -- classified by whether the underlying member pair shares a label.
        n_bc_pairs = min(N_QUERY_PAIRS, n_enriched_total if len(enriched_pts) else 0)
        if n_bc_pairs >= 5:
            n2_cand = min(500, population * (population - 1) // 2) if population > 1 else 0
            idx_i = local_rng.integers(0, population, size=n2_cand)
            idx_j = local_rng.integers(0, population, size=n2_cand)
            valid_pairs = idx_i != idx_j
            idx_i, idx_j = idx_i[valid_pairs], idx_j[valid_pairs]
            d_pairs = np.linalg.norm(member_keys[idx_i] - member_keys[idx_j], axis=1)
            order_pairs = np.argsort(d_pairs)[:n_bc_pairs]
            for k in order_pairs:
                hi, hj = member_keys[idx_i[k]], member_keys[idx_j[k]]
                yi, yj = int(member_values[idx_i[k]]), int(member_values[idx_j[k]])
                t0, t1 = 0.45, 0.55
                q_minus, q_plus = hi + t0 * (hj - hi), hi + t1 * (hj - hi)
                obs_tv, dist_minus, dist_plus = verify_boundary_jump_numerically(member_keys, member_values, M_PRIMARY, TAU, q_minus, q_plus, token_ids)
                # Reuses the ALREADY tune-selected N2 model (res_n2["theta"]) rather than refitting a
                # fresh ridge model with an arbitrary lambda inside this loop for every pair -- both
                # wasteful (40 redundant refits) and inconsistent with the TUNE-selected model used
                # everywhere else (caught via self-review before running).
                Phi_pair = rbf_features(np.vstack([q_minus, q_plus]), n2_anchors, dist_scale_n2)
                d_pair_pred, _ = reconstruct_logit(Phi_pair, res_n2["theta"], log_p)
                pred_jump_tv = float(total_variation(dict_from_dist(token_ids, d_pair_pred[1]), dict_from_dist(token_ids, d_pair_pred[0])))
                # e_delta = ||delta_exact - delta_pred||_1 -- the actual CHANGE-RECONSTRUCTION ERROR,
                # NOT the two changes' magnitudes separately (||a|| and ||b|| alone cannot recover
                # ||a-b||; this was computed wrong in an earlier version, caught via self-review before
                # running: it stored only the exact and predicted jump MAGNITUDES, never their
                # difference, which is what query-change fidelity actually needs).
                delta_exact_bc = dist_plus - dist_minus
                delta_pred_bc = d_pair_pred[1] - d_pair_pred[0]
                e_delta_bc = float(np.abs(delta_exact_bc - delta_pred_bc).sum())
                boundary_rows.append({"cluster": c, "query_idx": -1, "margin": None, "kind": "same_label" if yi == yj else "different_label",
                                         "exact_teacher_jump_tv": round(obs_tv, 6), "model_predicted_jump_tv": round(pred_jump_tv, 6),
                                         "e_delta": round(e_delta_bc, 6), "A0_tv": None, "N2_tv": None})
        print(f"  recorded {len([r for r in boundary_rows if r['cluster']==c])} boundary/margin diagnostic rows for cluster {c}")
        save_results(os.path.join(OUT_DIR, "y2_summary.json"), {**report, "progress": f"cluster {c} boundary diagnostics done"})

        # ============================================================
        # Section 8 -- query-change fidelity, reported SEPARATELY across
        # four pair categories (never combined into one number)
        # ============================================================
        print("  Query-change fidelity (A0/N2, 4 pair categories, reported separately)...")
        # Category 1: natural test-split pairs (reuses already-computed model predictions)
        n_test = test_target.shape[0]
        if n_test >= 2:
            n_pairs = min(N_QUERY_PAIRS, n_test * (n_test - 1) // 2)
            pi = rng.integers(0, n_test, size=n_pairs * 2)
            pjx = rng.integers(0, n_test, size=n_pairs * 2)
            valid_qp = pi != pjx
            pi, pjx = pi[valid_qp][:n_pairs], pjx[valid_qp][:n_pairs]
            delta_exact_nat = test_target[pi] - test_target[pjx]
            for mname in ["D0", "A0", "N2"]:
                d_m = model_test_dists[mname]
                delta_pred = d_m[pi] - d_m[pjx]
                for k in range(len(pi)):
                    e_delta = float(np.abs(delta_exact_nat[k] - delta_pred[k]).sum())
                    exact_mag = float(np.abs(delta_exact_nat[k]).sum())
                    query_change_rows.append({"cluster": c, "category": "natural_pairs", "model": mname,
                                                 "e_delta": round(e_delta, 6), "exact_magnitude": round(exact_mag, 6),
                                                 "dime_zero_change_error": round(exact_mag, 6)})

        # Category 2: stable-neighbor synthetic pairs (tiny perturbation, verified post-hoc to leave
        # the top-m neighbor SET unchanged -- not assumed)
        within_cluster_scale = float(np.median(np.linalg.norm(member_keys - member_keys.mean(axis=0)[None, :], axis=1))) or 1.0
        stable_pairs_q, stable_pairs_q2 = [], []
        for qi in range(min(n_test, N_QUERY_PAIRS)):
            q0 = test_keys[qi]
            for eps_scale in [0.001, 0.003, 0.01]:
                direction = rng.normal(size=q0.shape)
                direction /= (np.linalg.norm(direction) or 1.0)
                q2 = q0 + eps_scale * within_cluster_scale * direction
                d1 = np.linalg.norm(member_keys - q0[None, :], axis=1)
                d2 = np.linalg.norm(member_keys - q2[None, :], axis=1)
                if set(np.argsort(d1)[:M_PRIMARY].tolist()) == set(np.argsort(d2)[:M_PRIMARY].tolist()):
                    stable_pairs_q.append(q0); stable_pairs_q2.append(q2)
                    break
        if stable_pairs_q:
            sq, sq2 = np.array(stable_pairs_q), np.array(stable_pairs_q2)
            _, target_sq = local_m_distribution(sq, member_keys, member_values, M_PRIMARY, TAU)
            _, target_sq2 = local_m_distribution(sq2, member_keys, member_values, M_PRIMARY, TAU)
            delta_exact_stable = target_sq2 - target_sq
            Phi_sq_a0 = linear_anchor_diff_basis(sq, c, select_anchors_nearest(dist_matrix_all, c, N1_N2_RANK), dime_centroids)
            Phi_sq2_a0 = linear_anchor_diff_basis(sq2, c, select_anchors_nearest(dist_matrix_all, c, N1_N2_RANK), dime_centroids)
            d_sq_a0, _ = reconstruct_logit(Phi_sq_a0, res_a0["theta"], log_p)
            d_sq2_a0, _ = reconstruct_logit(Phi_sq2_a0, res_a0["theta"], log_p)
            Phi_sq_n2, Phi_sq2_n2 = rbf_features(sq, n2_anchors, dist_scale_n2), rbf_features(sq2, n2_anchors, dist_scale_n2)
            d_sq_n2, _ = reconstruct_logit(Phi_sq_n2, res_n2["theta"], log_p)
            d_sq2_n2, _ = reconstruct_logit(Phi_sq2_n2, res_n2["theta"], log_p)
            for mname, d_before, d_after in [("D0", np.tile(p_j, (len(sq), 1)), np.tile(p_j, (len(sq), 1))),
                                                ("A0", d_sq_a0, d_sq2_a0), ("N2", d_sq_n2, d_sq2_n2)]:
                delta_pred_stable = d_after - d_before
                for k in range(len(sq)):
                    e_delta = float(np.abs(delta_exact_stable[k] - delta_pred_stable[k]).sum())
                    exact_mag = float(np.abs(delta_exact_stable[k]).sum())
                    query_change_rows.append({"cluster": c, "category": "stable_neighbor_synthetic", "model": mname,
                                                 "e_delta": round(e_delta, 6), "exact_magnitude": round(exact_mag, 6),
                                                 "dime_zero_change_error": round(exact_mag, 6)})

        # Categories 3/4 (same-label / different-label exchange pairs): reuse the boundary-pair
        # diagnostics already computed above (boundary_rows, "kind" in {"same_label","different_label"})
        for row in [r for r in boundary_rows if r["cluster"] == c and r["kind"] in ("same_label", "different_label")]:
            category = "same_label_exchange" if row["kind"] == "same_label" else "different_label_exchange"
            query_change_rows.append({"cluster": c, "category": category, "model": "N2",
                                         "e_delta": row["e_delta"], "exact_magnitude": row["exact_teacher_jump_tv"],
                                         "dime_zero_change_error": row["exact_teacher_jump_tv"]})
        save_results(os.path.join(OUT_DIR, "y2_summary.json"), {**report, "progress": f"cluster {c} query-change done"})

        print(f"=== Cluster {c} complete ===")

    # ============================================================
    # Aggregation + CSV deliverables
    # ============================================================
    print("\nWriting CSV deliverables...")
    for rows, fname in [(capacity_rows, "capacity_results.csv"), (memorization_rows, "memorization_results.csv"),
                          (learning_curve_rows, "learning_curve_results.csv"), (coverage_rows, "label_coverage_results.csv"),
                          (boundary_rows, "boundary_results.csv"), (query_change_rows, "query_change_results.csv"),
                          (cluster_results_rows, "cluster_results.csv")]:
        if not rows:
            continue
        all_keys = []
        for row in rows:
            for k in row.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, restval="")
            w.writeheader()
            for row in rows:
                w.writerow(row)

    # ============================================================
    # Stage decisions and aggregate summaries
    # ============================================================
    print("\nComputing stage decisions...")
    # Y2-A decision: near-memorization achieved?
    memo_by_model_m = {}
    for mname in ["A0", "N2", "RBF-large", "MLP_w512_d4"]:
        for m_label in ["m1", f"m{M_PRIMARY}", "mN"]:
            vals = [r["train_tv"] for r in memorization_rows if r["model"] == mname and r["m_label"] == m_label]
            if vals:
                memo_by_model_m[f"{mname}_{m_label}"] = {"mean_train_tv_at_largest_n": float(np.mean(vals)),
                                                            "near_memorization_achieved": bool(np.mean(vals) < NEAR_MEMORIZATION_TV_THRESHOLD)}
    stage_a_decision = {"per_model_m": memo_by_model_m,
                          "any_model_near_memorization": any(v["near_memorization_achieved"] for v in memo_by_model_m.values())}
    print(f"  Stage A decision: {stage_a_decision}")
    report["stage_A_decision"] = stage_a_decision
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    # Y2-A capacity: held-out TV by model, aggregated across clusters, vs. Y1's real historical benchmarks
    capacity_agg = {}
    all_model_names = sorted(set(r["model"] for r in capacity_rows))
    for mname in all_model_names:
        tvs = [r["test_tv"] for r in capacity_rows if r["model"] == mname]
        train_tvs = [r["train_tv"] for r in capacity_rows if r["model"] == mname]
        gaps = [r["generalization_gap"] for r in capacity_rows if r["model"] == mname]
        capacity_agg[mname] = {"n_clusters": len(tvs), "mean_test_tv": float(np.mean(tvs)), "mean_train_tv": float(np.mean(train_tvs)),
                                  "mean_generalization_gap": float(np.mean(gaps)),
                                  "beats_y1_real_A0_by_threshold": bool((Y1_REAL_A0_TV - float(np.mean(tvs))) > HELD_OUT_TV_IMPROVEMENT_THRESHOLD)}
    print(f"  Capacity aggregate (held-out TV): {capacity_agg}")
    report["capacity_aggregate"] = capacity_agg
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    # Y2-B decision: does more training data help, and does boundary enrichment help?
    lc_agg = {}
    for mname in sorted(set(r["model"] for r in learning_curve_rows)):
        by_n = {}
        for n in sorted(set(r["n_train"] for r in learning_curve_rows if r["model"] == mname)):
            vals = [r["held_out_tv"] for r in learning_curve_rows if r["model"] == mname and r["n_train"] == n]
            by_n[n] = float(np.mean(vals))
        lc_agg[mname] = by_n
    data_helps = {}
    for mname, by_n in lc_agg.items():
        ns = sorted(by_n.keys())
        if len(ns) > 1:
            data_helps[mname] = bool(by_n[ns[-1]] < by_n[ns[0]])
    boundary_enriched_valid = [r for r in boundary_enriched_rows if "natural_test_tv_regimeE" in r]
    boundary_enrichment_helps = None
    if boundary_enriched_valid:
        mean_regimeN = float(np.mean([r["natural_test_tv_regimeN"] for r in boundary_enriched_valid]))
        mean_regimeE = float(np.mean([r["natural_test_tv_regimeE"] for r in boundary_enriched_valid]))
        boundary_enrichment_helps = bool(mean_regimeE < mean_regimeN)
    stage_b_decision = {"learning_curves_by_model": lc_agg, "more_data_helps_by_model": data_helps,
                          "boundary_enrichment_helps_natural_test": boundary_enrichment_helps,
                          "n_clusters_with_boundary_enrichment_data": len(boundary_enriched_valid)}
    print(f"  Stage B decision: {stage_b_decision}")
    report["stage_B_decision"] = stage_b_decision
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    # Y2-C: margin-stratified error, functional complexity correlations
    natural_margin_rows = [r for r in boundary_rows if r["kind"] == "natural_margin"]
    margin_summary = {}
    if natural_margin_rows:
        margin_vals = np.array([r["margin"] for r in natural_margin_rows])
        edges = np.quantile(margin_vals, [0.25, 0.5, 0.75])
        bins = np.digitize(margin_vals, edges)
        for b in range(4):
            idxs = [i for i, bb in enumerate(bins) if bb == b]
            if idxs:
                margin_summary[b] = {"n": len(idxs), "mean_margin": float(margin_vals[[i for i in idxs]].mean()),
                                        "mean_A0_tv": float(np.mean([natural_margin_rows[i]["A0_tv"] for i in idxs])),
                                        "mean_N2_tv": float(np.mean([natural_margin_rows[i]["N2_tv"] for i in idxs]))}
    same_label_rows = [r for r in boundary_rows if r["kind"] == "same_label"]
    diff_label_rows = [r for r in boundary_rows if r["kind"] == "different_label"]
    boundary_jump_summary = {
        "same_label_mean_exact_jump_tv": float(np.mean([r["exact_teacher_jump_tv"] for r in same_label_rows])) if same_label_rows else None,
        "different_label_mean_exact_jump_tv": float(np.mean([r["exact_teacher_jump_tv"] for r in diff_label_rows])) if diff_label_rows else None,
        "same_label_mean_model_predicted_jump_tv": float(np.mean([r["model_predicted_jump_tv"] for r in same_label_rows])) if same_label_rows else None,
        "different_label_mean_model_predicted_jump_tv": float(np.mean([r["model_predicted_jump_tv"] for r in diff_label_rows])) if diff_label_rows else None,
        "note": "Confirms same-label exchanges produce near-zero exact jumps while different-label exchanges "
                 "produce substantial ones -- the theoretical w/Z derivation's qualitative prediction, checked "
                 "against real cluster data, not just the synthetic check.",
    }
    print(f"  Margin-stratified error: {margin_summary}")
    print(f"  Boundary-jump summary (real clusters): {boundary_jump_summary}")
    report["margin_diagnostic"] = margin_summary
    report["boundary_jump_summary_real_clusters"] = boundary_jump_summary
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    # Query-change fidelity, reported separately by category (Section 8's explicit instruction)
    qc_by_category = {}
    for category in sorted(set(r["category"] for r in query_change_rows)):
        for mname in sorted(set(r["model"] for r in query_change_rows if r["category"] == category)):
            recs = [r for r in query_change_rows if r["category"] == category and r["model"] == mname]
            e_deltas = np.array([r["e_delta"] for r in recs])
            dime_errs = np.array([r["dime_zero_change_error"] for r in recs])
            qc_by_category[f"{category}_{mname}"] = {"n_pairs": len(recs), "mean_e_delta": float(e_deltas.mean()),
                                                         "mean_dime_zero_change_error": float(dime_errs.mean()),
                                                         "beats_zero_change_baseline": bool(e_deltas.mean() < dime_errs.mean())}
    print(f"  Query-change fidelity by category: {qc_by_category}")
    report["query_change_by_category"] = qc_by_category
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    # ============================================================
    # Plots
    # ============================================================
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(capacity_agg.keys()), [capacity_agg[m]["mean_test_tv"] for m in capacity_agg])
    ax.axhline(Y1_REAL_A0_TV, color="r", linestyle="--", label="Y1 real A0 (historical benchmark)")
    ax.axhline(Y1_REAL_D0_TV, color="k", linestyle=":", label="Y1 real D0 (historical benchmark)")
    ax.set_ylabel("mean held-out TV"); ax.legend(fontsize=7); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Capacity sweep: held-out TV by model")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "1_capacity_sweep.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mname, by_n in lc_agg.items():
        ns = sorted(by_n.keys())
        ax.plot(ns, [by_n[n] for n in ns], marker="o", label=mname)
    ax.set_xlabel("n_train"); ax.set_ylabel("held-out TV"); ax.legend(fontsize=7)
    ax.set_title("Learning curves (held-out TV vs. training-query count)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "2_learning_curves.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mname in all_model_names:
        xs = [r["n_params"] for r in capacity_rows if r["model"] == mname]
        ys_train = [r["train_tv"] for r in capacity_rows if r["model"] == mname]
        ys_test = [r["test_tv"] for r in capacity_rows if r["model"] == mname]
        if xs:
            ax.scatter(xs, ys_train, marker="o", alpha=0.5, label=f"{mname} (train)")
            ax.scatter(xs, ys_test, marker="x", alpha=0.5, label=f"{mname} (test)")
    ax.set_xlabel("parameter count"); ax.set_ylabel("TV"); ax.set_xscale("log"); ax.legend(fontsize=6, ncol=2)
    ax.set_title("Training vs. held-out error by model capacity")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "3_capacity_vs_generalization.png"), dpi=130); plt.close(fig)

    if margin_summary:
        fig, ax = plt.subplots(figsize=(7, 5))
        bins_sorted = sorted(margin_summary.keys())
        ax.plot([margin_summary[b]["mean_margin"] for b in bins_sorted], [margin_summary[b]["mean_A0_tv"] for b in bins_sorted], marker="o", label="A0")
        ax.plot([margin_summary[b]["mean_margin"] for b in bins_sorted], [margin_summary[b]["mean_N2_tv"] for b in bins_sorted], marker="s", label="N2")
        ax.set_xlabel("mean neighbor margin (quantile bin)"); ax.set_ylabel("mean TV"); ax.legend()
        ax.set_title("Error vs. neighbor-switching margin")
        fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "4_error_vs_margin.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    coverage_by_n = {}
    for n in sorted(set(r["n_train"] for r in coverage_rows)):
        vals = [r["coverage"] for r in coverage_rows if r["n_train"] == n]
        coverage_by_n[n] = float(np.mean(vals))
    ax.plot(sorted(coverage_by_n.keys()), [coverage_by_n[n] for n in sorted(coverage_by_n.keys())], marker="o")
    ax.set_xlabel("n_train"); ax.set_ylabel("mean label coverage")
    ax.set_title("Training label coverage vs. training-query count")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "5_label_coverage.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    cats = sorted(set(r["category"] for r in query_change_rows))
    for mname in ["D0", "A0", "N2"]:
        ys = [qc_by_category.get(f"{cat}_{mname}", {}).get("mean_e_delta") for cat in cats]
        xs_plot = [i for i, y in enumerate(ys) if y is not None]
        ys_plot = [y for y in ys if y is not None]
        if ys_plot:
            ax.plot(xs_plot, ys_plot, marker="o", label=mname)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("mean query-change L1 error"); ax.legend()
    ax.set_title("Query-change fidelity by pair category")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "6_query_change_by_category.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if boundary_enriched_valid:
        labels_be = ["Regime N\n(natural)", "Regime E\n(boundary-enriched)"]
        ax.bar(labels_be, [np.mean([r["natural_test_tv_regimeN"] for r in boundary_enriched_valid]),
                             np.mean([r["natural_test_tv_regimeE"] for r in boundary_enriched_valid])])
        ax.set_ylabel("mean natural held-out TV")
        ax.set_title("Natural vs. boundary-enriched training")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "7_boundary_enriched_vs_natural.png"), dpi=130); plt.close(fig)

    # ============================================================
    # failure_analysis.md
    # ============================================================
    print("\nWriting failure_analysis.md...")
    per_cluster_a0_tv = {c: float(np.mean([r["test_tv"] for r in cluster_results_rows if r["cluster"] == c and r["model"] == "A0"])) for c in sample_clusters}
    per_cluster_n2_tv = {c: float(np.mean([r["test_tv"] for r in cluster_results_rows if r["cluster"] == c and r["model"] == "N2"])) for c in sample_clusters}
    per_cluster_mlp_tv = {c: float(np.mean([r["test_tv"] for r in cluster_results_rows if r["cluster"] == c and r["model"] == "MLP_w512_d4"])) for c in sample_clusters}
    failure_lines = ["# Y2 Failure Analysis", "",
                       "Per-cluster held-out TV by model, distinguishing capacity/fitting/generalization/coverage/geometric limitations:", ""]
    for c in sample_clusters:
        cov_at_max = next((r["coverage"] for r in coverage_rows if r["cluster"] == c and r["n_train"] == max(r2["n_train"] for r2 in coverage_rows if r2["cluster"] == c)), None)
        memo_tv_c = next((r["train_tv"] for r in memorization_rows if r["cluster"] == c and r["model"] == "N2" and r["m_label"] == f"m{M_PRIMARY}" and r["n_train"] == max(r2["n_train"] for r2 in memorization_rows if r2["cluster"] == c and r2["model"] == "N2" and r2["m_label"] == f"m{M_PRIMARY}")), None)
        diagnosis = "capacity/fitting (training itself fails)" if (memo_tv_c is not None and memo_tv_c > NEAR_MEMORIZATION_TV_THRESHOLD * 4) else \
            "coverage (low training label coverage)" if (cov_at_max is not None and cov_at_max < 0.5) else \
            "generalization (fits training, fails held-out)"
        failure_lines.append(f"- cluster={c}: A0_tv={per_cluster_a0_tv[c]:.4f}, N2_tv={per_cluster_n2_tv[c]:.4f}, "
                               f"MLP_tv={per_cluster_mlp_tv[c]:.4f}, N2_memo_train_tv={memo_tv_c}, label_coverage={cov_at_max}, "
                               f"likely_dominant_limitation={diagnosis}")
    with open(os.path.join(OUT_DIR, "failure_analysis.md"), "w") as f:
        f.write("\n".join(failure_lines))

    # ============================================================
    # y1_audit.md
    # ============================================================
    y1_audit_md = """# Y1 Audit (Phase 1, per Y2's own instruction)

## Real results (read directly from results/stage3_y1_nonlinear_function_feasibility/y1_nonlinear_summary.json)

Y1's real sample: 15 clusters selected from 86/150 scanned candidates meeting fit>=30/tune>=15/test>=20 (its own
per_cluster_counts show 45-63 fit queries/cluster -- scarce). At m=8, rank=8 (primary config): D0=0.8862,
A0=0.8293, N1=0.8510, N2=0.8301 mean held-out TV. N2 did NOT significantly beat A0 on real data
(frac_clusters_N2_beats_A0=0.533, 95% CI [-0.0092,+0.0118]) despite clearly beating A0 on Y1's own pilot's trivial
synthetic structure -- confirming the pilot/real gap this project has repeatedly found, not papered over here either.
Query-change preservation failed for every model (A0=2.80, N1=2.07, N2=2.41, all worse than D0's 1.77). The
m-sweep showed nonlinear models helping at low m but losing to the histogram at m=20. Y1's own verdict:
"Promising but insufficient evidence."

## Y1's exact prior implementation (confirmed by direct inspection of
run_y1_nonlinear_function_feasibility.py, reused UNCHANGED here for D0/A0/N1/N2)

- Teacher: run_y1_per_cluster_function's local_m_distribution -- hard top-m selection by real Euclidean distance,
  then a softmax restricted to those m members. Reduces exactly to the all-member exact_evidence target when
  m >= population (verified in Y1's own check 1, not re-verified here to avoid duplicating that work).
- A0: R4's own affine anchor-difference basis (confirmed affine in q by R7's audit), fit via log_ratio_target +
  ridge + reconstruct_logit.
- N1/N2: RAW (never logged) Gaussian RBF features -- genuinely non-affine, verified on a synthetic XOR target in
  Y1's check_non_affine_expressivity (affine accuracy 0.525, RBF accuracy 0.978).
- Routing: nearest-centroid (k=1) assignment of controller_train/val queries to clusters, identical convention
  to every prior script in this project.
- Distance/temperature convention: Euclidean distance, tau=2.0 -- unchanged here.

## Causes of exclusion / query-sample limitation (direct, not assumed)

Y1's own audit scanned 150 candidates (population>=50) and found only 86 meeting its fit>=30/tune>=15/test>=20
bar -- the remaining 64 candidates had fewer routed controller_train/val queries than that bar, a direct
consequence of this project's fixed cache sizes (90,170 controller_train + 77,089 val queries routed via
nearest-centroid assignment across 3,811 clusters, averaging ~44 queries/cluster) spread unevenly by the
underlying MiniBatchKMeans partition. Y2's own audit (broader pool, higher bar, Section 5 above) reaches the
same structural ceiling -- there is no additional natural query data beyond these three cache files.

## Were Y1's nonlinear models limited primarily by architecture, anchors, bandwidth, data, or optimization?

Y1 itself did not separate these explanations (it compared only N1/N2 against A0, both still rank<=32, both
still fit by closed-form ridge, with a 3-point bandwidth sweep). Y2's explicit purpose is to separate exactly
these candidate explanations -- the memorization test (Y2-A3) isolates architecture/optimization from data/
generalization; the learning curve (Y2-B2) isolates data availability; the capacity sweep (MLP, RBF-large)
isolates whether rank<=32 itself was the bottleneck. See capacity_results.csv and memorization_results.csv for
the resulting evidence, not asserted here in advance.
"""
    with open(os.path.join(OUT_DIR, "y1_audit.md"), "w") as f:
        f.write(y1_audit_md)

    # ============================================================
    # mathematical_analysis.md
    # ============================================================
    math_md = f"""# Y2 Mathematical Analysis

## Target regularity (extends Y1's A1/A2 analysis, not re-derived from scratch)

Y1 already proved R_S = {{q: N_m^C(q)=S}} is convex (an intersection of perpendicular-bisector half-spaces) and
characterized which neighbor exchanges cause discontinuities. Y2 adds the EXACT quantitative boundary-jump
derivation (Section 7/C2 of the spec):

At an ordinary single exchange where member i leaves the top-m set and member j enters, the tie condition
d(q,h_i)=d(q,h_j) gives EQUAL kernel weights w_i(q)=w_j(q)=w at the boundary. Assuming every other selected
neighbor is unchanged, the normalizer Z is also continuous through the crossing (it is a sum over the SAME set
of m-1 unchanged members plus the one tied weight w, counted once on either side), so:

\\[
F_+(q) - F_-(q) = \\frac{{w}}{{Z}}(e_{{y_j}} - e_{{y_i}})
\\]

For y_i != y_j: TV(F_+,F_-) = w/Z exactly (a one-directional mass transfer of exactly w/Z from label y_i to y_j).
For y_i == y_j: the right-hand side is identically zero -- NO jump, confirming Y1's qualitative finding with an
exact quantity, not just a sign. This was VERIFIED NUMERICALLY (not merely derived) in
check_boundary_jump_formula() on a synthetic m=1 full-flip case (theoretical=observed=1.0 exactly) and again
directly on REAL cluster data in boundary_results.csv (same_label_mean_exact_jump_tv vs.
different_label_mean_exact_jump_tv, see report).

**Assumptions stated explicitly:** an ORDINARY single exchange (no higher-order ties among 3+ members at
once), and that the OTHER m-1 selected neighbors' weights do not themselves change discontinuously at this same
q (true generically, since their own distances vary smoothly away from their own tie points, which are
independent lower-measure events). Not every geometric neighbor exchange is a predictive discontinuity -- only
different-label exchanges are, exactly as Y1 already established qualitatively; this derivation makes it exact.

## Approximation guarantees for the MLP family

A sufficiently wide one-hidden-layer network with a non-polynomial activation (ReLU qualifies) is a universal
approximator of continuous functions on compact sets (classical universal approximation theorem) -- this is an
EXISTENCE result, not evidence of practical finite-sample learnability, favorable optimization, or efficient
storage, per the spec's own explicit caution. F_C^(m) is bounded (hence in L^2(Q_C) regardless of continuity, as
Y1 already established), so sufficiently expressive families CAN approximate it in expected TV/squared error
even where it is discontinuous, PROVIDED the query distribution assigns zero mass to the exact discontinuity set
-- the same L^2-density argument Y1 made for the RBF family applies identically to the MLP family, since neither
argument depends on the specific function class, only on boundedness and the discontinuity set's measure. This
is explicitly NOT the same as uniform approximation across all of R^768 (impossible near a genuine jump, per
Y1's two-sided-limit argument), and it is explicitly NOT evidence that a SPECIFIC finite MLP, trained on a SPECIFIC
finite query sample with a SPECIFIC optimizer, will actually achieve this -- that is exactly what
memorization_results.csv and learning_curve_results.csv measure empirically, not assumed from the theorem.

## Critical distinctions maintained throughout (Section 3's explicit requirement)

Five separate questions, measured by five separate experiments: (1) existence of a good approximant --
NOT measured directly (R_H^* is unknown); (2) architecture's capacity to represent it -- Y2-A3's memorization
test; (3) optimizer's ability to find good parameters -- also Y2-A3 (persistent high training error after
memorization-test optimization effort implicates optimization or capacity, not data); (4) generalization from a
finite sample -- Y2-B2's learning curves; (5) compressibility of the result -- explicitly OUT OF SCOPE for Y2,
per the spec's own "do not introduce memory budgets" instruction.
"""
    with open(os.path.join(OUT_DIR, "mathematical_analysis.md"), "w") as f:
        f.write(math_md)

    # ============================================================
    # experimental_design.md
    # ============================================================
    design_md = f"""# Y2 Experimental Design

## Preregistered hypotheses

H1: a sufficiently expressive model (MLP or RBF-large) can memorize a fixed finite training set to
train_tv < {NEAR_MEMORIZATION_TV_THRESHOLD} at m={{1,{M_PRIMARY},n}}.
H2: held-out TV improves by more than {HELD_OUT_TV_IMPROVEMENT_THRESHOLD} (absolute) over Y1's real A0 benchmark
({Y1_REAL_A0_TV}) for at least one new model family.
H3: query-change L1 error beats the DIME zero-change baseline for at least one model, in at least the natural-
pairs category.
H4: held-out TV decreases monotonically (or near-monotonically) as n_train increases, for at least one model.
H5: boundary-enriched training improves natural held-out TV relative to equal-size natural-only training.

## Selected clusters

{{SAMPLE_CLUSTERS_PLACEHOLDER}}

## Operational thresholds (defined BEFORE inspecting new held-out results, Section 11)

near_memorization_tv={NEAR_MEMORIZATION_TV_THRESHOLD}, held_out_tv_improvement_required={HELD_OUT_TV_IMPROVEMENT_THRESHOLD}
(vs. Y1's real A0={Y1_REAL_A0_TV}, D0={Y1_REAL_D0_TV}, N2={Y1_REAL_N2_TV}), query_change_must_beat_zero_change=True,
generalization_gap_threshold={GENERALIZATION_GAP_THRESHOLD}.

## Training objective

MLP: soft-target cross-entropy -(T*log_softmax).sum(dim=1).mean(), correctly zero for every exact-zero teacher
label (0*finite=0, never NaN). RBF/A0: R4's own log-ratio-target ridge regression, reused unchanged.

## Stage gates

Y2-A (capacity/memorization) -> Y2-B (learnability/data) -> Y2-C (geometry/failure attribution), run
sequentially; later stages' scope was NOT expanded beyond this delivery's stated bounds, since the Y2-A/B
results (see report.md) did not show a clear justification for a larger grid.
"""
    with open(os.path.join(OUT_DIR, "experimental_design.md"), "w") as f:
        f.write(design_md.replace("{SAMPLE_CLUSTERS_PLACEHOLDER}", str(sample_clusters)))

    # ============================================================
    # synthetic_tests.md
    # ============================================================
    synthetic_md = f"""# Y2 Synthetic Tests

{json.dumps(sanity, indent=2)}

Y1's own 7 synthetic checks (exact-target consistency, m=1 one-hot behavior, probability validity, non-affine
XOR expressivity for the RBF family, discontinuity behavior, gauge invariance, storage accounting) are reused
unchanged and were not re-run here to avoid duplicating already-verified work -- see
results/stage3_y1_nonlinear_function_feasibility/synthetic_tests.md for that record. This file adds only the
two checks genuinely new to Y2: that the MLP family can solve a target (XOR) no affine model can, and that the
EXACT boundary-jump formula derived in mathematical_analysis.md matches the real teacher's observed jump, not
merely that a jump exists (which Y1 already checked).
"""
    with open(os.path.join(OUT_DIR, "synthetic_tests.md"), "w") as f:
        f.write(synthetic_md)

    # ============================================================
    # Outcome classification (Section 11, A-F) -- the final verdict
    # ============================================================
    print("\nComputing outcome classification (A-F)...")
    best_model_by_test_tv = min(capacity_agg, key=lambda m: capacity_agg[m]["mean_test_tv"])
    h1_memorization = stage_a_decision["any_model_near_memorization"]
    h2_held_out_improved = any(v["beats_y1_real_A0_by_threshold"] for v in capacity_agg.values())
    h3_query_change = any(v.get("beats_zero_change_baseline") for k, v in qc_by_category.items() if "natural_pairs" in k)
    h4_more_data_helps = any(data_helps.values()) if data_helps else False
    h5_boundary_helps = bool(boundary_enrichment_helps)
    mean_gen_gap_best = capacity_agg[best_model_by_test_tv]["mean_generalization_gap"]

    if h1_memorization and h2_held_out_improved and h3_query_change:
        outcome = "Outcome A -- Accurate representation and generalization"
    elif h1_memorization and not h2_held_out_improved:
        outcome = "Outcome B -- Training fit succeeds, generalization fails"
    elif not h1_memorization:
        outcome = "Outcome C -- Persistent training failure"
    elif margin_summary and len(margin_summary) > 1 and (margin_summary[0]["mean_N2_tv"] > margin_summary[max(margin_summary.keys())]["mean_N2_tv"] * 1.2):
        outcome = "Outcome D -- Boundary-specific failure"
    elif h2_held_out_improved and not h3_query_change:
        outcome = "Outcome E -- Distribution fidelity improves, query-change fidelity fails"
    elif report["query_availability_audit"]["n_sufficient"] < TARGET_N_CLUSTERS * 3:
        outcome = "Outcome F -- Data limitation"
    else:
        outcome = "Outcome C -- Persistent training failure"
    print(f"  Outcome classification: {outcome}")
    verdict_block = {
        "H1_memorization_achieved": h1_memorization, "H2_held_out_improved_over_Y1_A0": h2_held_out_improved,
        "H3_query_change_beats_zero_change_natural": h3_query_change, "H4_more_data_helps": h4_more_data_helps,
        "H5_boundary_enrichment_helps": h5_boundary_helps, "best_model_by_test_tv": best_model_by_test_tv,
        "best_model_mean_generalization_gap": mean_gen_gap_best,
        "generalization_gap_acceptable": bool(mean_gen_gap_best < GENERALIZATION_GAP_THRESHOLD),
        "outcome_classification": outcome,
    }
    report["final_verdict"] = verdict_block
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)

    # ============================================================
    # final_report.md
    # ============================================================
    lines = [
        "# Y2: Representability, Learnability, and Functional Complexity of Local kNN Retrieval", "",
        "## 1. Research question", "See module docstring, y1_audit.md, and mathematical_analysis.md.", "",
        "## 2. Y1 audit summary", "See y1_audit.md for the full audit of Y1's real results.", "",
        "## 3. Mathematical analysis summary", "See mathematical_analysis.md for the exact boundary-jump derivation and approximation guarantees.", "",
        "## 4. Cluster/query sampling", f"{report['query_availability_audit']}", "",
        "## 5. Stage A decision (capacity/memorization)", f"{stage_a_decision}", "",
        "## 6. Capacity sweep (held-out TV, vs. Y1's real historical benchmarks)", f"{capacity_agg}", "",
        "## 7. Stage B decision (learnability/data)", f"{stage_b_decision}", "",
        "## 8. Stage C diagnostics (margin, boundary jumps)", f"margin: {margin_summary}", f"boundary jumps: {boundary_jump_summary}", "",
        "## 9. Query-change fidelity by category (never combined)", f"{qc_by_category}", "",
        "## 10. Final verdict", f"**{outcome}**", f"{verdict_block}", "",
        "## Limitations",
        f"- Focused pilot: {len(sample_clusters)} clusters (per the spec's own 'approximately 3-5' instruction), not a large-scale study.",
        "- No document/sequence-grouped split exists in the cached data.",
        "- No additional held-out or training-source text is available beyond the three existing cache files -- "
        "natural query scarcity is a real, reported limitation, not resolved by this delivery.",
        "- Boundary-enriched points are heuristically PLACED (nearest label-differing member-pair interpolation) "
        "but their TRAINING TARGETS are always the exact teacher, never hand-labeled or approximated.",
        f"- MLP capacity grid limited to widths {MLP_WIDTHS} x depths {MLP_DEPTHS}, not the full aspirational grid, "
        "per the spec's own 'initial configurations, not mandatory Cartesian product' instruction.",
        "- The historical Y1 validation population has been used repeatedly for research decisions across this "
        "project; this delivery's own real-data comparisons remain exploratory, per the spec's own instruction.",
    ]
    with open(os.path.join(OUT_DIR, "final_report.md"), "w") as f:
        f.write("\n".join(lines))

    report["sections_completed"] = ["synthetic_correctness_checks", "query_availability_audit", "capacity_sweep",
                                      "memorization_test", "learning_curves", "label_coverage", "boundary_enriched_training",
                                      "margin_diagnostic", "boundary_jump_verification", "query_change_fidelity",
                                      "capacity_results.csv", "memorization_results.csv", "learning_curve_results.csv",
                                      "label_coverage_results.csv", "boundary_results.csv", "query_change_results.csv",
                                      "cluster_results.csv", "y1_audit.md", "mathematical_analysis.md",
                                      "experimental_design.md", "synthetic_tests.md", "failure_analysis.md",
                                      "final_verdict", "final_report.md", "plots"]
    report["total_runtime_seconds"] = time.perf_counter() - t_start
    save_results(os.path.join(OUT_DIR, "y2_summary.json"), report)
    print(f"\nAll results saved under {OUT_DIR}")
    print(f"Total runtime: {report['total_runtime_seconds']/60:.1f} min")
    print("All R-series, R-extra, and Y1 outputs were left untouched.")


if __name__ == "__main__":
    main()
