"""Synthetic-data sanity tests for the E1 diagnostic pipeline (Section 11 of the
spec) — run before touching WikiText-103. No GPT-2, no cluster, pure numpy/sklearn
on tiny hand-built toy data. Run with: python test_e1_metrics.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "GPT_Module"))

import numpy as np
from collections import Counter

from knn import build_datastore, query_knn, mix_knn_and_lm
from mixing import mix_dime_and_lm
from helpers import shift_for_next_token
from e1_metrics import (
    raw_standalone_prob, dime_standalone_prob, raw_full_distribution,
    dime_full_distribution, js_divergence_dict, retrieval_coverage,
)

N_PASSED = 0


def check(name, cond):
    global N_PASSED
    assert cond, f"FAILED: {name}"
    N_PASSED += 1
    print(f"  PASS: {name}")


def test_distributions_sum_to_one():
    keys = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.array([10, 20, 30])
    index, stored = build_datastore(keys, values)
    q = np.array([[0.1, 0.1]])
    d, retrieved = query_knn(index, stored, q, k=3)
    raw_dist = raw_full_distribution(d, retrieved, tau=1.0)
    check("raw distribution sums to 1", abs(sum(raw_dist[0].values()) - 1.0) < 1e-9)

    centroids = keys.copy()
    counters = np.array([Counter({10: 1}), Counter({20: 1}), Counter({30: 1})], dtype=object)
    dime_index, dime_stored = build_datastore(centroids, counters)
    d2, retrieved2 = query_knn(dime_index, dime_stored, q, k=3)
    dime_dist = dime_full_distribution(d2, retrieved2, tau=1.0)
    check("DIME distribution sums to 1", abs(sum(dime_dist[0].values()) - 1.0) < 1e-9)


def test_singleton_equivalence():
    """Identical raw and compressed singleton representations (one cluster per
    raw point) must give the same retrieval distribution for matching params."""
    keys = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    values = np.array([10, 20, 30])
    raw_index, raw_stored = build_datastore(keys, values)
    counters = np.array([Counter({v: 1}) for v in values], dtype=object)
    dime_index, dime_stored = build_datastore(keys, counters)

    q = np.array([[0.3, 0.1]])
    true_targets = np.array([10])
    p_lm = np.array([0.4])

    d_raw, r_raw = query_knn(raw_index, raw_stored, q, k=3)
    d_dime, r_dime = query_knn(dime_index, dime_stored, q, k=3)
    p_mixed_raw, nll_raw = mix_knn_and_lm(d_raw, r_raw, true_targets, p_lm, tau=1.0, alpha=0.5)
    p_mixed_dime, nll_dime = mix_dime_and_lm(d_dime, r_dime, true_targets, p_lm, tau=1.0, alpha=0.5)
    check("singleton raw == singleton DIME (p_mixed)", abs(p_mixed_raw[0] - p_mixed_dime[0]) < 1e-9)
    check("singleton raw == singleton DIME (nll)", abs(nll_raw[0] - nll_dime[0]) < 1e-9)


def test_equal_count_cluster_is_50_50():
    centroid = np.array([[0.0, 0.0]])
    cluster = np.array([Counter({10: 5, 20: 5})], dtype=object)
    index, stored = build_datastore(centroid, cluster)
    q = np.array([[0.5, 0.5]])
    d, retrieved = query_knn(index, stored, q, k=1)

    p10 = dime_standalone_prob(d, retrieved, np.array([10]), tau=1.0)
    p20 = dime_standalone_prob(d, retrieved, np.array([20]), tau=1.0)
    check("single equal-count cluster -> P(10)=0.5", abs(p10[0] - 0.5) < 1e-9)
    check("single equal-count cluster -> P(20)=0.5", abs(p20[0] - 0.5) < 1e-9)


def test_js_divergence_properties():
    p = {1: 0.5, 2: 0.5}
    check("JS(p,p) == 0", abs(js_divergence_dict(p, p)) < 1e-12)

    q = {3: 0.5, 4: 0.5}  # fully disjoint support from p
    js_disjoint = js_divergence_dict(p, q)
    check("JS(disjoint) <= ln(2)", js_disjoint <= np.log(2) + 1e-9)
    check("JS(disjoint) > 0", js_disjoint > 0)

    r = {1: 0.3, 2: 0.3, 5: 0.4}
    js_partial = js_divergence_dict(p, r)
    check("JS(partial overlap) >= 0", js_partial >= -1e-12)


def test_zero_probability_when_absent():
    keys = np.array([[0.0, 0.0], [1.0, 0.0]])
    values = np.array([10, 20])  # true token 99 never appears
    index, stored = build_datastore(keys, values)
    q = np.array([[0.5, 0.0]])
    d, retrieved = query_knn(index, stored, q, k=2)
    p = raw_standalone_prob(d, retrieved, np.array([99]), tau=1.0)
    check("absent true token -> exact 0.0 standalone prob", p[0] == 0.0)
    check("retrieval_coverage reports 0% for this query", retrieval_coverage(p) == 0.0)


def test_manual_interpolated_nll():
    """p_lm_true=0.5, one retrieved neighbor exactly matching the true token
    (so p_knn_true=1.0), alpha=0.5 -> p_mixed = 0.5*1.0 + 0.5*0.5 = 0.75."""
    keys = np.array([[0.0, 0.0]])
    values = np.array([7])
    index, stored = build_datastore(keys, values)
    q = np.array([[0.0, 0.0]])
    d, retrieved = query_knn(index, stored, q, k=1)
    p_mixed, nll = mix_knn_and_lm(d, retrieved, np.array([7]), np.array([0.5]), tau=1.0, alpha=0.5)
    expected_p = 0.75
    check("manual interpolated p_mixed matches hand calculation", abs(p_mixed[0] - expected_p) < 1e-6)
    check("manual interpolated NLL matches -log(0.75)", abs(nll[0] - (-np.log(expected_p))) < 1e-6)


def test_next_token_alignment():
    hidden = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(np.float32)  # [B=2, L=4, D=3]
    logits = np.arange(2 * 4 * 5).reshape(2, 4, 5).astype(np.float32)  # [B=2, L=4, V=5]
    input_ids = np.array([[100, 101, 102, 103], [200, 201, 202, 203]])
    import torch
    h_pred, lgts_pred, y_target = shift_for_next_token(
        torch.as_tensor(hidden), torch.as_tensor(logits), torch.as_tensor(input_ids)
    )
    check("h_pred drops last position", h_pred.shape[1] == 3 and torch.equal(h_pred[0, 0], torch.as_tensor(hidden[0, 0])))
    check("y_target is input_ids shifted by one", torch.equal(y_target, torch.as_tensor(input_ids[:, 1:])))
    check("h_pred[t] pairs with the token originally at position t+1",
          y_target[0, 0].item() == 101 and torch.equal(h_pred[0, 0], torch.as_tensor(hidden[0, 0])))


def test_paris_berlin_toy():
    """Two raw points (Paris-labeled, Berlin-labeled) merged into ONE DIME cluster.
    Raw retrieval probabilities must vary with query position; the one-cluster DIME
    distribution must NOT."""
    PARIS, BERLIN = 1, 2
    paris_point = np.array([0.0, 0.0])
    berlin_point = np.array([10.0, 0.0])
    keys = np.array([paris_point, berlin_point])
    values = np.array([PARIS, BERLIN])
    raw_index, raw_stored = build_datastore(keys, values)

    centroid = np.array([[5.0, 0.0]])  # merged cluster location (irrelevant to the point being tested)
    merged_cluster = np.array([Counter({PARIS: 1, BERLIN: 1})], dtype=object)
    dime_index, dime_stored = build_datastore(centroid, merged_cluster)

    q_near_paris = np.array([[0.5, 0.0]])
    q_near_berlin = np.array([[9.5, 0.0]])

    d1, r1 = query_knn(raw_index, raw_stored, q_near_paris, k=2)
    d2, r2 = query_knn(raw_index, raw_stored, q_near_berlin, k=2)
    p_paris_near_paris = raw_standalone_prob(d1, r1, np.array([PARIS]), tau=1.0)[0]
    p_paris_near_berlin = raw_standalone_prob(d2, r2, np.array([PARIS]), tau=1.0)[0]
    check("raw retrieval P(Paris) shifts with query position",
          abs(p_paris_near_paris - p_paris_near_berlin) > 0.1)

    dd1, dr1 = query_knn(dime_index, dime_stored, q_near_paris, k=1)
    dd2, dr2 = query_knn(dime_index, dime_stored, q_near_berlin, k=1)
    dime_p_paris_near_paris = dime_standalone_prob(dd1, dr1, np.array([PARIS]), tau=1.0)[0]
    dime_p_paris_near_berlin = dime_standalone_prob(dd2, dr2, np.array([PARIS]), tau=1.0)[0]
    check("DIME's one-cluster P(Paris) does NOT shift with query position",
          abs(dime_p_paris_near_paris - dime_p_paris_near_berlin) < 1e-9)
    check("DIME's one-cluster distribution is exactly 50/50",
          abs(dime_p_paris_near_paris - 0.5) < 1e-9)


if __name__ == "__main__":
    test_distributions_sum_to_one()
    test_singleton_equivalence()
    test_equal_count_cluster_is_50_50()
    test_js_divergence_properties()
    test_zero_probability_when_absent()
    test_manual_interpolated_nll()
    test_next_token_alignment()
    test_paris_berlin_toy()
    print(f"\nAll {N_PASSED} checks passed.")
