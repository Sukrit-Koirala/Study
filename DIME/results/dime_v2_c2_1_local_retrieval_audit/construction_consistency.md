# C2.1 Construction Consistency

## C1 artifact verification

- Loaded from: `/home/sukrit.koirala/ondemand/upload_me/Research/Study/DIME/results/dime_v2_c1_cluster_structure_audit/dime_v1_cluster_artifact.npz`
- Centroids SHA-256 (verified against externally supplied expected hash): `e7adc93629e102ad7dc16fee9b1c0b1a29be599c44e1f5e42fedbf514d96957e`
- Assignment SHA-256 (verified against externally supplied expected hash): `9dae0c746a8f89dc6f6a3e96b4e0cead73d00965f474d7c3795bdffe6767f13e`
- C1's own construction_fingerprint.json REQUIRED and matching (review point 7): True
- Centroid/assignment dtypes validated against the fingerprint's own recorded dtypes: True
- Centroids/assignment loaded read-only (setflags(write=False)): True
- Datastore row count at load time: 381127
- n_clusters: 3811

## Routing semantics reused (not re-derived)

- Centroid routing: `GPT_Module/knn.py`'s `build_datastore`/`query_knn_indices`, the exact mechanism `mixing.py`'s `mix_dime_and_lm` uses for production inference. k_route=20, tau_route=2.0.
- Centroid MIXTURE WEIGHTS (review point 1): `centroid_mixture_weights` applies `softmax_np(-distances/tau_route, axis=1)` to the routing distances -- the actual DIME matched-configuration softmax, not just the raw distances/indices Revision 0 stopped at. Verified to sum to 1 for every query (asserted at computation time and in `check_centroid_mixture_weights_known_values`).
- Local within-cluster candidate selection: m_local=20, tau_local=2.0 (a separate, diagnostic-only choice -- see module docstring for why these are never conflated with k_route/tau_route).

## Tie-breaking convention (review point 7)

Local top-m selection uses `np.argsort(d, kind="stable")`: ties are broken by ascending position within the cluster's member array, which itself preserves ascending original datastore row order (`build_cluster_to_members` uses a stable sort). This is deterministic and reproducible across repeated calls (`check_boundary_tie_handling`). Boundary ties at the m/m+1 split, and whether the tied pair carries different labels, are detected per query and aggregated into `cluster_behavior_statistics.csv`'s `frac_boundary_tie`/`frac_boundary_tie_diff_label` columns.

## Recomputed routing traffic vs. C1's recorded traffic (review point 7)

max |diff| vs C1's recorded `traffic_top20_val`: 9.97e-17; vs `traffic_top1_val`: 9.89e-17 (tolerance 1e-06) -- PASSED.

## Source references

- `DIME/state_object.py` lines 39-56 (`minibatch_kmeans_partition`) -- authoritative construction, NOT called here; C1's persisted artifact is loaded instead.
- `DIME/run_e1_wikitext103.py` line 58 (`MATCHED_CONFIG = {'k': 20, 'tau': 2.0, 'alpha': 0.25}`) -- source of k_route/tau_route.
- `DIME/mixing.py` (`mix_dime_and_lm`) -- confirms the static per-cluster histogram this diagnostic compares against is `count(y)/n_C`, no smoothing (beta=0 in the matched config), and the `weights = softmax_np(-distances/tau)` formula `centroid_mixture_weights` reuses.
- `GPT_Module/cache_io.py` -- confirms the cache format has no row-id field; query identifiers used throughout this diagnostic are POSITIONAL row indices into the controller_train/val cache arrays, structural identifiers only, not independently verifiable semantic provenance.
