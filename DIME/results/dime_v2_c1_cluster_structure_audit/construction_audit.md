# Construction Audit

Dataset: WikiText-103/gpt2. Datastore size: 381127. n_clusters: 3811 (= max(100, round(N/100)), verified against tier1.json: 3811). Seed: 42.

Construction call (verbatim, every Stage-3 script including this one): `MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED)` -- every other parameter left at the installed sklearn version's default (never set explicitly anywhere in this repository). Fit on `ds_keys.astype(np.float32)`, the full datastore, no subset, no normalization.

## This run's actual environment and fingerprint

```
{
  "n_clusters": 3811,
  "seed": 42,
  "centroids_sha256": "e7adc93629e102ad7dc16fee9b1c0b1a29be599c44e1f5e42fedbf514d96957e",
  "assignment_sha256": "9dae0c746a8f89dc6f6a3e96b4e0cead73d00965f474d7c3795bdffe6767f13e",
  "centroids_shape": [
    3811,
    768
  ],
  "assignment_shape": [
    381127
  ],
  "centroids_dtype": "float32",
  "assignment_dtype": "int32",
  "ds_keys_on_disk_dtype": "float32",
  "environment": {
    "python_version": "3.13.12",
    "numpy_version": "2.4.6",
    "sklearn_version": "1.9.0",
    "scipy_version": "1.18.0",
    "pandas_version": "3.0.2",
    "platform": "Linux-6.1.0-52-amd64-x86_64-with-glibc2.36"
  },
  "saved_at_unix_time": 1795815054.841268
}
```

## Reproducibility limitation

No sklearn version is recorded for the artifacts this project cites as historical reference (`wikitext103_gpt2_tier1.json`, `run_manifest.json`) -- a fixed seed alone does not prove this run's partition is bitwise identical to any earlier one. This run's own centroids/assignment/environment are now saved (`dime_v1_cluster_artifact.npz`, `construction_fingerprint.json`) so FUTURE runs can compare directly instead of assuming.
