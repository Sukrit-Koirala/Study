# E2.1 — auto-generated numeric summary

Mechanical outcome flag: **A** (pre-registered heuristic; not the conclusion).

Baselines (val, matched k=20 τ=2.0 α=0.25): S1 raw 3.9387 · S2 DIME 4.0209 · S3 3.9282

## Required comparison (exact E2 retention sets)

| Budget | β=0.25 NLL | Controller-selected β | Selected-β NLL | ΔNLL vs β=0.25 [95% chunk CI] | S3 recovery (0.25 → selected) | Val-optimal β (exploratory) |
|---|---:|---:|---:|---|---|---:|
| 1% | 4.0177 | 1 | 4.0142 | -0.00344 [-0.00395, -0.00289] | 3.44% → 7.15% | 0.988 (7.18%) |
| 5% | 4.0108 | 1 | 4.0007 | -0.01008 [-0.01109, -0.00899] | 10.91% → 21.79% | 0.983 (21.91%) |
| 10% | 4.0060 | 1 | 3.9912 | -0.01478 [-0.01621, -0.01336] | 16.02% → 31.98% | 0.974 (32.24%) |

## Split-independent selection (reconstructed sets; ranking half R, β tuned on half T)

| Budget | Selected β | Continuous opt. | S3 recovery (0.25 → selected) | Member-ID overlap with E2 set |
|---|---:|---:|---|---:|
| 1% | 1 | 0.983 | 2.69% → 5.50% | 0.624 |
| 5% | 1 | 0.952 | 9.83% → 19.50% | 0.714 |
| 10% | 1 | 0.939 | 15.20% → 30.15% | 0.756 |

## Worst-10% (retrospective) at β = 0.25 / selected / 1.0

| Budget | β | Repaired | Improved vs DIME | Harmed vs DIME | New severe in rest-90% |
|---|---:|---:|---:|---:|---:|
| 1% | 0.25 | 0.0180 | 0.168 | 0.071 | 0 |
| 1% | 1 | 0.0393 | 0.168 | 0.071 | 10 |
| 5% | 0.25 | 0.0553 | 0.373 | 0.177 | 7 |
| 5% | 1 | 0.1170 | 0.373 | 0.177 | 39 |
| 10% | 0.25 | 0.0786 | 0.521 | 0.217 | 10 |
| 10% | 1 | 0.1679 | 0.521 | 0.217 | 57 |

S3 reference on worst-10%: repaired 0.4723.

Limitations: PRIMARY beta selection uses controller_train, which E2 also used to build the sensitivity ranking that chose the retained members -- tuning and ranking share data; not independent validation. SECONDARY split analysis rebuilds retention from controller_train half R and tunes beta on half T; its retention sets are reconstructed and differ from E2's, so it is reported separately. Val-optimal beta is exploratory/oracle only. Worst-query groups are defined with held-out val labels (retrospective); never used to tune beta. Bootstrap resamples 127-position chunks; article identity is not persisted, so cross-chunk dependence within an article is not modeled.
