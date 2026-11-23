# Y1 Failure Cases (family B, rank=8)

Worst = largest (test_tv - dime_tv), i.e. function performs WORSE than the fixed histogram. Best = smallest (most negative), i.e. largest improvement.

## Worst (function underperforms DIME)
- cluster=1465, m=20, population=30, n_tokens=28, sensitivity=low: test_tv=0.925923, dime_tv=0.611655, gap=0.31427
- cluster=795, m=20, population=332, n_tokens=68, sensitivity=low: test_tv=0.658273, dime_tv=0.477906, gap=0.18037
- cluster=1465, m=8, population=30, n_tokens=28, sensitivity=low: test_tv=0.885528, dime_tv=0.740431, gap=0.1451
- cluster=2059, m=20, population=323, n_tokens=165, sensitivity=low: test_tv=0.802554, dime_tv=0.700165, gap=0.10239
- cluster=923, m=20, population=313, n_tokens=184, sensitivity=high: test_tv=0.855173, dime_tv=0.793263, gap=0.06191
- cluster=647, m=20, population=368, n_tokens=229, sensitivity=low: test_tv=0.816268, dime_tv=0.774699, gap=0.04157
- cluster=2161, m=20, population=317, n_tokens=229, sensitivity=high: test_tv=0.885198, dime_tv=0.8453, gap=0.0399
- cluster=795, m=8, population=332, n_tokens=68, sensitivity=low: test_tv=0.661354, dime_tv=0.62531, gap=0.03604
- cluster=569, m=20, population=309, n_tokens=245, sensitivity=moderate: test_tv=0.90453, dime_tv=0.870496, gap=0.03403
- cluster=2059, m=8, population=323, n_tokens=165, sensitivity=low: test_tv=0.817892, dime_tv=0.793831, gap=0.02406

## Best (function substantially improves over DIME)
- cluster=795, m=1, population=332, n_tokens=68, sensitivity=low: test_tv=0.707275, dime_tv=0.900068, gap=-0.19279
- cluster=2059, m=1, population=323, n_tokens=165, sensitivity=low: test_tv=0.811984, dime_tv=0.950756, gap=-0.13877
- cluster=647, m=1, population=368, n_tokens=229, sensitivity=low: test_tv=0.830833, dime_tv=0.968415, gap=-0.13758
- cluster=965, m=4, population=320, n_tokens=218, sensitivity=moderate: test_tv=0.812601, dime_tv=0.943411, gap=-0.13081
- cluster=965, m=1, population=320, n_tokens=218, sensitivity=moderate: test_tv=0.853685, dime_tv=0.984327, gap=-0.13064
- cluster=965, m=8, population=320, n_tokens=218, sensitivity=moderate: test_tv=0.789139, dime_tv=0.907428, gap=-0.11829
- cluster=1465, m=1, population=30, n_tokens=28, sensitivity=low: test_tv=0.856691, dime_tv=0.966667, gap=-0.10998
- cluster=3072, m=8, population=312, n_tokens=284, sensitivity=high: test_tv=0.870348, dime_tv=0.968924, gap=-0.09858
- cluster=2161, m=4, population=317, n_tokens=229, sensitivity=high: test_tv=0.863125, dime_tv=0.958872, gap=-0.09575
- cluster=569, m=4, population=309, n_tokens=245, sensitivity=moderate: test_tv=0.877365, dime_tv=0.967461, gap=-0.0901