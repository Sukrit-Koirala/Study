import torch
import numpy as np
from collections import Counter
from helpers import predictive_entropy
from q_read import retrieval_purity_entropy

# GPT entropy: one very peaked distribution, one very flat one
logits = torch.tensor([[[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])  # [1, 2, 3] — 2 positions, 3-token vocab
ent = predictive_entropy(logits)
print("entropy (peaked, then flat):", ent)
# peaked should be near 0; flat (uniform over 3) should be near log(3) ≈ 1.0986

# Retrieval purity/entropy: one dominated cluster, one perfectly even split
retrieved_top1 = np.array([Counter({5: 9, 2: 1}), Counter({5: 5, 2: 5})], dtype=object)
entropy, purity = retrieval_purity_entropy(retrieved_top1)
print("retrieval entropy:", entropy, " purity:", purity)
# first: purity should be 0.9 (9/10), low entropy. second: purity 0.5, entropy at its max for 2 outcomes (ln(2)≈0.693)
