from q_read_dense import build_qsa_dataset
import numpy as np

toy_obs = np.array([[1.0]*11, [2.0]*11])          # N=2, 11 features, easy to eyeball
toy_actions = np.array([[0.1]*4, [0.2]*4, [0.3]*4])  # A=3, 4 features
toy_reward = np.array([[10, 20, 30], [40, 50, 60]])  # [N=2, A=3]

X, y = build_qsa_dataset(toy_obs, toy_actions, toy_reward)
print("X shape:", X.shape, "expect (6, 15)")
print("y:", y, "expect [10 20 30 40 50 60]")
print("row 0 (query0,action0):", X[0], "expect eleven 1.0s then [0.1]*4")
print("row 4 (query1,action1):", X[4], "expect eleven 2.0s then [0.2]*4")
