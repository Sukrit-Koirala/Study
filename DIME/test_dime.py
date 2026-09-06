import numpy as np
from q_read import train_q_read_controller

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 2)).astype(np.float32)
true_reward = 2.0 * X[:, 0] - 1.0 * X[:, 1]   # known linear relationship
y = true_reward + rng.normal(scale=0.1, size=500)  # small noise

model = train_q_read_controller(X, y)

X_test = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
preds = model.predict(X_test)
print("predictions:", preds)
print("true reward: ", 2.0 * X_test[:, 0] - 1.0 * X_test[:, 1])
