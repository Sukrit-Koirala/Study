def build_action_grid(fast = False):
    """ Menu of read-policy configs, plus one gpt_only action
    Each entry: {"name", "k", "tau", "alpha", "beta"}."""

    actions = [{"name": "gpt_only", "k": 0, "tau": 1.0, "alpha": 1.0, "beta": 0.0}]

    k_values = [50,200] if fast else [50,100,200,300]
    tau_values = [1.0,2.0] if fast else [0.5,1.0,2.0,5.0]
    alpha_values = [0.1,0.25] if fast else [0.05,0.1,0.25,0.5]
    beta_values = [0.0, 5.0] if fast else [0.0,1.0,5.0,20.0]

    for k in k_values:
        for tau in tau_values:
            for alpha in alpha_values:
                for beta in beta_values:
                    actions.append({
                        "name": f"k{k}_t{tau}_a{alpha}_b{beta}",
                        "k": k, "tau": tau, "alpha": alpha, "beta": beta,
                    })
    return actions


def build_action_features(actions, k_max=300, beta_max=20.0):
    """[A, 4] — k, tau, alpha, beta, each roughly normalized to a similar scale. """
    import numpy as np
    rows = []
    for a in actions:
        rows.append([a["k"] / k_max, a["tau"], a["alpha"], a["beta"] / beta_max])
    return np.array(rows, dtype=np.float32)

