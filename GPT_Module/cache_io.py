import numpy as np
from pathlib import Path


def save_cache(path, keys, values, p_lm_true, entropy):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=keys, values=values, p_lm_true=p_lm_true, entropy=entropy)


def load_cache(path):
    data = np.load(path)
    return {
        "keys": data["keys"],
        "values": data["values"],
        "p_lm_true": data["p_lm_true"],
        "entropy": data["entropy"],
    }
