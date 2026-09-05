import json
from pathlib import Path


def save_results(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path):
    with open(path) as f:
        return json.load(f)
