import argparse
from gpt_class import FrozenGPT2
from dataset_helpers import collect_chunks_split
from extract import run_batch_with_entropy
from cache_io import save_cache
from datasets import load_dataset
import numpy as np

DATASET_REGISTRY = {
    "tinystories": lambda: load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True),
    "wikitext103": lambda: load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True),
    "wikitext2": lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True),
}


def encode_split_with_entropy(gpt, chunks_list, batch_size=256):
    all_keys, all_targets, all_p_lm, all_entropy = [], [], [], []
    for i in range(0, len(chunks_list), batch_size):
        batch = chunks_list[i:i + batch_size]
        h_pred, y_target, p_true, nll, entropy = run_batch_with_entropy(gpt, batch)
        B, L, D = h_pred.shape
        all_keys.append(h_pred.reshape(B * L, D).cpu().numpy())
        all_targets.append(y_target.reshape(B * L).cpu().numpy())
        all_p_lm.append(p_true.reshape(B * L).cpu().numpy())
        all_entropy.append(entropy.reshape(B * L).cpu().numpy())
    return (np.concatenate(all_keys, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_p_lm, axis=0),
            np.concatenate(all_entropy, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--model", required=True, help='e.g. "gpt2", "gpt2-medium"')
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_datastore", type=int, required=True)
    parser.add_argument("--n_controller_train", type=int, required=True)
    parser.add_argument("--n_val", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    dataset = DATASET_REGISTRY[args.dataset]()
    gpt = FrozenGPT2(model_name=args.model)

    chunks, chunk_story_ids = collect_chunks_split(
        dataset, gpt.tokenizer, seq_len=args.seq_len,
        n_chunks_needed={
            "datastore": args.n_datastore,
            "controller_train": args.n_controller_train,
            "val": args.n_val,
        }
    )

    for split_name in ["datastore", "controller_train", "val"]:
        keys, values, p_lm_true, entropy = encode_split_with_entropy(gpt, chunks[split_name], batch_size=args.batch_size)
        cache_path = f"{args.out_dir}/{args.dataset}_{args.model}_{split_name}.npz"
        save_cache(cache_path, keys, values, p_lm_true, entropy)
        print(f"cached {split_name}: {len(values)} positions -> {cache_path}")


if __name__ == "__main__":
    main()
