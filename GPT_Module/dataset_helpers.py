import random

def assign_story_split(rand_val, frac_datastore=0.70, frac_controller=0.15):
    """Map one random draw in [0,1) to a split name using fixed cutoffs."""
    if rand_val < frac_datastore:
        return "datastore"
    elif rand_val < frac_datastore + frac_controller:
        return "controller_train"
    else:
        return "val"


def collect_chunks_split(dataset, tokenizer, seq_len, n_chunks_needed,frac_datastore=0.7, frac_controller=0.15, seed = 42):
    rng = random.Random(seed)

    buffers = {"datastore": [], "controller_train": [], "val": []}
    chunks  = {"datastore": [], "controller_train": [], "val": []}
    chunk_story_ids = {"datastore": [], "controller_train": [], "val": []}

    for story_id, story in enumerate(dataset):
        r = rng.random()
        split = assign_story_split(r, frac_datastore, frac_controller)

        ids = tokenizer.encode(story["text"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buffers[split].extend(ids)

        while len(buffers[split]) >= seq_len:
            chunks[split].append(buffers[split][:seq_len])
            chunk_story_ids[split].append(story_id)
            buffers[split] = buffers[split][seq_len:]

        if all(len(chunks[s]) >= n_chunks_needed[s] for s in chunks):
            break

    return chunks, chunk_story_ids





    





