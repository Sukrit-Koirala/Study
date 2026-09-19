import random
import re

def assign_story_split(rand_val, frac_datastore=0.70, frac_controller=0.15):
    """Map one random draw in [0,1) to a split name using fixed cutoffs."""
    if rand_val < frac_datastore:
        return "datastore"
    elif rand_val < frac_datastore + frac_controller:
        return "controller_train"
    else:
        return "val"


_ARTICLE_TITLE = re.compile(r"^ = [^=].* = \n?$")


def is_wikitext_article_title(text):
    """WikiText raw rows are lines; an article starts at a level-1 heading like ' = Title = '
    (section headings are ' = = Section = = ')."""
    return bool(_ARTICLE_TITLE.match(text))


def collect_chunks_split(dataset, tokenizer, seq_len, n_chunks_needed,frac_datastore=0.7, frac_controller=0.15, seed = 42,
                         article_split=False):
    """article_split=False (default, used by every result so far): one random draw per dataset ROW.
    article_split=True (WikiText only): one draw per ARTICLE, so all paragraphs of an article land in
    the same split. Default behaviour and rng call order are unchanged."""
    rng = random.Random(seed)

    buffers = {"datastore": [], "controller_train": [], "val": []}
    chunks  = {"datastore": [], "controller_train": [], "val": []}
    chunk_story_ids = {"datastore": [], "controller_train": [], "val": []}
    current_split = None

    for story_id, story in enumerate(dataset):
        if article_split:
            if current_split is None or is_wikitext_article_title(story["text"]):
                current_split = assign_story_split(rng.random(), frac_datastore, frac_controller)
            split = current_split
        else:
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






    





