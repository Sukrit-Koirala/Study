# Audit: Representation-Ensembling Experiment (2026-10-14 run)

Triggered by: results don't match Xu et al. (2023)'s reported pattern, specifically
the FFN-vs-ATT head ordering and the FFN head's unusually large gain. This audit
does not modify DIME or retrain anything — every check below is either a static
code re-read, a reuse of the already-pasted run log, or a small additive,
non-invasive local diagnostic (tokenizer-only vocabulary scan, a hand-computed unit
test of the head's log-prob math). Nothing in `run_representation_ensembling_experiment.py`,
`learned_head.py`, `extract_dual.py`, or `attn_hook.py` was changed.

**Observed run** (TinyStories/gpt2, `datastore=49657, controller_train(VAL)=9652, val(TEST)=14097`):

| Config | Test NLL | Δ vs. A |
|---|---|---|
| A — base LM | 2.7984 | — |
| B — base + FFN head (α=0.5) | 2.5090 | **0.2894** |
| C — base + ATT head (α=0.4) | 2.5675 | 0.2309 |
| D — base + DIME (on h_ATT) | 2.6741 | 0.1243 |
| E — full raw kNN (on h_ATT) | 2.5833 | 0.2151 |
| F — equal-budget raw (on h_ATT) | 2.7575 | 0.0409 |
| G — base + ATT head + DIME | 2.5675 (= C exactly) | — |

Two things flagged as anomalous, matching what prompted this audit: **(a)** Config B's
gain (0.2894) exceeds every retrieval-based config, including full uncompressed raw
kNN (0.2151) — a freshly-trained linear head apparently beats retrieval over the
*entire* datastore; **(b)** B > C (FFN head beats ATT head), the opposite of what
Xu et al.'s framing would predict if ATT really is the "better" representation.

## 1. Data leakage and split integrity

**Exact training data for the heads**: `ds_ffn`/`ds_att` — built by
`build_dual_datastore_from_chunks` (`extract_dual.py:25`) from `chunks["datastore"]`,
49,657 positions from 350 chunks. This is the **same text DIME's own datastore is
built from** — not new data, not a separate corpus.

**Split mapping**: `run_representation_ensembling_experiment.py`'s own docstring
(lines 5-11) documents the interpretive choice explicitly: `datastore`→TRAIN (head
gradient updates), `controller_train`→VALIDATION (head checkpoint selection *and*
every interpolation-weight tuning decision), `val`→TEST (touched once, final numbers
only). This reuses DIME's existing `collect_chunks_split` 3-way split — the *reason*
the logs say "(VALIDATION)"/"(TEST)" next to the original split names is purely this
script's own relabeling for clarity; nothing about the underlying split mechanism
changed.

**Leakage mechanism**: `collect_chunks_split` (`dataset_helpers.py`) assigns an
entire *story* to exactly one split before any chunking happens — this is the same
function, same `seed=42`, that an earlier phase of this project explicitly leakage-audited
(zero cross-split story IDs found). **Caveat, stated plainly**: that audit was run
once, historically, on a different invocation — I have not re-run a fresh leakage
check against *this specific run's* `chunk_story_ids`, because doing so requires
re-running extraction, which the instruction was to avoid. Structurally the same
guarantee applies (same function, same seed, same split fractions), but this is
"trusted by construction," not "independently re-verified for this exact run."

**Interpolation hyperparameters independent of test**: confirmed by direct code
read. `tune_alpha_two_way` and `tune_simplex_three_way`
(`run_representation_ensembling_experiment.py:57-85`) both grid-search *only* over
the `_val` (controller_train) arguments inside their loops, and apply the winning
parameter to the `_test` (val) arrays exactly once, after the loop — test positions
never appear inside either tuning loop. Confirmed structurally sound.

**One real, if minor, methodological wrinkle — not a bug, worth stating**:
`controller_train` is used for *two* separate tuning decisions on the heads:
checkpoint selection during gradient training (`learned_head.py`'s `train_head`,
early-stopping on `ct_ffn`/`ct_att` validation loss) **and** the subsequent
alpha-tuning (`tune_alpha_two_way`, same `ct_*` arrays). This is "double-dipping" on
one 9,652-position set for two decisions. It would make the *controller_train*-measured
performance a touch optimistic, but **cannot by itself explain an inflated *test*-set
gain** — test is a completely separate 14,097-position set never touched by either
tuning step. Worth fixing in a stricter design (e.g., a 4th split), not the likely
cause of what's being investigated here.

## 2. Representation correctness

**Exact tensors**: `h_ffn` = `hidden_states[-1]` via `FrozenGPT2.forward()`
(`gpt_class.py`) — verified earlier this session, with real numbers, that
`lm_head(hidden_ffn)` reproduces `out.logits` **exactly** (max abs diff = 0.0). This
directly answers "does the FFN head receive precisely the representation used by
the pretrained LM output layer": **yes, confirmed, not assumed.**

`h_att` = output of the last transformer block's `ln_2` (the FFN-input, post-layer-norm
representation), captured via a forward hook in `attn_hook.py`. This was corrected
mid-session after cross-checking Khandelwal et al. (2020)'s own Section 5 ablation —
the first implementation (a pre-hook, capturing `ln_2`'s *input*) matched their
measurably-worse "FFN input, before layer norm" variant (17.06 PPL); the fix (forward
hook on `ln_2`'s *output*) matches their actual best/adopted choice (16.06 PPL).
Verified locally against real GPT-2 output both times (mean abs diff from `h_ffn`:
4.52 before the fix, 0.72 after — both clearly non-trivial, confirming the hook
captures a genuinely different tensor either way, with the corrected version being
the methodologically right one).

**Same context/target alignment for both representations**: `run_batch_dual`
(`extract_dual.py:12-19`) applies the identical "chop last position" rule to both:
`shift_for_next_token` does `hidden[:, :-1, :]` for `h_ffn`, and the code does
`hidden_att[:, :-1, :]` manually for `h_att` — the exact same slicing rule, against
the exact same `y_target = input_ids[:, 1:]`. Confirmed no indexing divergence
between the two representations.

**Causal shifting**: reuses `helpers.shift_for_next_token`, the same function every
other result in this entire project depends on — not new code, already battle-tested.

## 3. Learned-head implementation

**Training data, optimizer, procedure**: `learned_head.py`'s `train_head` —
49,657 training positions (`ds_*`/`ds_values`), Adam (default `lr=1e-3`,
`weight_decay=0.0`), up to `epochs=30`, `batch_size=4096`, early-stopping
`patience=5` on `controller_train` cross-entropy, best-val-loss checkpoint restored
via `copy.deepcopy(head.state_dict())`. Confirmed correct checkpoint-restore logic by
direct read (`best_state` only updates on strict improvement; `head.load_state_dict(best_state)`
runs once at the end, not per-epoch).

**Gap found — not yet a bug, a missing diagnostic**: the pasted log reports Config
B/C's NLL *after* validation-tuned interpolation (`alpha=0.5`/`0.4`), never the
*standalone* head's own NLL (`alpha=1.0`, pure head, no base-LM blending) — exactly
what item 3 asks to compare and exactly what I don't have. The per-position
probability arrays were never serialized to `results/representation_ensembling_experiment.json`
(only scalars were saved), and the trained head objects themselves were never
persisted to disk — both live only in the now-finished job's memory. **I cannot
retroactively compute the standalone-head NLL from what currently exists; it
requires re-running** (flagged as a concrete next step in Section 6, not done here
per "do not retrain yet").

**Probability-calculation correctness — directly unit-tested this session** (new,
additive, no retraining): `LinearHead.log_probs` (softmax + log) was checked against
a hand-computed softmax on synthetic data — max abs diff `1.19e-7` (float32 precision
limit). The gather-based target-indexing (`logp.gather(-1, y.unsqueeze(-1))`) was
checked row-by-row against direct indexing — exact match. **Confirmed: no
probability-calculation or indexing bug in the head's forward path.**

**Leading, *plausible-but-unconfirmed* explanation for the FFN head's large gain**
(stated as a hypothesis, not a fact, per the instruction): I ran a cheap, additive,
tokenizer-only diagnostic (no model, no training) over 400 TinyStories documents —

```
distinct token ids used: 3,647 / 50,257  (7.26% of GPT-2's vocabulary)
top 100 token types cover    61.0% of all token occurrences
top 1,000 token types cover  91.2% of all token occurrences
```

TinyStories is an extremely narrow, repetitive corpus relative to GPT-2's original
(broad, web-scale) pretraining distribution — and **GPT-2 itself is never fine-tuned
on TinyStories anywhere in this project; it is applied zero-shot.** A freshly-trained,
**untied** linear head (unlike GPT-2's own `lm_head`, which is weight-tied to the
input embedding matrix, a representational constraint, not a free choice) gets to
specialize directly to this narrow in-domain token distribution. This is a
plausible, substantial source of "free" improvement that has nothing to do with
retrieval or even with *which* representation (FFN vs. ATT) is used — it's closer to
lightweight domain adaptation of the output layer than a representation-ensembling
effect in Xu et al.'s sense. **This is not confirmed as "the" explanation** — it is
consistent with the evidence gathered (vocabulary narrowness + weight-tying
asymmetry + the fact that B's gain exceeds even full raw kNN's) but was not isolated
via a controlled ablation.

## 4. Retrieval correctness

**Internal consistency check (strong, real evidence)**: Configs D/E/F preserve
the *exact* ordering seen in every other validated result across this entire
project — raw kNN (2.5833) < DIME (2.6741) < equal-budget raw (2.7575). This
ordering has held, without exception, across TinyStories/gpt2, TinyStories/gpt2-medium,
WikiText-103/gpt2, WikiText-103/gpt2-medium, WikiText-2/gpt2, WikiText-2/gpt2-medium,
and the GMM-variant study — strong evidence the retrieval math (`query_knn`,
`mix_knn_and_lm`, `mix_dime_and_lm`, `grid_search_hyperparams`) is behaving
correctly here too, since none of that code was touched for this experiment.

**Distance sign / temperature**: `mix_knn_and_lm`/`mix_dime_and_lm` both use
`softmax(-distance/tau)` — negative distance, so closer neighbors get higher
weight; correct convention, unchanged code.

**Brute-force cross-check**: not independently re-verified in *this* audit —
`query_knn` wraps `sklearn.neighbors.NearestNeighbors` (exact, not approximate), the
same call used and validated (including an early isolation test against hand-built
dummy vectors) throughout this project's history. I'm relying on that prior
validation rather than re-deriving it here; flagged explicitly rather than claimed
as freshly confirmed.

**DIME/raw-kNN representation matching**: confirmed by code read — `ds_att` (not
`ds_ffn`) feeds `build_datastore` for configs D, E, and F uniformly
(`run_representation_ensembling_experiment.py:196,208,227`), and the same `ct_att`/`val_att`
query accordingly. No FFN/ATT mismatch between the three retrieval configs.

## 5. Comparability with Xu et al. (2023)

The single biggest, most likely-impactful difference, stated as a difference (not
yet a quantified causal factor): **Xu et al.'s setup presumably evaluates a LM that
is trained/fine-tuned on the same corpus the datastore and learned head are built
from** (consistent with the standard kNN-LM experimental recipe, which targets
in-domain perplexity gains on e.g. WikiText-103). **This project's GPT-2 is never
fine-tuned on TinyStories at all** — it's the same frozen, web-scale-pretrained
checkpoint used everywhere else in this project. That asymmetry (in-domain datastore
and head vs. out-of-domain frozen LM) does not exist, or exists to a much smaller
degree, in a setup where the base LM was itself trained on the target corpus. I have
not read Xu et al.'s actual paper in this session to confirm their exact setup — I'm
flagging this as a difference inferred from how kNN-LM evaluations are conventionally
structured, not as a verified fact about their specific methodology.

Other known, unverified-in-detail differences: corpus scale (TinyStories ≈ tens of
thousands of training tokens in this run vs. WikiText-103's ~103M, the dataset kNN-LM
literature typically uses), head-training corpus size (same gap), model scale
(GPT-2-small, 117M — not confirmed whether this matches Xu et al.'s model choice).
**None of these were assumed away** — Section 5's instruction ("do not assume the
published ranking must hold on GPT-2 or this corpus") is being followed: the
B > C ordering is reported as a real observation on *this* corpus/model, not
evidence against Xu et al.'s own reported results on a different corpus/model.

## 6. Interpretation

**Confirmed (direct evidence, not hypothesis)**:
- Representation extraction is correct: h_FFN exactly reproduces the LM's own
  logits; h_ATT is verified to be genuinely different (both before and after the
  mid-session hook fix); shift/alignment is identical and correct for both.
- No leakage of test positions into either head-checkpoint-selection or
  interpolation-weight tuning.
- No probability-calculation or gather-indexing bug in the learned head.
- Retrieval math (D/E/F) produces the same internally-consistent ordering seen in
  every other validated result in this project.
- TinyStories' effective vocabulary is extremely narrow (7.26% of GPT-2's full
  vocabulary; top 1,000 types cover 91.2% of occurrences) — measured, not assumed.

**Plausible, not confirmed**:
- The FFN head's outsized gain is most likely explained by domain-adaptation value
  (untied, freshly-fit linear layer specializing to TinyStories' narrow token
  distribution, something GPT-2's frozen, tied `lm_head` cannot do) rather than a
  genuine "complementary representation" effect in Xu et al.'s sense — but this was
  not isolated via a dedicated ablation and should be treated as the leading
  hypothesis, not an established fact.
- The B > C ordering (FFN head beating ATT head) may simply not replicate Xu et
  al.'s reported ranking on a tiny, repetitive, out-of-domain corpus with a
  never-fine-tuned base LM — a plausible dataset/setup effect, not necessarily a
  contradiction of their result on their own setup.

**Not yet checked (real gaps, not bugs)**:
- Standalone (α=1.0) head NLL was never computed or logged — cannot yet separate
  "the head alone is extraordinary" from "interpolation with an always-present base-LM
  term is doing most of the work."
- No fresh, this-run-specific leakage re-audit (relies on the mechanism's prior
  validation, not a new check of this run's actual `chunk_story_ids`).
- No controlled test isolating domain-adaptation value from representation choice
  (e.g., comparing the FFN head's gain against what a head would show if GPT-2 *had*
  been lightly fine-tuned on TinyStories first).

**What remains justified from the existing run**: Configs D/E/F's internal
ordering (DIME beats equal-budget raw, loses to full raw kNN) — consistent with
every other validated result in this project, no reason to doubt it. The core
split-integrity and representation-correctness claims — directly verified. The
B-vs-C comparison and the magnitude of B's gain should **not** yet be treated as a
settled finding about representation ensembling — too much of the explanation is
still "plausible, unconfirmed."

**Smallest next steps before rerunning or drawing conclusions**:
1. Log standalone (α=1.0) head NLL for both B and C — cheap, additive, requires one
   rerun (can't be retrofit onto the finished job).
2. Persist per-position probability arrays (or at minimum NLL arrays) to the results
   JSON, not just scalars — needed for any future re-analysis without rerunning.
3. Add a quick, explicit re-verification that *this run's* `chunk_story_ids` has zero
   cross-split story overlap (cheap, a few lines, doesn't require retraining anything).
4. Consider an explicit domain-adaptation control: a head trained on a *shuffled*
   version of the same training tokens (same marginal token frequencies, no real
   context-to-token structure) — if that control also shows a large gain, it would
   strongly support "this is about narrow marginal statistics, not real contextual
   representation value," sharpening the hypothesis in Section 3 into a confirmed finding.
