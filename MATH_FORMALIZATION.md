# Formalizing the Mathematics — Working Notes & Mockup

## What this document is

This is a first mockup of the "formalize the math" work Austin asked for — not a
finished paper section, a worked example to learn the *shape* of what that means
before doing it for real across the whole project.

**The core idea:** you already have *measured* numbers from running experiments
("this specific run, with N=49,657 and B=500, used 152.94 MB"). Formalizing means
writing the *general formula* those numbers are one instance of — something like
`M = N(4D+8)` — so a reader can predict the cost for *any* N, D, B without rerunning
the experiment. A paper needs both: the formula (so the claim is general and
falsifiable) and the measured number (so the formula is verified against reality, not
just asserted).

## Background needed (you already have most of this)

- **Basic algebra** — this is literally counting bytes and multiplying, nothing more.
- **Big-O / asymptotic notation** (`O(N·D)`) — one concept: describing how a cost
  *scales* with size, not its exact value. Used in the latency argument below.
- **A little probability** — you already built and understand softmax weighting and
  the `beta` shrinkage formula in code; formalizing just means writing them as
  equations instead of code.
- **LaTeX**, eventually, to typeset the final paper — not needed to *understand* this,
  just to write the finished document later.

---

## Worked example: byte-level compression ratio

### Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of raw datastore entries |
| $B$ | number of compressed states (clusters) |
| $D$ | hidden dimension |
| $V$ | vocabulary size |
| $K$ | fixed sparsity budget (top-$K$ tokens retained per compressed state) |

### Memory footprint

Each **raw** entry stores one key (a $D$-dimensional float32 vector) and one value
(an int64 token id):

$$M_{\text{raw}} = N(4D + 8) \text{ bytes}$$

Each **compressed (DIME)** state stores one key (same cost) plus a sparse top-$K$
token distribution (each of the $K$ entries: one int32 token id + one float32 count):

$$M_{\text{DIME}} = B(4D + 8K) \text{ bytes}$$

### Compression ratio

$$\rho_{\text{bytes}} = \frac{M_{\text{raw}}}{M_{\text{DIME}}} = \frac{N(4D+8)}{B(4D+8K)}$$

### Checking this against real measured data

Using this project's own TinyStories/GPT-2-small numbers: $N=49{,}657$, $B=500$,
$D=768$, $K=5$.

$$M_{\text{raw}} = 49{,}657 \times (4{\times}768+8) = 152{,}943{,}560 \text{ bytes} = 152.94\text{ MB}$$

— matches the Phase F16 efficiency-analysis measurement **exactly**.

$$M_{\text{DIME}} = 500 \times (4{\times}768 + 8{\times}5) = 1{,}556{,}000 \text{ bytes} = 1.56\text{ MB}, \quad \rho_{\text{bytes}} \approx 98.3\text{x}$$

— the measured top-5-truncation result was **98.5x**. The formula predicts the
measurement to within rounding, confirming this is the right general relationship,
not a coincidence specific to one run.

### Retrieval latency

Brute-force nearest-neighbor search over $N$ candidates costs $O(N \cdot D)$ per
query (one distance computed per stored entry). Since $\rho_{\text{bytes}} \approx
\rho_{\text{entries}} = N/B$ when $D \gg K$, the *asymptotic* speedup from
compression should also scale as $N/B \approx 99.3\text{x}$.

The **measured** speedup was **58.6x** — lower than the asymptotic prediction,
because real latency includes fixed per-call overhead (Python/`sklearn` dispatch
cost) that doesn't shrink with the datastore. State this gap honestly in the paper
rather than paper over it — it's a real, explainable discrepancy between the
theoretical scaling law and the measured constant-factor reality.

---

## Not yet done — next pieces to formalize

1. **The read-time mixing formula itself** — write
   $p_{\text{mixed}} = \alpha \cdot p_{\text{retrieval}} + (1-\alpha) \cdot p_{\text{LM}}$,
   the softmax retrieval-weighting, and the `beta`-shrinkage extension in clean
   notation (all already implemented in code, just needs to be written as equations).
2. **A semi-formal bias-variance argument** for why compression can preserve (or
   possibly exceed) raw retrieval's quality at a fixed budget — ties directly into
   the compression-sweep investigation (does DIME's curve ever cross above raw kNN's
   own tuned ceiling at a moderate compression ratio?).
3. **Apply the same byte/latency formulas to the WikiText-103/gpt2-medium numbers**
   to confirm the formulas generalize across scale, not just the one TinyStories
   config used above.
4. Re-derive $\rho_{\text{bytes}}$ for the *dense* (non-truncated) state-object case
   too, for completeness — the current derivation assumes fixed-$K$ sparse storage.
