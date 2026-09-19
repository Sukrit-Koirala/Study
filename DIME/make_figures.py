"""Generate the two paper figures: compression-quality tradeoff and illusion-check ablation.

Runs entirely on already-known numbers (no GPU/cluster needed) -- data below is
hand-curated from DIME/results/*.json and, for TinyStories/gpt2's four-point
compression sweep specifically, from DIME/logs/dime_tinystories_extras-2555.out
(that sweep was never saved to a clean JSON, only printed to the SLURM log).

Update the RESULTS dict below once the Tier 3 raw-baseline jobs land (adds the
best-of-eight raw baseline instead of just raw_kmeans_representative) and re-run.

Usage: python make_figures.py
Output: figures/compression_tradeoff.png, figures/illusion_check.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _compact_number(x, _pos=None):
    """Format a log-axis tick as 500 / 1.5K / 380K / 1.7M instead of 10^n."""
    if x <= 0:
        return "0"
    for div, suffix in ((1e6, "M"), (1e3, "K")):
        if x >= div:
            val = x / div
            return f"{val:.0f}{suffix}" if val == int(val) else f"{val:.1f}{suffix}"
    return f"{x:.0f}"


def use_compact_log_xaxis(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_compact_number))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Source: DIME/results/{prefix}_tier1.json, {prefix}_tier2.json for all settings
# except TinyStories/gpt2, which predates the tier system and is sourced from
# significance_testing.json (gpt_only, dime_tuned, raw_kmeans_representative),
# tinystories_rich_ablations.json (majority_token, top5, shuffled_distribution),
# and dime_tinystories_extras-2555.out (the compression sweep points).
RESULTS = {
    "TinyStories/gpt2": {
        "N": 49657, "B": 500,
        "gpt_only": 2.7984, "raw_knn": 2.6941, "dime": 2.7452, "raw_best": 2.7818,
        "majority_token": 2.7646, "top5": 2.7471, "shuffled": 2.8349,
        # Only setting with real untuned numbers -- the original roadmap script ran
        # an untuned pass before grid search for both raw kNN and DIME;
        # run_tier1_core.py (used for all 5 other settings) goes straight to grid
        # search, no untuned step exists there. Source: results/minibatch_kmeans_baseline.json
        # ("mean_nll_mixed": 2.7924) and PAPER_CONTEXT's "raw kNN, untuned -> 2.757".
        "raw_knn_untuned": 2.757,
        "dime_untuned": 2.7924,
    },
    "TinyStories/gpt2-medium": {
        "N": 381000, "B": 3810,
        "gpt_only": 2.501, "raw_knn": 2.404, "dime": 2.463, "raw_best": 2.590,
        "majority_token": 2.472, "top5": 2.456, "shuffled": 2.588,
    },
    "WikiText-103/gpt2": {
        "N": 381127, "B": 3811,
        "gpt_only": 4.361, "raw_knn": 3.883, "dime": 4.011, "raw_best": 4.190,
        "majority_token": 4.049, "top5": 4.009, "shuffled": 4.272,
    },
    "WikiText-103/gpt2-medium": {
        "N": 381127, "B": 3811,
        "gpt_only": 4.049, "raw_knn": 3.659, "dime": 3.754, "raw_best": 3.956,
        "majority_token": 3.770, "top5": 3.739, "shuffled": 3.985,
    },
    "WikiText-2/gpt2": {
        "N": 1677035, "B": 16770,
        "gpt_only": 4.343, "raw_knn": 3.844, "dime": 3.974, "raw_best": 4.118,
        "majority_token": 4.052, "top5": 3.979, "shuffled": 4.388,
    },
    "WikiText-2/gpt2-medium": {
        "N": 1677035, "B": 16770,
        "gpt_only": 4.022, "raw_knn": 3.614, "dime": 3.722, "raw_best": 3.865,
        "majority_token": 3.747, "top5": 3.708, "shuffled": 3.992,
    },
}

SETTINGS = list(RESULTS.keys())

# Estimate untuned numbers for the 5 settings that don't have real ones yet, using
# the relative tuning gap measured on the one real data point (TinyStories/gpt2).
# These are PLACEHOLDERS -- replace with real numbers once those jobs are run;
# plotted with reduced opacity and a distinct legend entry so they're never
# mistaken for measured data.
_REF = RESULTS["TinyStories/gpt2"]
RAW_KNN_UNTUNED_RATIO = _REF["raw_knn_untuned"] / _REF["raw_knn"]   # ~1.0233 (+2.33%)
DIME_UNTUNED_RATIO = _REF["dime_untuned"] / _REF["dime"]            # ~1.0172 (+1.72%)

for _setting, _r in RESULTS.items():
    if "raw_knn_untuned" not in _r:
        _r["raw_knn_untuned"] = _r["raw_knn"] * RAW_KNN_UNTUNED_RATIO
        _r["raw_knn_untuned_is_estimate"] = True
    if "dime_untuned" not in _r:
        _r["dime_untuned"] = _r["dime"] * DIME_UNTUNED_RATIO
        _r["dime_untuned_is_estimate"] = True


# Validated categorical palette (dataviz skill, references/palette.md) -- first
# three slots clear all-pairs CVD/normal-vision gates in both modes, so DIME /
# best-raw / raw-kNN can share this legend across every small-multiple panel.
C_DIME = "#2a78d6"      # slot 1, blue -- the method the paper is about
C_RAWBEST = "#eb6834"   # slot 2, orange
C_RAWKNN = "#1baf7a"    # slot 3, aqua/green
C_DIME_SERIES = "#4a3aa7"  # slot 7, violet -- DIME as its own series (distinct from
                            # C_DIME, which is reused elsewhere as gpt2's model color)
C_MUTED = "#898781"     # muted ink -- GPT-only is a reference floor, not a competing series
C_GRID = "#e1e0d9"      # hairline gridline


def plot_merged():
    """One figure: storage-vs-quality tradeoff, with the illusion-check content
    variants (majority-token/top-5/shuffled) placed at DIME's own budget B,
    fanned out on the log-x axis for legibility. Untuned points are deliberately
    left out here -- that's a different experimental axis (hyperparameter tuning,
    not content/geometry) and would just clutter an already-busy cluster at x~B."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, setting in zip(axes.flat, SETTINGS):
        r = RESULTS[setting]
        B = r["B"]

        ax.axhline(r["gpt_only"], color=C_MUTED, linestyle="--", linewidth=1.2, zorder=2)
        ax.scatter([r["N"]], [r["raw_knn"]], color=C_RAWKNN, s=90, marker="^",
                   zorder=5, edgecolors="#fcfcfb", linewidths=1.2)
        ax.scatter([B * 0.8], [r["raw_best"]], color=C_RAWBEST, s=90, marker="s",
                   zorder=5, edgecolors="#fcfcfb", linewidths=1.2)

        # DIME's own budget B: tuned point plus the illusion-check content variants,
        # same hue (same entity -- all are "DIME at budget B"), shape encodes variant.
        ax.scatter([B * 1.0], [r["dime"]], color=C_DIME, s=95, marker="o",
                   zorder=6, edgecolors="#fcfcfb", linewidths=1.2)
        ax.scatter([B * 1.3], [r["majority_token"]], color=C_DIME, s=75, marker="D",
                   zorder=5, edgecolors="#fcfcfb", linewidths=1.0)
        ax.scatter([B * 1.7], [r["top5"]], color=C_DIME, s=85, marker="p",
                   zorder=5, edgecolors="#fcfcfb", linewidths=1.0)
        ax.scatter([B * 2.2], [r["shuffled"]], color=C_DIME, s=85, marker="X",
                   zorder=5, edgecolors="#fcfcfb", linewidths=1.0)

        ax.set_xscale("log")
        erased = r["shuffled"] > r["gpt_only"]
        ax.set_title(f"{setting}  ({'shuffle erases it' if erased else 'shuffle survives'})",
                     fontsize=10, color="#0b0b0b")
        ax.set_xlabel("stored entries (log)", fontsize=8.5, color="#52514e")
        ax.set_ylabel("mean val NLL", fontsize=8.5, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")
        ax.grid(True, which="major", axis="both", color=C_GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")

    legend_handles = [
        plt.Line2D([0], [0], color=C_MUTED, linestyle="--", linewidth=1.2, label="GPT-only (no retrieval)"),
        plt.Line2D([0], [0], marker="o", color=C_DIME, linestyle="", markersize=9, label="DIME (tuned)"),
        plt.Line2D([0], [0], marker="D", color=C_DIME, linestyle="", markersize=8, label="DIME (majority-token)"),
        plt.Line2D([0], [0], marker="p", color=C_DIME, linestyle="", markersize=9, label="DIME (top-5)"),
        plt.Line2D([0], [0], marker="X", color=C_DIME, linestyle="", markersize=8, label="DIME (shuffled)"),
        plt.Line2D([0], [0], marker="s", color=C_RAWBEST, linestyle="", markersize=9, label="best equal-budget raw"),
        plt.Line2D([0], [0], marker="^", color=C_RAWKNN, linestyle="", markersize=9, label="raw kNN (tuned)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04),
               frameon=False, fontsize=9.5)
    fig.suptitle("Storage-quality tradeoff and content ablations, per setting", fontsize=13, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out_path = os.path.join(OUT_DIR, "results_overview.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


def plot_grid_summary():
    """Single combined diagram: the compression-quality tradeoff across the whole
    grid, one row per setting. Raw NLL isn't comparable across settings (TinyStories
    sits near 2.5, WikiText near 4.0), so each row is normalized to "% of the gap
    between GPT-only (0%) and raw kNN's tuned ceiling (100%) that gets closed" --
    a common, meaningful x-axis. This is a horizontal dumbbell chart: color encodes
    method (reused identically per row), not setting, so it stays a single legend."""
    def normalize(x, r):
        span = r["gpt_only"] - r["raw_knn"]
        return (r["gpt_only"] - x) / span * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = list(reversed(SETTINGS))  # top-to-bottom in reading order
    y_pos = {s: i for i, s in enumerate(rows)}

    for s in rows:
        r = RESULTS[s]
        y = y_pos[s]
        dime_pct = normalize(r["dime"], r)
        raw_best_pct = normalize(r["raw_best"], r)
        ax.plot([0, 100], [y, y], color=C_GRID, linewidth=3, zorder=1, solid_capstyle="round")
        ax.scatter([0], [y], color=C_MUTED, s=45, marker="*", zorder=4)
        ax.scatter([100], [y], color=C_RAWKNN, s=70, marker="^", zorder=4, edgecolors="#fcfcfb", linewidths=1)
        ax.scatter([raw_best_pct], [y], color=C_RAWBEST, s=90, marker="s", zorder=5, edgecolors="#fcfcfb", linewidths=1.2)
        ax.scatter([dime_pct], [y], color=C_DIME, s=100, marker="o", zorder=6, edgecolors="#fcfcfb", linewidths=1.2)
        ax.text(dime_pct, y + 0.28, f"{dime_pct:.0f}%", ha="center", fontsize=8, color="#52514e")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10, color="#0b0b0b")
    ax.set_xlim(-8, 112)
    ax.set_ylim(-0.8, len(rows) - 0.2)  # headroom above top row so labels clear the title
    ax.set_xlabel("% of GPT-only -> raw-kNN gap closed  (0% = GPT-only, 100% = raw kNN ceiling)",
                  fontsize=9.5, color="#52514e")
    ax.tick_params(labelsize=9, colors="#52514e")
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    legend_handles = [
        plt.Line2D([0], [0], marker="*", color=C_MUTED, linestyle="", markersize=9, label="GPT-only (0%)"),
        plt.Line2D([0], [0], marker="s", color=C_RAWBEST, linestyle="", markersize=9, label="best equal-budget raw"),
        plt.Line2D([0], [0], marker="o", color=C_DIME, linestyle="", markersize=9, label="DIME (tuned)"),
        plt.Line2D([0], [0], marker="^", color=C_RAWKNN, linestyle="", markersize=9, label="raw kNN ceiling (100%)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9)
    ax.set_title("Compression-quality tradeoff, normalized across the full grid", fontsize=13, color="#0b0b0b", pad=14)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUT_DIR, "compression_tradeoff_grid.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


DATASET_MARKER = {
    "TinyStories": "o",
    "WikiText-103": "D",
    "WikiText-2": "^",
}
MODEL_COLOR = {
    "gpt2": "#2a78d6",         # slot 1, blue
    "gpt2-medium": "#eb6834",  # slot 2, orange
}
MODEL_LINESTYLE = {
    "gpt2": ":",
    "gpt2-medium": "-",
}
GPT_ONLY_X = 10  # nominal placeholder (no real storage cost) -- log-scale needs x>0


def plot_grid_trajectories():
    """Single combined diagram, one dashed trajectory per setting: GPT-only -> best
    equal-budget raw -> DIME -> raw kNN, left to right on a log storage axis. Shape
    encodes dataset, color encodes model size -- a natural 3x2 split of the 6
    settings, instead of needing 6 arbitrary hues."""
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for setting in SETTINGS:
        dataset, model = setting.split("/")
        r = RESULTS[setting]
        marker = DATASET_MARKER[dataset]
        color = MODEL_COLOR[model]
        xs = [GPT_ONLY_X, r["B"] * 0.85, r["B"] * 1.15, r["N"]]
        ys = [r["gpt_only"], r["raw_best"], r["dime"], r["raw_knn"]]
        ax.plot(xs, ys, color=color, linewidth=1.3, linestyle="--", alpha=0.55, zorder=2)
        ax.scatter(xs, ys, color=color, marker=marker, s=85, zorder=5,
                   edgecolors="#fcfcfb", linewidths=1.1)

    ax.set_xscale("log")
    ax.set_xlabel("stored entries (log; leftmost = GPT-only, no storage)", fontsize=9.5, color="#52514e")
    ax.set_ylabel("mean val NLL", fontsize=9.5, color="#52514e")
    ax.tick_params(labelsize=9, colors="#52514e")
    ax.grid(True, which="major", axis="both", color=C_GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")

    dataset_handles = [
        plt.Line2D([0], [0], marker=m, color="#52514e", linestyle="", markersize=9, label=d)
        for d, m in DATASET_MARKER.items()
    ]
    model_handles = [
        plt.Line2D([0], [0], marker="o", color=c, linestyle="", markersize=9, label=m)
        for m, c in MODEL_COLOR.items()
    ]
    legend1 = ax.legend(handles=dataset_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         title="dataset (shape)", fontsize=9, borderaxespad=0)
    ax.add_artist(legend1)
    ax.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(1.02, 0.62),
              title="model (color)", fontsize=9, borderaxespad=0)

    ax.set_title("Storage-quality trajectories: GPT-only -> best raw -> DIME -> raw kNN",
                  fontsize=13, color="#0b0b0b")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "grid_trajectories.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


def plot_baseline_scaling():
    """GPT-only NLL is constant per setting -- it never touches the datastore, so
    it doesn't belong at a single x. Drawn as a horizontal line (dotted = gpt2,
    solid = gpt2-medium) spanning the shared datastore-size range, with the
    dataset's shape marker repeated along it. Color carries series identity
    (GPT-only / raw kNN / DIME); model is covered by linestyle+fill; shape covers
    dataset.

    Linear x-axis, deliberately not log: log scale equalizes ratios, so a 10x jump
    from 100K->1M reads as the same visual distance as 1K->10K, which flattens how
    much bigger WikiText-2's compression actually is in absolute terms. Linear is
    the only scale that shows that honestly, at the cost of TinyStories' points
    bunching up near the origin."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    all_x = [r["N"] for r in RESULTS.values()] + [r["B"] for r in RESULTS.values()]
    x_lo, x_hi = -max(all_x) * 0.03, max(all_x) * 1.05
    marker_xs = np.linspace(x_lo + max(all_x) * 0.02, x_hi - max(all_x) * 0.02, 6)

    for setting in SETTINGS:
        dataset, model = setting.split("/")
        r = RESULTS[setting]
        marker = DATASET_MARKER[dataset]
        line_style = MODEL_LINESTYLE[model]
        # gpt2 = hollow markers, gpt2-medium = filled -- a second, redundant cue for
        # model on top of linestyle, so it doesn't rely on the line alone.
        hollow = (model == "gpt2")
        y = r["gpt_only"]
        ax.plot([x_lo, x_hi], [y, y], color=C_MUTED, linewidth=1, linestyle=line_style, alpha=0.7, zorder=2)
        ax.scatter(marker_xs, [y] * len(marker_xs),
                   facecolors="none" if hollow else C_MUTED, edgecolors=C_MUTED,
                   marker=marker, s=30, zorder=5, linewidths=1.0 if hollow else 0.7)

        # raw kNN and DIME: just points at their real x (N and B) -- same dataset
        # shape, own fixed series color, same hollow/filled convention for model.
        ax.scatter([r["N"]], [r["raw_knn"]],
                   facecolors="none" if hollow else C_RAWKNN, edgecolors=C_RAWKNN,
                   marker=marker, s=55, zorder=6, linewidths=1.3 if hollow else 0.9)
        ax.scatter([r["B"]], [r["dime"]],
                   facecolors="none" if hollow else C_DIME_SERIES, edgecolors=C_DIME_SERIES,
                   marker=marker, s=55, zorder=6, linewidths=1.3 if hollow else 0.9)

    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_compact_number))
    ax.set_xlabel("datastore size (linear -- shows true absolute scale, not just ratio)",
                  fontsize=9.5, color="#52514e")
    ax.set_ylabel("mean val NLL", fontsize=9.5, color="#52514e")
    ax.tick_params(labelsize=9, colors="#52514e")
    ax.grid(True, which="major", axis="y", color=C_GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")

    dataset_handles = [
        plt.Line2D([0], [0], marker=m, color="#52514e", linestyle="", markersize=9, label=d)
        for d, m in DATASET_MARKER.items()
    ]
    model_handles = [
        plt.Line2D([0], [0], color="#52514e", linestyle=ls, linewidth=1.5,
                   marker="o", markersize=8,
                   markerfacecolor="none" if m == "gpt2" else "#52514e",
                   markeredgecolor="#52514e", label=m)
        for m, ls in MODEL_LINESTYLE.items()
    ]
    series_handles = [
        plt.Line2D([0], [0], marker="o", color=C_MUTED, linestyle="", markersize=9, label="GPT-only"),
        plt.Line2D([0], [0], marker="o", color=C_RAWKNN, linestyle="", markersize=9, label="raw kNN (tuned)"),
        plt.Line2D([0], [0], marker="o", color=C_DIME_SERIES, linestyle="", markersize=9, label="DIME (tuned)"),
    ]
    legend1 = ax.legend(handles=dataset_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         title="dataset (shape)", fontsize=9, borderaxespad=0)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(1.02, 0.68),
                         title="model (line + fill)", fontsize=9, borderaxespad=0)
    ax.add_artist(legend2)
    ax.legend(handles=series_handles, loc="upper left", bbox_to_anchor=(1.02, 0.44),
              title="series (color)", fontsize=9, borderaxespad=0)

    ax.set_title("GPT-only, raw kNN, and DIME, by dataset and model", fontsize=13, color="#0b0b0b")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "baseline_scaling.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_merged()
    plot_grid_summary()
    plot_grid_trajectories()
    plot_baseline_scaling()
