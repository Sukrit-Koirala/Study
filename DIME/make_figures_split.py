"""Split version of baseline_scaling.png: same data (GPT-only / raw kNN / DIME,
one setting each), but one panel per setting instead of one combined linear plot.

Each panel gets its own x-range (that setting's real B-to-N span), so TinyStories
and WikiText-103 -- squashed near the origin in the combined linear plot -- get
full resolution here. Trade-off: settings are no longer directly comparable on a
shared axis, which is exactly the "let me see the difference" this file is for.

Usage: python make_figures_split.py
Output: figures/baseline_scaling_split.png
"""
import os
import matplotlib.pyplot as plt

from make_figures import (
    RESULTS, SETTINGS, DATASET_MARKER, MODEL_LINESTYLE,
    C_MUTED, C_RAWKNN, C_DIME_SERIES, OUT_DIR, _compact_number,
)
import matplotlib.ticker as mticker


def plot_baseline_scaling_split():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, setting in zip(axes.flat, SETTINGS):
        dataset, model = setting.split("/")
        r = RESULTS[setting]
        marker = DATASET_MARKER[dataset]
        line_style = MODEL_LINESTYLE[model]
        hollow = (model == "gpt2")
        y = r["gpt_only"]

        x_lo, x_hi = -r["N"] * 0.05, r["N"] * 1.08
        ax.plot([x_lo, x_hi], [y, y], color=C_MUTED, linewidth=1.2, linestyle=line_style, alpha=0.8, zorder=2)
        ax.scatter([r["N"] * 0.5], [y], facecolors="none" if hollow else C_MUTED, edgecolors=C_MUTED,
                   marker=marker, s=70, zorder=5, linewidths=1.3 if hollow else 0.9)
        ax.scatter([r["N"]], [r["raw_knn"]], facecolors="none" if hollow else C_RAWKNN, edgecolors=C_RAWKNN,
                   marker=marker, s=110, zorder=6, linewidths=1.6 if hollow else 1.1)
        ax.scatter([r["B"]], [r["dime"]], facecolors="none" if hollow else C_DIME_SERIES, edgecolors=C_DIME_SERIES,
                   marker=marker, s=110, zorder=6, linewidths=1.6 if hollow else 1.1)

        ax.set_xlim(x_lo, x_hi)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_compact_number))
        ax.set_title(setting, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("datastore size (linear)", fontsize=8.5, color="#52514e")
        ax.set_ylabel("mean val NLL", fontsize=8.5, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=C_MUTED, linestyle="", markersize=9, label="GPT-only"),
        plt.Line2D([0], [0], marker="o", color=C_RAWKNN, linestyle="", markersize=9, label="raw kNN (tuned)"),
        plt.Line2D([0], [0], marker="o", color=C_DIME_SERIES, linestyle="", markersize=9, label="DIME (tuned)"),
        plt.Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#52514e",
                   linestyle="", markersize=9, label="gpt2 (hollow)"),
        plt.Line2D([0], [0], marker="o", color="#52514e", linestyle="", markersize=9, label="gpt2-medium (filled)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9.5)
    fig.suptitle("GPT-only, raw kNN, and DIME -- one panel per setting, own scale", fontsize=13, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path = os.path.join(OUT_DIR, "baseline_scaling_split.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_baseline_scaling_split()
