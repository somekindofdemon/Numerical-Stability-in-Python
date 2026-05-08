"""
Stage 5 — visualize
Produces 4 figures as required by the assignment:

  fig1_distributions.png  — KDE of intra/inter distances for each precision
  fig2_comparison.png     — mean distances bar chart + separability ratio
  fig3_residuals.png      — per-pair numerical error vs float64
  fig4_efficiency.png     — storage size and computation time
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import gaussian_kde

PRECISIONS   = ["float64", "float32", "float16", "int8"]
PREC_COLORS  = {"float64": "#2c3e50", "float32": "#2980b9",
                "float16": "#e67e22", "int8":    "#c0392b"}
INTRA_COLOR  = "#2980b9"
INTER_COLOR  = "#c0392b"
FIG_DIR      = "data/figures"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def kde(values, n=400):
    values = np.asarray(values, dtype=np.float64)
    est = gaussian_kde(values, bw_method="scott")
    xs = np.linspace(max(0, values.min() - 0.04),
                     min(1.3, values.max() + 0.04), n)
    return xs, est(xs)


# ── Figure 1 : Distance distributions ────────────────────────────────────────
def fig1_distributions(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, prec in zip(axes, PRECISIONS):
        raw = results[prec]["raw"]
        xi, yi = kde(raw["intra"])
        xe, ye = kde(raw["inter"])

        ax.fill_between(xi, yi, alpha=0.18, color=INTRA_COLOR)
        ax.fill_between(xe, ye, alpha=0.18, color=INTER_COLOR)
        ax.plot(xi, yi, color=INTRA_COLOR, linewidth=2,
                label=f'intra-speaker  (μ={np.mean(raw["intra"]):.3f})')
        ax.plot(xe, ye, color=INTER_COLOR, linewidth=2,
                label=f'inter-speaker  (μ={np.mean(raw["inter"]):.3f})')

        ax.set_title(prec, fontsize=13, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, framealpha=0.6)

    for ax in axes[2:]:
        ax.set_xlabel("Cosine distance")

    fig.suptitle(
        "Distribution of cosine distances — intra-speaker vs inter-speaker\n"
        "across precision levels",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1_distributions.png")


# ── Figure 2 : Comparison (means + ratio) ────────────────────────────────────
def fig2_comparison(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # — left : connected dot plot (no forced zero baseline) —
    imeans = [results[p]["intra_mean"] for p in PRECISIONS]
    emeans = [results[p]["inter_mean"] for p in PRECISIONS]

    ax1.plot(PRECISIONS, imeans, color=INTRA_COLOR, linewidth=1.8,
             marker="o", markersize=0, zorder=1)
    ax1.plot(PRECISIONS, emeans, color=INTER_COLOR, linewidth=1.8,
             marker="o", markersize=0, zorder=1)

    for i, (p, iv, ev) in enumerate(zip(PRECISIONS, imeans, emeans)):
        ax1.scatter(i, iv, color=INTRA_COLOR, s=80, zorder=2)
        ax1.scatter(i, ev, color=INTER_COLOR, s=80, zorder=2)
        ax1.annotate(f"{iv:.10f}", (i, iv), textcoords="offset points",
                     xytext=(-58, 4), fontsize=7.5, color=INTRA_COLOR)
        ax1.annotate(f"{ev:.10f}", (i, ev), textcoords="offset points",
                     xytext=(-58, 4), fontsize=7.5, color=INTER_COLOR)

    # tight y-axis — does not start from 0
    all_means = imeans + emeans
    margin = (max(all_means) - min(all_means)) * 0.6
    ax1.set_ylim(min(all_means) - margin, max(all_means) + margin)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.10f"))

    # custom legend
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([0], [0], color=INTRA_COLOR, marker="o", linewidth=1.8, label="intra-speaker"),
        Line2D([0], [0], color=INTER_COLOR,  marker="o", linewidth=1.8, label="inter-speaker"),
    ], fontsize=9)
    ax1.set_ylabel("Mean cosine distance")
    ax1.set_title("Mean intra- and inter-speaker distances\nby precision level")

    # — right : separability ratio —
    ratios = [results[p]["ratio_inter_intra"] for p in PRECISIONS]
    ax2.plot(PRECISIONS, ratios, color="#555", linewidth=1.8,
             marker="o", markersize=0, zorder=1)
    for i, (p, r) in enumerate(zip(PRECISIONS, ratios)):
        ax2.scatter(i, r, color=PREC_COLORS[p], s=90, zorder=2)
        ax2.annotate(f"{r:.10f}", (i, r),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=7.5)
    ax2.axhline(ratios[0], linestyle="--", color="gray",
                alpha=0.55, linewidth=1.2, label="float64 baseline")

    # tight y-axis
    ax2.set_ylim(min(ratios) - 0.001, max(ratios) + 0.001)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax2.set_ylabel("inter / intra ratio")
    ax2.set_title("Speaker separability ratio\nacross precision levels")
    ax2.legend(fontsize=9)

    fig.suptitle("Effect of numerical precision on distance structure", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2_comparison.png")


# ── Figure 3 : Per-pair residuals ────────────────────────────────────────────
def fig3_residuals(results):
    ref_intra = np.array(results["float64"]["raw"]["intra"], dtype=np.float64)
    ref_inter = np.array(results["float64"]["raw"]["inter"], dtype=np.float64)
    compare   = [p for p in PRECISIONS if p != "float64"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (ref, kind, title) in zip(axes, [
        (ref_intra, "intra", "intra-speaker pairs"),
        (ref_inter, "inter", "inter-speaker pairs"),
    ]):
        data   = [np.array(results[p]["raw"][kind], dtype=np.float64) - ref
                  for p in compare]
        colors = [PREC_COLORS[p] for p in compare]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker=".", markersize=1.5, alpha=0.25))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        # annotate median
        for i, d in enumerate(data):
            ax.text(i + 1, np.median(d),
                    f" {np.median(d):+.1e}", va="center",
                    fontsize=8, color="black")

        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.45)
        ax.set_xticks(range(1, len(compare) + 1))
        ax.set_xticklabels(compare)
        ax.set_ylabel("d_prec − d_float64")
        ax.set_title(f"Per-pair distance error — {title}")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))

    fig.suptitle(
        "Numerical error introduced by precision reduction\n"
        "(each box = distribution over all pairs, relative to float64 reference)",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig3_residuals.png")


# ── Figure 4 : Efficiency (storage + timing) ─────────────────────────────────
def fig4_efficiency(results):
    files  = {p: f"data/reps_{p}.npy" for p in PRECISIONS}
    sizes  = [os.path.getsize(files[p]) / 1e6 for p in PRECISIONS]
    times  = [results[p]["compute_time_s"] for p in PRECISIONS]
    colors = [PREC_COLORS[p] for p in PRECISIONS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, vals, ylabel, title, unit in [
        (ax1, sizes, "Size (MB)",    "Storage size of representations", "MB"),
        (ax2, times, "Time (s)",     "Distance matrix computation time", "s"),
    ]:
        bars = ax.bar(PRECISIONS, vals, color=colors, alpha=0.82, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(vals) * 0.02,
                    f"{v:.2f} {unit}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.18)

    fig.suptitle("Computational cost of reduced precision", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig4_efficiency.png")


def main():
    with open("data/distances.json") as f:
        results = json.load(f)

    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating figures...")

    fig1_distributions(results)
    fig2_comparison(results)
    fig3_residuals(results)
    fig4_efficiency(results)

    print("Done.")


if __name__ == "__main__":
    main()
