"""Q2 figures: embedding maps, metric comparison and similarity structure.

The palette is colour-blind safe. On the t-SNE maps colour encodes topic (three hues) and marker
shape encodes the filing within it, so no reading depends on colour alone; every figure has a
legend and its numbers are also in summary.md.
"""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from virallens.common.catalog import CATALOG, Topic

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TOPIC_COLOURS = dict(zip([str(t) for t in Topic], ["#2a78d6", "#eb6834", "#1baf7a"], strict=True))
FILING_MARKERS = dict(
    zip([info.short_name for info in CATALOG.values()], ["o", "^", "v", "s", "D"], strict=True)
)
SINGLE_SERIES = "#2a78d6"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# Marker area in pt²: about 8 px across at 100 dpi, large enough to read the shape.
MARKER_AREA = 48
# t-SNE perplexity (an effective neighbour count) must stay below the ~70 held-out points; 15 is a
# chosen value, about a fifth of them.
TSNE_PERPLEXITY = 15


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(TEXT_SECONDARY)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def tsne_maps(
    embeddings: dict[str, np.ndarray], topics: list[str], filings: list[str], seed: int, path: Path
) -> None:
    fig = Figure(figsize=(5 * len(embeddings), 4.6), facecolor=SURFACE)
    axes = fig.subplots(1, len(embeddings))
    for ax, (name, vectors) in zip(axes, embeddings.items(), strict=True):
        points = TSNE(
            perplexity=TSNE_PERPLEXITY, metric="cosine", init="pca", random_state=seed
        ).fit_transform(vectors)
        for filing, marker in FILING_MARKERS.items():
            rows = [i for i, f in enumerate(filings) if f == filing]
            ax.scatter(
                points[rows, 0],
                points[rows, 1],
                s=MARKER_AREA,
                marker=marker,
                c=[TOPIC_COLOURS[topics[i]] for i in rows],
                edgecolors=SURFACE,
                linewidths=1.0,
            )
        ax.set_title(name, color=TEXT_PRIMARY, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        _style(ax)
    topic_handles = [
        Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=t.replace("_", " "))
        for t, c in TOPIC_COLOURS.items()
    ]
    filing_handles = [
        Line2D([], [], marker=m, linestyle="", color=TEXT_SECONDARY, markersize=7, label=f)
        for f, m in FILING_MARKERS.items()
    ]
    fig.legend(
        handles=topic_handles + filing_handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT_PRIMARY,
    )
    fig.suptitle(
        "Held-out chunks, t-SNE of each embedding (colour: topic, shape: filing)",
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=100, facecolor=SURFACE)


def metric_bars(summary: dict[str, dict[str, tuple[float, float]]], path: Path) -> None:
    """One panel per metric; ``summary[metric][model] = (mean, std)`` over seeds."""
    metrics = list(summary)
    fig = Figure(figsize=(3.4 * len(metrics), 2.8), facecolor=SURFACE)
    axes = fig.subplots(1, len(metrics))
    for ax, metric in zip(axes, metrics, strict=True):
        models = list(summary[metric])
        means = [summary[metric][m][0] for m in models]
        stds = [summary[metric][m][1] for m in models]
        y = np.arange(len(models))
        ax.barh(
            y,
            means,
            height=0.6,
            color=SINGLE_SERIES,
            xerr=stds,
            error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1, "capsize": 3},
        )
        # Labels sit past the error bar so the cap never overprints the number.
        for yi, mean, std in zip(y, means, stds, strict=True):
            ax.text(
                mean + std,
                yi,
                f"  {mean:.2f}",
                va="center",
                ha="left",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
        ax.set_yticks(y, models, color=TEXT_PRIMARY)
        ax.invert_yaxis()
        ax.set_title(metric, color=TEXT_PRIMARY, fontsize=10)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.6)
        ax.margins(x=0.25)
        _style(ax)
    fig.suptitle(
        "Topic separation on held-out chunks (mean ± std over seeds)",
        color=TEXT_PRIMARY,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=100, facecolor=SURFACE)


def similarity_heatmaps(embeddings: dict[str, np.ndarray], filings: list[str], path: Path) -> None:
    """Cosine similarity of held-out chunks, ordered by filing, with filing boundaries marked."""
    order = sorted(range(len(filings)), key=lambda i: list(FILING_MARKERS).index(filings[i]))
    ordered = [filings[i] for i in order]
    bounds = [i for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1]]
    centres = []
    start = 0
    for end in [*bounds, len(ordered)]:
        centres.append(((start + end - 1) / 2, ordered[start]))
        start = end
    # One shared scale from the lowest cosine in any panel: LSA has negative similarities, which a
    # floor of 0 would clip to the lightest colour.
    low = min(float((v @ v.T).min()) for v in embeddings.values())
    fig = Figure(figsize=(4.6 * len(embeddings), 4.4), facecolor=SURFACE)
    axes = fig.subplots(1, len(embeddings))
    for panel, (ax, (name, vectors)) in enumerate(zip(axes, embeddings.items(), strict=True)):
        v = vectors[order]
        image = ax.imshow(v @ v.T, cmap=SEQUENTIAL, vmin=low, vmax=1, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color=SURFACE, linewidth=1.5)
            ax.axvline(b - 0.5, color=SURFACE, linewidth=1.5)
        # Filing names on the first panel only; on the others they would run into the panel left.
        if panel == 0:
            ax.set_yticks(
                [c for c, _ in centres], [f for _, f in centres], fontsize=7, color=TEXT_PRIMARY
            )
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(name, color=TEXT_PRIMARY, fontsize=11)
        _style(ax)
    cbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("cosine similarity", color=TEXT_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    fig.suptitle(
        "Pairwise similarity of held-out chunks, grouped by filing", color=TEXT_PRIMARY, fontsize=12
    )
    fig.savefig(path, dpi=100, facecolor=SURFACE, bbox_inches="tight")
