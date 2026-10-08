"""Render four-panel Hasse (confidence) diagrams used in the paper.

Reads the Imax dominance matrices produced by sensitivity_estimation.py,
applies transitive reduction, assigns levels graphviz-style (each node sits
just above its highest child), and draws rounded-box nodes with thin arrows,
matching the look of Figure 8 and Appendix F.

Example:
    python render_hasse.py --indir results/sensitivity --mode rotated \
        --out figures/embedding_sensitivity.pdf
"""

import argparse
import logging
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUBS = ["anatomy", "clinical_knowledge", "college_biology",
        "medical_genetics"]
TITLES = ["(A) Anatomy", "(B) Clinical Knowledge", "(C) College Biology",
          "(D) Medical Genetics"]
# Index order of VNAMES in sensitivity_estimation.py -> display names.
DISPLAY = ["GPT-3.5", "Alpaca", "GPT-4o mini", "LLaMA-1", "LLaMA-2"]


def transitive_reduction(adj: np.ndarray) -> np.ndarray:
    """Remove edges implied by transitivity (adj is a DAG 0/1 matrix)."""
    n = adj.shape[0]
    closure = adj.astype(bool).copy()
    for k in range(n):
        for i in range(n):
            if closure[i, k]:
                closure[i] |= closure[k]
    red = adj.astype(bool).copy()
    for i in range(n):
        for j in range(n):
            if red[i, j]:
                for k in range(n):
                    if k not in (i, j) and closure[i, k] and closure[k, j]:
                        red[i, j] = False
                        break
    return red.astype(int)


def assign_levels(red: np.ndarray) -> np.ndarray:
    """Longest-path levels from the top, then pull parents down to sit just
    above their highest child (mimics graphviz tight ranking)."""
    n = red.shape[0]
    levels = np.zeros(n, dtype=int)
    for _ in range(n):
        for j in range(n):
            parents = np.where(red[:, j])[0]
            if parents.size:
                levels[j] = max(levels[j], levels[parents].max() + 1)
    for i in range(n):
        children = np.where(red[i])[0]
        if children.size and not np.where(red[:, i])[0].size:
            levels[i] = levels[children].min() - 1
    return levels


def draw_panel(ax: plt.Axes, red: np.ndarray, levels: np.ndarray,
               title: str, max_levels: int) -> None:
    n = red.shape[0]
    # Horizontal positions: spread nodes within each level, ordered by the
    # mean position of their parents (barycenter) for fewer crossings.
    xs = np.zeros(n)
    order_prev: Dict[int, float] = {}
    for lev in range(levels.min(), levels.max() + 1):
        nodes: List[int] = [v for v in range(n) if levels[v] == lev]
        nodes.sort(key=lambda v: (np.mean([order_prev.get(u, 0.5)
                                           for u in np.where(red[:, v])[0]])
                                  if np.where(red[:, v])[0].size else 0.5,
                                  v))
        for pos, v in enumerate(nodes):
            if len(nodes) == 1:
                xs[v] = 0.5
            else:
                xs[v] = 0.17 + 0.66 * pos / (len(nodes) - 1)
            order_prev[v] = xs[v]
    ys = 1 - (levels - levels.min() + 0.5) / max(max_levels, 1)

    box_w, box_h = 0.28, 0.10
    for v in range(n):
        ax.add_patch(FancyBboxPatch(
            (xs[v] - box_w / 2, ys[v] - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0.9, edgecolor="0.15", facecolor="white", zorder=3))
        ax.text(xs[v], ys[v], DISPLAY[v], ha="center", va="center",
                fontsize=9.5, zorder=4)

    for i in range(n):
        for j in range(n):
            if red[i, j]:
                span = levels[j] - levels[i]
                rad = 0.15 if span > 1 else (0.0 if abs(xs[i] - xs[j]) < 0.28
                                             else 0.08)
                ax.add_patch(FancyArrowPatch(
                    (xs[i], ys[i] - box_h / 2 - 0.012),
                    (xs[j], ys[j] + box_h / 2 + 0.012),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>", mutation_scale=9,
                    linewidth=0.8, color="0.15", zorder=2,
                    shrinkA=2, shrinkB=2))

    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", type=str, default="results/sensitivity")
    parser.add_argument("--mode", type=str, default="rotated",
                        choices=["pca", "rotated"])
    parser.add_argument("--h", type=float, default=0.3)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--out", type=str,
                        default="figures/embedding_sensitivity.pdf")
    args = parser.parse_args()

    reds, levels_all = [], []
    for sub in SUBS:
        Imax = np.load(os.path.join(
            args.indir, f"{sub}_{args.mode}_h{args.h}_dim{args.dim}.npz"))["Imax"]
        red = transitive_reduction(Imax)
        reds.append(red)
        levels_all.append(assign_levels(red))
    max_levels = max(lv.max() - lv.min() + 1 for lv in levels_all)

    plt.rcParams["font.family"] = "Helvetica"
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, red, lv, title in zip(axes, reds, levels_all, TITLES):
        draw_panel(ax, red, lv, title, max_levels)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.05,
                        wspace=0.15)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, format="pdf", dpi=300)
    fig.savefig(args.out.replace(".pdf", ".png"), format="png", dpi=150)
    logger.info("saved %s (+ .png preview)", args.out)


if __name__ == "__main__":
    main()
