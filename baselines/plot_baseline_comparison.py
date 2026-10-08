"""Figure for the Section 5.2 baseline comparison.

Draws the estimated score functions and simultaneous confidence bands of the
two parametric contextual BTL baselines and of the proposed nonparametric
estimator along a one-dimensional slice of the prompt domain, under the
reversal design (setting 3). The slice varies x_1 with x_2 and x_3 fixed at
the center of the evaluation window, so a contextual reversal is visible as a
crossing of two score curves.

Example:
    python plot_baseline_comparison.py --models 1 2 --out figures/baseline_comparison.pdf
"""

import argparse
import logging
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline_common import generate_data, run_proposed
from baseline_parametric import evaluate_with_bands, fit_contextual_btl
from run_baseline_comparison import make_theta_fn, true_theta_on_grid

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TITLES = ["(A) Linear contextual BTL", "(B) Polynomial contextual BTL",
          "(C) Proposed (nonparametric)"]
# Paper palette (cf. figure/coverage2.pdf): ground truth blue, bands orange
# and green; a second model is drawn in a muted red for the reversal.
COLORS = ["tab:blue", "tab:red"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", type=str, default="setting3")
    parser.add_argument("--models", nargs="+", type=int, default=[1, 2],
                        help="0-indexed models to draw (default: the "
                             "reversing model and its neighbour)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--L_val", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--h", type=float, default=0.25)
    parser.add_argument("--lambda_val", type=float, default=1e-5)
    parser.add_argument("--w_iter", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--amp", type=float, default=2.0)
    parser.add_argument("--slice_num", type=int, default=25)
    parser.add_argument("--slice_lo", type=float, default=0.3)
    parser.add_argument("--slice_hi", type=float, default=0.7)
    parser.add_argument("--fixed", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="figures/baseline_comparison.pdf")
    args = parser.parse_args()
    logger.info("args: %s", args)

    np.random.seed(args.seed)
    theta_fn = make_theta_fn(args.setting, args.scale, args.amp)
    L_ij = np.full((args.n * args.n,), args.L_val)
    data = generate_data(L_ij, args.p, args.n, args.dim, theta_fn)

    x1 = np.linspace(args.slice_lo, args.slice_hi, args.slice_num)
    grid = np.column_stack([x1, np.full_like(x1, args.fixed),
                            np.full_like(x1, args.fixed)])
    truth = true_theta_on_grid(theta_fn, grid, args.n)

    panels = []
    for basis in ("linear", "poly2"):
        fit = fit_contextual_btl(data, args.n, args.L_val, basis,
                                 args.lambda_val)
        ev = evaluate_with_bands(fit, grid, basis, args.alpha, args.n,
                                 args.w_iter)
        panels.append((ev["estimates"], ev["q_sup"]))
    prop = run_proposed(data, grid, args.n, args.L_val, args.h,
                        args.lambda_val, args.w_iter, args.alpha,
                        rng_seed=args.seed + 10_000)
    panels.append((prop["estimates"], prop["q_sup"]))

    plt.rcParams["font.family"] = "Helvetica"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (est, q), title in zip(axes, panels, TITLES):
        for c, m in zip(COLORS, args.models):
            ax.plot(x1, truth[:, m], linestyle="--", color=c, linewidth=1.8,
                    label=f"Model {m + 1}, ground truth")
            ax.plot(x1, est[:, m], linestyle="-", color=c, linewidth=1.8,
                    label=f"Model {m + 1}, estimate and band")
            ax.fill_between(x1, est[:, m] - q, est[:, m] + q, color=c,
                            alpha=0.18, linewidth=0)
        ax.set_title(title, fontsize=15, pad=8)
        ax.set_xlabel(r"$x_1$", fontsize=15)
        ax.set_xticks(np.linspace(args.slice_lo, args.slice_hi, 5))
        ax.tick_params(labelsize=12.5)
    axes[0].set_ylabel(r"$\theta_i(\mathbf{x})$", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=12.5,
               frameon=False, ncol=4, columnspacing=1.6,
               handletextpad=0.6, bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.28,
                        wspace=0.12)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, format="pdf", dpi=300)
    fig.savefig(args.out.replace(".pdf", ".png"), format="png", dpi=150)
    logger.info("saved %s (+ .png preview)", args.out)


if __name__ == "__main__":
    main()
