"""Driver for the Section 5.2 baseline comparison (Table 1).

Runs the proposed nonparametric estimator against linear and polynomial
contextual BTL baselines on the Section 5.1 synthetic designs plus a new
contextual-reversal design, over repeated replications. Reports estimation
MSE, empirical simultaneous coverage of the 90% confidence bands, ranking
accuracy, and contextual-reversal detection.

Example:
    python run_baseline_comparison.py --setting setting3 --reps 50 --cpu 8
"""

import argparse
import functools
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List

import numpy as np

from baseline_common import (THETA_FUNCS, generate_data, generate_grid,
                             run_proposed)
from baseline_parametric import evaluate_with_bands, fit_contextual_btl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METHODS = ("linear", "poly2", "proposed")


def make_theta_fn(setting: str, scale: float, amp: float) -> Callable:
    """Configured score function; scale/amp apply to settings 2 and 3 only."""
    if setting == "setting2":
        return functools.partial(THETA_FUNCS[setting], scale=scale)
    if setting == "setting3":
        return functools.partial(THETA_FUNCS[setting], scale=scale, amp=amp)
    return THETA_FUNCS[setting]


def true_theta_on_grid(fn: Callable, grid: np.ndarray, n: int) -> np.ndarray:
    return np.stack([fn(x, n) for x in grid], axis=0)  # (G, n)


def rank_accuracy(est: np.ndarray, truth: np.ndarray) -> float:
    """Mean fraction of correctly ordered model pairs across grid points."""
    G, n = est.shape
    iu = np.triu_indices(n, k=1)
    d_est = est[:, :, None] - est[:, None, :]
    d_true = truth[:, :, None] - truth[:, None, :]
    concordant = (np.sign(d_est[:, iu[0], iu[1]])
                  == np.sign(d_true[:, iu[0], iu[1]]))
    return float(np.mean(concordant))


def reversal_metrics(est: np.ndarray, truth: np.ndarray, C_pair: np.ndarray,
                     margin: float = 0.25) -> Dict[str, float]:
    """Detection rate over true reversal pairs and false-positive rate.

    A pair (i, j) is a true reversal if theta*_i - theta*_j reaches at least
    +margin and -margin on the grid; a method detects it if the estimated
    difference exceeds the pair's simultaneous critical value C_pair[i, j] in
    both directions somewhere on the grid. Pairs whose true difference never
    changes sign are the null set for the false-positive rate; pairs with a
    sub-margin sign change are excluded from both sets.
    """
    G, n = truth.shape
    detected, n_rev, false_pos, n_null = 0, 0, 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d_true = truth[:, i] - truth[:, j]
            d_est = est[:, i] - est[:, j]
            is_rev = d_true.max() > margin and d_true.min() < -margin
            is_null = d_true.max() <= 0 or d_true.min() >= 0
            sig_rev = (d_est.max() > C_pair[i, j]
                       and d_est.min() < -C_pair[i, j])
            if is_rev:
                n_rev += 1
                detected += int(sig_rev)
            elif is_null:
                n_null += 1
                false_pos += int(sig_rev)
    return {"rev_detect": detected / n_rev if n_rev else np.nan,
            "rev_false": false_pos / n_null if n_null else np.nan,
            "n_rev_pairs": float(n_rev)}


def one_replication(rep: int, setting: str, n: int, L_val: int, p: float,
                    dim: int, h: float, lambda_val: float, w_iter: int,
                    alpha: float, grid_num: int, grid_lo: float,
                    grid_hi: float, scale: float, amp: float, base_seed: int
                    ) -> Dict[str, Dict[str, float]]:
    seed = base_seed + rep
    np.random.seed(seed)
    theta_fn = make_theta_fn(setting, scale, amp)
    L_ij = np.full((n * n,), L_val)
    data = generate_data(L_ij, p, n, dim, theta_fn)
    # Metrics are evaluated on an interior grid (one bandwidth away from the
    # domain boundary), following the truncation convention of
    # Ranking_Codes/testing.py coverage_mat.
    grid = generate_grid(grid_lo, grid_hi, dim, grid_num)
    truth = true_theta_on_grid(theta_fn, grid, n)

    out: Dict[str, Dict[str, float]] = {}

    for basis in ("linear", "poly2"):
        fit = fit_contextual_btl(data, n, L_val, basis, lambda_val)
        ev = evaluate_with_bands(fit, grid, basis, alpha, n, w_iter)
        est, q_sup = ev["estimates"], ev["q_sup"]
        res = {"mse": float(np.mean((est - truth) ** 2)),
               "cover": float(np.all(np.abs(est - truth) <= q_sup)),
               "rank_acc": rank_accuracy(est, truth),
               "q_sup": q_sup}
        res.update(reversal_metrics(est, truth, ev["C_pair"]))
        out[basis] = res

    prop = run_proposed(data, grid, n, L_val, h, lambda_val, w_iter, alpha,
                        rng_seed=seed + 10_000)
    est, q_sup = prop["estimates"], prop["q_sup"]
    res = {"mse": float(np.mean((est - truth) ** 2)),
           "cover": float(np.all(np.abs(est - truth) <= q_sup)),
           "rank_acc": rank_accuracy(est, truth),
           "q_sup": q_sup}
    res.update(reversal_metrics(est, truth, prop["C_pair"]))
    out["proposed"] = res
    return out


def summarize(results: List[Dict[str, Dict[str, float]]],
              setting: str) -> str:
    lines = [f"% Setting: {setting}, reps = {len(results)}"]
    names = {"linear": "Linear contextual BTL",
             "poly2": "Polynomial contextual BTL",
             "proposed": "Proposed (nonparametric)"}
    header = (f"{'method':<28}{'MSE':>12}{'coverage':>10}"
              f"{'rank acc':>10}{'rev detect':>12}{'rev false':>11}")
    print(header)
    for meth in METHODS:
        vals = {k: np.array([r[meth][k] for r in results])
                for k in results[0][meth]}
        mse, mse_sd = vals["mse"].mean(), vals["mse"].std()
        cov = vals["cover"].mean()
        acc = vals["rank_acc"].mean()
        det = np.nanmean(vals["rev_detect"])
        fp = np.nanmean(vals["rev_false"])
        print(f"{names[meth]:<28}{mse:>12.4f}{cov:>10.2f}"
              f"{acc:>10.3f}{det:>12.3f}{fp:>11.3f}")
        det_s = "--" if np.isnan(det) else f"{det:.2f}"
        lines.append(f"{names[meth]} & {mse:.3f} ({mse_sd:.3f}) & "
                     f"{cov:.2f} & {acc:.3f} & {det_s} \\\\")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", type=str, default="setting3",
                        choices=list(THETA_FUNCS) + ["all"])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--L_val", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--h", type=float, default=0.3)
    parser.add_argument("--lambda_val", type=float, default=1e-5)
    parser.add_argument("--w_iter", type=int, default=100)
    parser.add_argument("--pp", type=float, default=400)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--grid_num", type=int, default=125)
    parser.add_argument("--grid_lo", type=float, default=0.3)
    parser.add_argument("--grid_hi", type=float, default=0.7)
    parser.add_argument("--scale", type=float, default=0.02,
                        help="Score scale for settings 2 and 3")
    parser.add_argument("--amp", type=float, default=0.4,
                        help="Reversal amplitude for setting 3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--out", type=str, default="out_baselines")
    args = parser.parse_args()
    logger.info("args: %s", args)

    os.makedirs(args.out, exist_ok=True)
    settings = list(THETA_FUNCS) if args.setting == "all" else [args.setting]

    for setting in settings:
        keys = dict(setting=setting, n=args.n, L_val=args.L_val, p=args.p,
                    dim=args.dim, h=args.h, lambda_val=args.lambda_val,
                    w_iter=args.w_iter, alpha=args.alpha,
                    grid_num=args.grid_num, grid_lo=args.grid_lo,
                    grid_hi=args.grid_hi, scale=args.scale, amp=args.amp,
                    base_seed=args.seed)
        results: List[Dict[str, Dict[str, float]]] = [None] * args.reps
        with ProcessPoolExecutor(max_workers=args.cpu) as ex:
            futures = {ex.submit(one_replication, r, **keys): r
                       for r in range(args.reps)}
            for fut in as_completed(futures):
                r = futures[fut]
                results[r] = fut.result()
                logger.info("[%s] replication %d/%d done",
                            setting, sum(x is not None for x in results),
                            args.reps)
        latex = summarize(results, setting)
        tag = (f"{setting}_n{args.n}_L{args.L_val}_p{args.p}_h{args.h}"
               f"_grid{args.grid_num}_reps{args.reps}_seed{args.seed}")
        np.savez(os.path.join(args.out, f"{tag}.npz"),
                 results=np.array(results, dtype=object), args=np.array(keys, dtype=object))
        with open(os.path.join(args.out, f"{tag}_table.tex"), "w") as fh:
            fh.write(latex + "\n")
        logger.info("saved %s", tag)


if __name__ == "__main__":
    main()
