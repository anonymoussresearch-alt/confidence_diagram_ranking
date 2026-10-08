"""Embedding-sensitivity experiment for the MMLU application (Appendix F of the paper).

Recomputes the model rankings and confidence-diagram dominance matrices under
(a) the original projection: per-subject PCA to 16 dimensions, replicating
    estimation.py, and
(b) an alternative projection matrix in the dimension-reduction step: a
    random orthonormal basis of the span of the top-32 principal components.

For each subject and projection, the script estimates the scores at the
held-out test prompts, runs the pairwise dominance tests (statistic T_ij with
the global GMB pairwise quantile, as in synthetic/plot2.ipynb), and
saves the 0/1 dominance matrix Imax consumed by plot3.R for the Hasse figure.

Example:
    python sensitivity_estimation.py --sub all --projection both
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from baseline_common import fit_theta_at, gmb_ratio_draws  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent / "data"
SUBS = ["anatomy", "clinical_knowledge", "college_biology", "medical_genetics"]
VNAMES = ["gpt3", "alpaca", "gpt4o", "llama1", "llama2"]


def _read_embeddings(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    col = df.iloc[:, 2]
    return np.array([np.fromstring(s.strip("[]"), sep=",") for s in col])


def load_subject(sub: str) -> Dict[str, np.ndarray]:
    train = _read_embeddings(DATA / "prompt" / f"embedded_questions_{sub}.csv")
    test = _read_embeddings(DATA / "prompt"
                            / f"embedded_questions_test_{sub}.csv")
    with open(DATA / "eval" / f"{sub}_num=100.json") as fh:
        out_data = json.load(fh)
    return {"train": train, "test": test, "eval": out_data}


def project(train: np.ndarray, test: np.ndarray, mode: str, dim: int,
            proj_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dimension reduction fitted on train+test jointly (as estimation.py)."""
    both = np.concatenate([train, test], axis=0)
    # svd_solver='full': the default randomized solver is nondeterministic
    # here and can flip borderline test edges between runs.
    if mode == "pca":
        red = PCA(n_components=dim, svd_solver="full").fit_transform(both)
    elif mode == "rotated":
        # Random orthonormal basis of the top-2*dim principal subspace: a
        # different projection matrix spanning a comparable representation.
        wide = PCA(n_components=2 * dim,
                   svd_solver="full").fit_transform(both)
        rng = np.random.RandomState(proj_seed)
        Q, _ = np.linalg.qr(rng.normal(size=(2 * dim, 2 * dim)))
        red = wide @ Q[:, :dim]
    else:
        raise ValueError(f"unknown projection mode: {mode}")
    return red[:train.shape[0]], red[train.shape[0]:]


def build_pairs(out_data: list, X_train: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair arrays (i<j) with Y = 1 meaning the larger-index model wins.

    Follows estimation.py: y[idx1, idx2] = 1 - GPT4_output, i.e. GPT4_output
    = 1 means variable_name1 wins the comparison.
    """
    name_to_index = {v: k for k, v in enumerate(VNAMES)}
    ii, jj, X, Y = [], [], [], []
    for entry in out_data:
        a = name_to_index[entry["variable_name1"]]
        b = name_to_index[entry["variable_name2"]]
        out = np.array(entry["GPT4_output"], dtype=float)
        i, j = min(a, b), max(a, b)
        y_j_wins = (1 - out) if j == b else out
        ii.append(i)
        jj.append(j)
        X.append(X_train[:len(out)])
        Y.append(y_j_wins)
    return (np.array(ii), np.array(jj), np.stack(X), np.stack(Y))


def run_subject(sub: str, mode: str, args: argparse.Namespace
                ) -> Dict[str, np.ndarray]:
    d = load_subject(sub)
    X_train, X_test = project(d["train"], d["test"], mode,
                              args.dim, args.proj_seed)
    pair_i, pair_j, Xp, Yp = build_pairs(d["eval"], X_train)
    n = len(VNAMES)

    # Score estimates at the unique training prompts (GMB plug-in) ...
    theta_train = np.stack([
        fit_theta_at(x, pair_i, pair_j, Xp, Yp, n, args.h, args.lambda_val)
        for x in X_train], axis=0)
    n_zero = int(np.sum(~theta_train.any(axis=1)))
    # ... and at the held-out test prompts (evaluation grid).
    estimates = np.stack([
        fit_theta_at(x, pair_i, pair_j, Xp, Yp, n, args.h, args.lambda_val)
        for x in X_test], axis=0)
    logger.info("[%s|%s] fitted %d train + %d test prompts "
                "(%d train fits had zero kernel mass)",
                sub, mode, X_train.shape[0], X_test.shape[0], n_zero)

    L = Xp.shape[1]
    d_hat_obs = np.stack([theta_train[:L, pair_j[p]]
                          - theta_train[:L, pair_i[p]]
                          for p in range(Xp.shape[0])])

    np.random.seed(args.seed)
    ratio = gmb_ratio_draws(pair_i, pair_j, Xp, Yp, d_hat_obs, X_test,
                            n, args.h, args.w_iter)

    # Pairwise dominance tests: one global quantile of the max over pairs
    # and prompts of (W_i - W_j), following plot2.ipynb. Under the "legacy"
    # convention the draws are additionally scaled by h^(d/2) as in
    # testing.py/plot2.ipynb, which reproduces the published pipeline; under
    # "paper" the unscaled theta-unit ratio is used (manuscript Def_W, where
    # the sqrt(h^d Xi) factors cancel).
    if args.convention == "legacy":
        ratio = (np.sqrt(args.h) ** X_train.shape[1]) * ratio
    TS = np.stack([(ratio[:, i, :] - ratio[:, j, :]).max(axis=0)
                   for i in range(n) for j in range(n) if i < j]).max(axis=0)
    q = float(np.quantile(TS, args.q_level))
    T = np.array([[np.min(estimates[:, i] - estimates[:, j])
                   for j in range(n)] for i in range(n)])
    Imax = (T > q).astype(int)
    return {"estimates": estimates, "T": T, "q": np.array(q), "Imax": Imax}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", type=str, default="all",
                        choices=SUBS + ["all"])
    parser.add_argument("--projection", type=str, default="both",
                        choices=["pca", "rotated", "both"])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--h", type=float, default=0.3)
    parser.add_argument("--lambda_val", type=float, default=1e-5)
    parser.add_argument("--w_iter", type=int, default=200)
    parser.add_argument("--q_level", type=float, default=0.95)
    parser.add_argument("--convention", type=str, default="legacy",
                        choices=["legacy", "paper"])
    parser.add_argument("--proj_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/sensitivity")
    args = parser.parse_args()
    logger.info("args: %s", args)

    os.makedirs(args.out, exist_ok=True)
    subs = SUBS if args.sub == "all" else [args.sub]
    modes = ["pca", "rotated"] if args.projection == "both" \
        else [args.projection]

    for sub in subs:
        for mode in modes:
            res = run_subject(sub, mode, args)
            tag = f"{sub}_{mode}_h{args.h}_dim{args.dim}"
            np.savez(os.path.join(args.out, f"{tag}.npz"), **res)
            np.savetxt(os.path.join(args.out, f"Imax_{tag}.txt"),
                       res["Imax"], fmt="%d")
            logger.info("[%s|%s] q = %.4f  Imax:\n%s",
                        sub, mode, float(res["q"]), res["Imax"])


if __name__ == "__main__":
    main()
