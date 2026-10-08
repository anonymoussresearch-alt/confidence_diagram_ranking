"""Parametric contextual BTL baselines for the Section 5.2 comparison.

Fits theta_i(x) = beta_i^T phi(x) by (ridge-stabilized) maximum likelihood,
where phi is either the linear basis [1, x] or the full degree-2 polynomial
basis. Simultaneous confidence bands are obtained from a Gaussian multiplier
draw on the asymptotic MLE covariance, mirroring the sup-statistic convention
used for the proposed method (see synthetic/plot2.ipynb).
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def phi_linear(X: np.ndarray) -> np.ndarray:
    """Basis [1, x1, ..., xd] for points X of shape (N, d)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def phi_poly2(X: np.ndarray) -> np.ndarray:
    """Full degree-2 polynomial basis (intercept, linear, squares, cross)."""
    N, d = X.shape
    cols = [np.ones((N, 1)), X]
    for a in range(d):
        for b in range(a, d):
            cols.append((X[:, a] * X[:, b])[:, None])
    return np.hstack(cols)


BASES = {"linear": phi_linear, "poly2": phi_poly2}


def _flatten_comparisons(Y_l: np.ndarray, time_points: np.ndarray,
                         edge: np.ndarray, L_ij: np.ndarray, n: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx_i, idx_j, X, Y) over all comparisons of pairs i<j.

    Y = 1 indicates that model j (the larger index) wins, matching the
    Y_l[n*i+j] convention of generate_data.
    """
    ii, jj, X, Y = [], [], [], []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if edge[i, j] == 1:
                Lp = L_ij[n * i + j]
                ii.append(np.full(Lp, i))
                jj.append(np.full(Lp, j))
                X.append(time_points[n * i + j, :Lp])
                Y.append(Y_l[n * i + j, :Lp])
    return (np.concatenate(ii), np.concatenate(jj),
            np.vstack(X), np.concatenate(Y))


def fit_contextual_btl(data: Dict[str, np.ndarray], n: int, L_val: int,
                       basis: str, lambda_val: float = 1e-5
                       ) -> Dict[str, np.ndarray]:
    """MLE of B (n x q) with theta_i(x) = B[i] @ phi(x); returns B and Cov."""
    L_ij = np.full((n * n,), L_val)
    idx_i, idx_j, Xc, Yc = _flatten_comparisons(
        data["Y_l"], data["time"], data["edge"], L_ij, n)
    Phi = BASES[basis](Xc)                       # (N, q)
    N, q = Phi.shape

    def unpack(beta: np.ndarray) -> np.ndarray:
        return beta.reshape(n, q)

    def nll(beta: np.ndarray) -> float:
        B = unpack(beta)
        d = np.einsum("nq,nq->n", Phi, B[idx_j] - B[idx_i])
        val = np.sum(-Yc * d + np.logaddexp(0.0, d)) / N
        return val + lambda_val / 2 * np.sum(beta ** 2)

    def grad(beta: np.ndarray) -> np.ndarray:
        B = unpack(beta)
        d = np.einsum("nq,nq->n", Phi, B[idx_j] - B[idx_i])
        r = (-Yc + 1 / (1 + np.exp(-d)))[:, None] * Phi  # (N, q)
        G = np.zeros((n, q))
        np.add.at(G, idx_j, r)
        np.add.at(G, idx_i, -r)
        return G.ravel() / N + lambda_val * beta

    res = minimize(nll, np.zeros(n * q), jac=grad, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
    if not res.success:
        logger.warning("contextual BTL (%s) did not fully converge: %s",
                       basis, res.message)
    B = unpack(res.x)

    # Observed information of the *summed* penalized likelihood.
    d = np.einsum("nq,nq->n", Phi, B[idx_j] - B[idx_i])
    w = np.exp(d) / (1 + np.exp(d)) ** 2         # sigma'(d), (N,)
    H = np.zeros((n * q, n * q))
    # z = (e_j - e_i) kron phi; accumulate w * z z^T blockwise.
    WPhi = w[:, None] * Phi
    for a in range(n):
        sel_jj = idx_j == a
        sel_ii = idx_i == a
        for b in range(a, n):
            blk = np.zeros((q, q))
            if a == b:
                sel = sel_jj | sel_ii
                blk = WPhi[sel].T @ Phi[sel]
            else:
                sel = (sel_ii & (idx_j == b)) | (sel_jj & (idx_i == b))
                if np.any(sel):
                    blk = -(WPhi[sel].T @ Phi[sel])
            H[a * q:(a + 1) * q, b * q:(b + 1) * q] = blk
            if a != b:
                H[b * q:(b + 1) * q, a * q:(a + 1) * q] = blk.T
    H += N * lambda_val * np.eye(n * q)
    cov = np.linalg.inv(H)
    return {"B": B, "cov": cov, "q": np.array(q)}


def evaluate_with_bands(fit: Dict[str, np.ndarray], grid: np.ndarray,
                        basis: str, alpha: float, n: int,
                        w_iter: int = 100) -> Dict[str, np.ndarray]:
    """Centered estimates on the grid plus sup and pairwise band quantiles.

    Multiplier draws eta ~ N(0, Cov) are mapped to centered score
    perturbations; quantile conventions follow Ranking_Codes/testing.py.
    """
    B, cov, q = fit["B"], fit["cov"], int(fit["q"])
    Phi = BASES[basis](grid)                      # (G, q)
    theta = Phi @ B.T                             # (G, n)
    theta = theta - theta.mean(axis=1, keepdims=True)

    jitter = 1e-10 * np.eye(cov.shape[0])
    Lc = np.linalg.cholesky(cov + jitter)
    eta = Lc @ np.random.normal(size=(cov.shape[0], w_iter))  # (nq, S)
    # Perturbations of theta on the grid, centered across models.
    pert = np.einsum("gq,nqs->gns", Phi, eta.reshape(n, q, w_iter))
    pert = pert - pert.mean(axis=1, keepdims=True)            # (G, n, S)

    q_sup = float(np.quantile(np.abs(pert).max(axis=(0, 1)), 1 - alpha))
    C = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            TS = np.abs(pert[:, i, :] - pert[:, j, :]).max(axis=0)
            C[i, j] = C[j, i] = np.quantile(TS, 1 - alpha)
    return {"estimates": theta, "q_sup": q_sup, "C_pair": C}
