"""Shared components for the baseline-comparison experiment (Section 5.2).

Provides the synthetic DGPs of Section 5.1 (plus a new reversal design) and a
vectorized reimplementation of the proposed nonparametric kernel-BTL estimator
and Gaussian multiplier bootstrap from simu1.py / simu2.py. The math is
identical to the original scripts; only the inner loops are replaced by
precomputed kernel sums so that the 50-replication comparison is feasible
on a laptop.

Data layout follows simu1/simu2: Y_l is (n*n, L), time is (n*n, L, dim),
edge is (n, n) with rows n*i+j holding pair (i, j).
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

DGPFunc = Callable[[np.ndarray, int], np.ndarray]


# ---------------------------------------------------------------------------
# True score functions (theta = log BTL weight, centered across models)
# ---------------------------------------------------------------------------

def theta_setting1(t: np.ndarray, n: int, k: float = 0.01) -> np.ndarray:
    """Section 5.1 first design (simu1.py): theta_i(x) = exp(k * i * sum(x))."""
    theta = np.exp(k * np.sum(t) * np.arange(1, n + 1))
    return theta - np.mean(theta)


def theta_setting2(t: np.ndarray, n: int, k: float = 1.0,
                   scale: float = 0.02) -> np.ndarray:
    """Rescaled Section 5.1 second design: scale * (i * exp(k*sum(x)) + i).

    The scale keeps pairwise win probabilities away from 0/1 so that
    estimation error (rather than likelihood saturation) drives the
    comparison; the unscaled design (scale=1) is only identifiable up to
    ordering.
    """
    theta = np.exp(k * np.sum(t)) * np.arange(1, n + 1) + np.arange(1, n + 1)
    theta = scale * theta
    return theta - np.mean(theta)


def theta_setting3(t: np.ndarray, n: int, k: float = 1.0,
                   scale: float = 0.02, rev_model: int = 1,
                   amp: float = 0.4) -> np.ndarray:
    """Reversal design: setting 2 plus an oscillatory perturbation on one model.

    Model `rev_model` (0-indexed) receives an additive amp*sin(4*pi*s) term
    with s = sum(x)/3, so its score crosses those of nearby models several
    times inside the domain. The full oscillation completes two periods over
    the domain diagonal, so the sign pattern of the pairwise differences is
    not representable by linear or quadratic functions of x.
    """
    s = np.sum(t) / 3.0
    theta = np.exp(k * np.sum(t)) * np.arange(1, n + 1) + np.arange(1, n + 1)
    theta = scale * theta
    theta[rev_model] += amp * np.sin(4.0 * np.pi * s)
    return theta - np.mean(theta)


THETA_FUNCS: Dict[str, DGPFunc] = {
    "setting1": theta_setting1,
    "setting2": theta_setting2,
    "setting3": theta_setting3,
}


# ---------------------------------------------------------------------------
# Data generation (identical to simu1/simu2 generate_data, theta_fn injected)
# ---------------------------------------------------------------------------

def generate_data(L_ij: np.ndarray, p: float, n: int, dim: int,
                  theta_fn: DGPFunc) -> Dict[str, np.ndarray]:
    L = max(L_ij)
    edge = np.full((n, n), np.nan)
    Y_l = np.full((n * n, L), np.nan)
    time_points = np.full((n * n, L, dim), np.nan)

    def fill_pair(i: int, j: int) -> None:
        Lp = L_ij[n * i + j]
        time_points[n * i + j, :Lp] = np.random.uniform(0, 1, size=(Lp, dim))
        time_points[n * j + i, :Lp] = time_points[n * i + j, :Lp]
        theta = [theta_fn(t, n) for t in time_points[n * i + j, :Lp]]
        for kk in range(Lp):
            wj = np.exp(theta[kk][j])
            wi = np.exp(theta[kk][i])
            Y_l[n * i + j, kk] = np.random.binomial(1, wj / (wi + wj))
        Y_l[n * j + i, :Lp] = 1 - Y_l[n * i + j, :Lp]

    for i in range(n - 1):
        for j in range(i + 1, n):
            edge[i, j] = np.random.binomial(1, p)
            edge[j, i] = edge[i, j]
            if edge[i, j] == 1:
                fill_pair(i, j)

    for i in range(n - 1):
        if not np.any(edge[i, :] == 1):
            j = np.random.choice([x for x in range(n) if x != i])
            edge[i, j] = 1
            edge[j, i] = 1
            fill_pair(min(i, j), max(i, j))

    return {"edge": edge, "Y_l": Y_l, "time": time_points}


# ---------------------------------------------------------------------------
# Vectorized kernel-BTL estimator (math identical to simu2 L_t2/gradient2/
# f_theta_h2; kernel sums are precomputed per evaluation point)
# ---------------------------------------------------------------------------

def _pair_arrays(Y_l: np.ndarray, time_points: np.ndarray, edge: np.ndarray,
                 L_ij: np.ndarray, n: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-pair (i<j) observation arrays for edges present."""
    ii, jj, X, Y = [], [], [], []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if edge[i, j] == 1:
                Lp = L_ij[n * i + j]
                ii.append(i)
                jj.append(j)
                X.append(time_points[n * i + j, :Lp])
                Y.append(Y_l[n * i + j, :Lp])
    return (np.array(ii), np.array(jj),
            np.stack(X, axis=0), np.stack(Y, axis=0))


def _kernel_weights(X: np.ndarray, t: np.ndarray, h: float) -> np.ndarray:
    """Product Epanechnikov kernel, matching K() in simu1/simu2."""
    uh = (X - t) / h
    vals = 0.75 * (1 - uh ** 2) * (np.abs(uh) <= 1) / h
    return np.prod(vals, axis=-1)


def fit_theta_at(t: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray,
                 Xp: np.ndarray, Yp: np.ndarray, n: int, h: float,
                 lambda_val: float, gd: bool = False) -> np.ndarray:
    """Kernel-localized penalized BTL MLE at prompt t.

    Minimizes the objective of L_t2 in simu1/simu2 (verified numerically
    identical). By default the minimizer is L-BFGS run to convergence; the
    original fixed-step gradient descent (`gd=True`, 200 iterations)
    under-converges when the score amplitude is larger than ~1, which biases
    MSE and coverage although it preserves the ranking.
    """
    Kw = _kernel_weights(Xp, t, h)          # (P, L)
    S = Kw.sum(axis=1)                      # (P,)
    SY = (Kw * Yp).sum(axis=1)              # (P,)
    Z = S.sum()
    if Z == 0 or np.isnan(t).all():
        return np.zeros(n)

    def loss(theta: np.ndarray) -> float:
        d = theta[pair_j] - theta[pair_i]
        M = np.sum(-SY * d + S * np.logaddexp(0.0, d))
        return M / Z + lambda_val / 2 * np.sum(theta ** 2)

    def grad(theta: np.ndarray) -> np.ndarray:
        d = theta[pair_j] - theta[pair_i]
        g_pair = -SY + S / (1 + np.exp(-d))
        g = np.zeros(n)
        np.add.at(g, pair_j, g_pair)
        np.add.at(g, pair_i, -g_pair)
        return g / Z + lambda_val * theta

    if gd:
        theta_h = np.zeros(n)
        g1 = grad(theta_h)
        sum_iter = 0
        while np.sum(np.abs(g1)) > 1e-6 and sum_iter <= 200:
            m, reduce = 0.1, 0
            sum_iter += 1
            f_cur = loss(theta_h)
            while loss(theta_h - m * g1) > f_cur - 0.1 * m * np.dot(g1, g1):
                m = 0.1 * m
                reduce += 1
            theta_h = theta_h - m * g1
            g1 = grad(theta_h)
            if reduce >= 2:
                break
        return theta_h

    res = minimize(loss, np.zeros(n), jac=grad, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9})
    return res.x


# ---------------------------------------------------------------------------
# Gaussian multiplier bootstrap, implemented exactly as in Section 3 of the
# manuscript (equations Def_W, Def_G, Def_V and equ:confidence_band).
#
# In theta units the sqrt(h^d Xi) factors of W and of the band cancel, so the
# band half-width is the (1-alpha)-quantile of sup_{i,x} |G_i(x)/V_i(x)|,
# with G and V summed over ALL observed comparisons and a single multiplier
# xi_ij^l shared by both orientations of a comparison.
# ---------------------------------------------------------------------------

def gmb_ratio_draws(pair_i: np.ndarray, pair_j: np.ndarray, Xp: np.ndarray,
                    Yp: np.ndarray, d_hat_obs: np.ndarray, grid: np.ndarray,
                    n: int, h: float, w_iter: int) -> np.ndarray:
    """Bootstrap draws of G_i(x)/V_i(x) on the grid, shape (G, n, w_iter).

    d_hat_obs[p, l] holds theta_hat_{pair_j[p]} - theta_hat_{pair_i[p]}
    evaluated at the observed prompt X_p^l. Yp = 1 means pair_j wins, so the
    comparison residual for coordinate pair_j is -Y + psi(d) and the residual
    for pair_i is its negative.
    """
    G = grid.shape[0]
    psi = 1 / (1 + np.exp(-d_hat_obs))            # (P, L)
    psi_p = np.exp(d_hat_obs) / (1 + np.exp(d_hat_obs)) ** 2
    resid = -Yp + psi                              # residual for coordinate j
    xi = np.random.normal(size=(Xp.shape[0], Xp.shape[1], w_iter))

    Gm = np.zeros((n, w_iter, G))
    Vm = np.zeros((n, G))
    for p in range(Xp.shape[0]):
        Kmat = _kernel_weights(Xp[p][:, None, :], grid[None, :, :], h)  # (L,G)
        Vp = Kmat.T @ psi_p[p]                     # (G,)
        Vm[pair_i[p]] += Vp
        Vm[pair_j[p]] += Vp
        Gp = (xi[p] * resid[p][:, None]).T @ Kmat  # (w_iter, G)
        Gm[pair_j[p]] += Gp
        Gm[pair_i[p]] -= Gp
    ratio = np.divide(Gm, Vm[:, None, :], where=Vm[:, None, :] != 0,
                      out=np.zeros_like(Gm))
    return np.transpose(ratio, (2, 0, 1))          # (G, n, w_iter)


def band_halfwidth(ratio: np.ndarray, alpha: float) -> float:
    """(1-alpha)-quantile of sup_{x,i} |G/V|: half-width of the theta band."""
    TS = np.abs(ratio).max(axis=(0, 1))
    return float(np.quantile(TS, 1 - alpha))


def pairwise_halfwidths(ratio: np.ndarray, alpha: float) -> np.ndarray:
    """Per-pair (1-alpha)-quantiles of sup_x |W_i(x) - W_j(x)| in theta units.

    Mirrors the pairwise statistic W_ij (equ:Wij) applied two-sidedly;
    returns a symmetric (n, n) matrix of critical values.
    """
    G, n, w = ratio.shape
    C = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            TS = np.abs(ratio[:, i, :] - ratio[:, j, :]).max(axis=0)
            C[i, j] = C[j, i] = np.quantile(TS, 1 - alpha)
    return C


def generate_grid(begin: float, end: float, dim: int,
                  time_num: int) -> np.ndarray:
    points_per_dim = int(round(time_num ** (1 / dim)))
    grid1d = np.linspace(begin, end, points_per_dim)
    mesh = np.meshgrid(*([grid1d] * dim))
    grid_points = np.vstack([g.ravel() for g in mesh]).T
    if len(grid_points) > time_num:
        grid_points = grid_points[:time_num]
    return grid_points


# ---------------------------------------------------------------------------
# One full replication of the proposed method (mirrors simu2 __main__)
# ---------------------------------------------------------------------------

def run_proposed(data: Dict[str, np.ndarray], grid: np.ndarray, n: int,
                 L_val: int, h: float, lambda_val: float, w_iter: int,
                 alpha: float,
                 rng_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Estimate theta on `grid` with GMB simultaneous and pairwise bands.

    Returns centered grid estimates, the sup-band half-width q_sup (theta
    units) and the symmetric matrix C of per-pair critical values.
    """
    edge, Y_l, time0 = data["edge"], data["Y_l"], data["time"]
    L_ij = np.full((n * n,), L_val)
    pair_i, pair_j, Xp, Yp = _pair_arrays(Y_l, time0, edge, L_ij, n)

    if rng_seed is not None:
        np.random.seed(rng_seed)

    # theta_hat at every observed prompt of every pair (plug-in for G and V).
    P, L = Xp.shape[0], Xp.shape[1]
    d_hat_obs = np.zeros((P, L))
    for p in range(P):
        for ell in range(L):
            th = fit_theta_at(Xp[p, ell], pair_i, pair_j, Xp, Yp,
                              n, h, lambda_val)
            d_hat_obs[p, ell] = th[pair_j[p]] - th[pair_i[p]]

    G = grid.shape[0]
    estimates = np.zeros((G, n))
    for g in range(G):
        estimates[g] = fit_theta_at(grid[g], pair_i, pair_j, Xp, Yp,
                                    n, h, lambda_val)

    ratio = gmb_ratio_draws(pair_i, pair_j, Xp, Yp, d_hat_obs, grid,
                            n, h, w_iter)
    return {"estimates": estimates,
            "q_sup": band_halfwidth(ratio, alpha),
            "C_pair": pairwise_halfwidths(ratio, alpha)}
