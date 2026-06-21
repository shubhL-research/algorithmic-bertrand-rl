"""
Algorithmic Bertrand competition: learning dynamics, with corrected metrics.

Implements REAL Q-learning
    Q(s,a) <- Q(s,a) + alpha * [ pi + gamma * max_a' Q(s',a') - Q(s,a) ]
(state s = opponent's previous-period price) and a genuinely stateless
mean-based (bandit) benchmark.

This version adds, relative to the first draft:
  * a corrected collusion index Delta anchored on the true DISCRETE
    Bertrand-Nash profit (not zero);
  * non-circular convergence diagnostics (greedy-policy stability +
    state-action visitation), so "converged" is NOT inferred from low entropy;
  * matched (annealed/constant) exploration for BOTH learners;
  * K-normalised Shannon entropy with optional Miller-Madow bias correction.

Author: Shubh Lamba. Seeded and reproducible.
"""
import numpy as np


# ----------------------------------------------------------------------
# Market primitives
# ----------------------------------------------------------------------
def make_grid(K, p_min=0.0, p_max=10.0):
    return np.linspace(p_min, p_max, K)


def step_profits(action_idx, grid, c):
    """Bertrand allocation: lowest price serves 1 unit of demand, ties split."""
    prices = grid[action_idx]
    pmin = prices.min()
    winners = prices == pmin
    q = np.where(winners, 1.0 / winners.sum(), 0.0)
    return (prices - c) * q


def nash_profit(grid, c, n):
    """Per-firm profit at the DISCRETE competitive (Bertrand-Nash) benchmark.

    On a discrete grid the static competitive equilibrium has both firms at the
    lowest grid price weakly above marginal cost; that price earns a small but
    POSITIVE per-firm profit (split). Returns 0 only if a grid point equals c.
    """
    above = grid[grid >= c]
    p_comp = above.min() if above.size else grid.max()
    return (p_comp - c) / n


def monopoly_profit(grid, c, n):
    """Per-firm profit at symmetric joint monopoly (both at the top price, split)."""
    return (grid.max() - c) / n


def _randargmax(row, rng):
    m = row.max()
    cands = np.flatnonzero(row == m)
    return cands[rng.integers(len(cands))] if cands.size > 1 else cands[0]


# ----------------------------------------------------------------------
# Learning agents (n=2 specialised; both support convergence tracking)
# ----------------------------------------------------------------------
def run_qlearning(K, T, alpha, gamma, epsilon, c, grid, rng,
                  eps_decay=None, track_conv=False, return_Q=False):
    """Two-firm epsilon-greedy Q-learning. State = opponent's previous price."""
    Q = [rng.normal(0.0, 1e-6, size=(K, K)) for _ in range(2)]
    prev = rng.integers(0, K, size=2)
    s = [int(prev[1]), int(prev[0])]            # agent i's state = opponent's prev price
    price_hist = np.empty((T, 2), dtype=np.int64)
    profit_hist = np.empty((T, 2), dtype=np.float64)
    rand, randint = rng.random, rng.integers

    if track_conv:
        visits = [np.zeros((K, K), dtype=np.int64) for _ in range(2)]
        t_snap = int(0.9 * T)
        pol_snap = [None, None]
        pol_stable = [0.0, 0.0]

    for t in range(T):
        e = epsilon if eps_decay is None else epsilon / (1.0 + eps_decay * t)
        a0 = randint(K) if rand() < e else int(np.argmax(Q[0][s[0]]))
        a1 = randint(K) if rand() < e else int(np.argmax(Q[1][s[1]]))
        pr = step_profits(np.array((a0, a1)), grid, c)
        ns = [a1, a0]
        Q[0][s[0], a0] += alpha * (pr[0] + gamma * Q[0][ns[0]].max() - Q[0][s[0], a0])
        Q[1][s[1], a1] += alpha * (pr[1] + gamma * Q[1][ns[1]].max() - Q[1][s[1], a1])
        price_hist[t, 0] = a0; price_hist[t, 1] = a1
        profit_hist[t] = pr

        if track_conv:
            visits[0][s[0], a0] += 1; visits[1][s[1], a1] += 1
            if t == t_snap:
                pol_snap[0] = Q[0].argmax(axis=1).copy()
                pol_snap[1] = Q[1].argmax(axis=1).copy()
        s = ns

    out = {"price": price_hist, "profit": profit_hist}
    if track_conv:
        for i in range(2):
            seen = visits[i].sum(axis=1) > 0           # states visited at all
            final_pol = Q[i].argmax(axis=1)
            denom = seen.sum()
            pol_stable[i] = float((pol_snap[i][seen] == final_pol[seen]).mean()) if denom else 1.0
        out["pol_stable"] = pol_stable                  # frac of visited states whose greedy action
        out["min_visit"] = [int(visits[i][visits[i] > 0].min()) if (visits[i] > 0).any() else 0
                            for i in range(2)]           #   was unchanged over the final 10% of T
        out["cells_visited"] = [int((visits[i] > 0).sum()) for i in range(2)]
        out["total_cells"] = K * K
    if return_Q:
        out["Q"] = Q
    return out


def run_meanbased(K, T, epsilon, c, grid, rng, eps_decay=None, track_conv=False):
    """Two-firm stateless mean-based learning (bandit). Supports matched exploration."""
    sums = [np.zeros(K) for _ in range(2)]
    counts = [np.zeros(K) for _ in range(2)]
    val = [np.zeros(K) for _ in range(2)]
    price_hist = np.empty((T, 2), dtype=np.int64)
    profit_hist = np.empty((T, 2), dtype=np.float64)
    rand, randint = rng.random, rng.integers

    if track_conv:
        t_snap = int(0.9 * T)
        pol_snap = [None, None]

    for t in range(T):
        e = epsilon if eps_decay is None else epsilon / (1.0 + eps_decay * t)
        a = (randint(K) if rand() < e else _randargmax(val[0], rng),
             randint(K) if rand() < e else _randargmax(val[1], rng))
        pr = step_profits(np.array(a), grid, c)
        for i in range(2):
            ai = a[i]; counts[i][ai] += 1; sums[i][ai] += pr[i]
            val[i][ai] = sums[i][ai] / counts[i][ai]
        price_hist[t, 0] = a[0]; price_hist[t, 1] = a[1]
        profit_hist[t] = pr
        if track_conv and t == t_snap:
            pol_snap[0] = int(np.argmax(val[0])); pol_snap[1] = int(np.argmax(val[1]))

    out = {"price": price_hist, "profit": profit_hist, "val": val}
    if track_conv:
        out["pol_stable"] = [float(pol_snap[i] == int(np.argmax(val[i]))) for i in range(2)]
        out["min_visit"] = [int(counts[i][counts[i] > 0].min()) if (counts[i] > 0).any() else 0
                            for i in range(2)]
        out["cells_visited"] = [int((counts[i] > 0).sum()) for i in range(2)]
        out["total_cells"] = K
    return out


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def entropy_bits(col, K, miller_madow=True):
    """Shannon entropy (bits) of an action sequence, with optional Miller-Madow
    bias correction. Returns (raw_bits, K_normalised in [0,1])."""
    counts = np.bincount(col, minlength=K).astype(float)
    nobs = counts.sum()
    f = counts[counts > 0] / nobs
    H = float(-(f * np.log2(f)).sum())
    if miller_madow:
        H += (np.count_nonzero(counts) - 1) / (2.0 * nobs) / np.log(2)  # bias correction (bits)
    return H, H / np.log2(K)


def mean_entropy(price_hist, K, T0, miller_madow=True):
    w = price_hist[-T0:]
    vals = [entropy_bits(w[:, i], K, miller_madow) for i in range(w.shape[1])]
    Hbar = float(np.mean([v[0] for v in vals]))
    Hnorm = float(np.mean([v[1] for v in vals]))
    return Hbar, Hnorm


def is_converged(res, tol=0.99):
    """Non-circular convergence: greedy policy unchanged (>= tol of visited states)
    over the final 10% of the run, for BOTH firms. Requires track_conv=True."""
    ps = res.get("pol_stable")
    return bool(ps) and min(ps) >= tol


def delta_index(price_hist, profit_hist, grid, c, n, T0):
    """Collusion index normalised between the DISCRETE Bertrand-Nash (0) and
    symmetric joint monopoly (1). pi is mean per-firm profit over the window."""
    pi = float(profit_hist[-T0:].mean())
    piN = nash_profit(grid, c, n)
    piM = monopoly_profit(grid, c, n)
    return (pi - piN) / (piM - piN)


def regime(Hnorm, p_bar, c, H_star, p_star_factor=2.0):
    if Hnorm >= H_star:
        return "Chaotic"
    return "Competitive" if p_bar < p_star_factor * c else "Collusive"
