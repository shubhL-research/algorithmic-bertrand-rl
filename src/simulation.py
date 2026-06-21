"""
Learning Dynamics in Algorithmic Bertrand Competition
=====================================================
Correct, reproducible implementation of the model described in the paper.

Key point vs. the old code: this implements REAL Q-learning ---
  Q(s,a) <- Q(s,a) + alpha * [ pi + gamma * max_a' Q(s',a') - Q(s,a) ]
with state s = opponent(s)' previous-period price, and a genuinely
stateless mean-based (bandit) benchmark, so the two are actually different.

Baseline (matches the paper text):
  n=2 firms, homogeneous good, marginal cost c=1, demand = 1 unit (inelastic),
  price grid [0,10] with K=41 points (Delta=0.25),
  alpha=0.10, gamma=0.95, epsilon=0.10, T=5000, classification window T0=1000.

Author: Shubh Lamba.  Everything here is seeded and reproducible.
"""
import numpy as np


# ----------------------------------------------------------------------
# Market primitives
# ----------------------------------------------------------------------
def make_grid(K=41, p_min=0.0, p_max=10.0):
    return np.linspace(p_min, p_max, K)


def step_profits(action_idx, grid, c):
    """Bertrand allocation: lowest price serves 1 unit of demand, ties split."""
    prices = grid[action_idx]
    pmin = prices.min()
    winners = prices == pmin
    q = np.where(winners, 1.0 / winners.sum(), 0.0)   # total demand normalised to 1
    return (prices - c) * q


def _randargmax(row, rng):
    """argmax with random tie-breaking (faithful to zero-init without locking on index 0)."""
    m = row.max()
    cands = np.flatnonzero(row == m)
    return cands[rng.integers(len(cands))] if cands.size > 1 else cands[0]


def _state_index(opp_idx, K):
    s = 0
    for x in opp_idx:
        s = s * K + int(x)
    return s


# ----------------------------------------------------------------------
# Learning agents
# ----------------------------------------------------------------------
def run_qlearning(n, K, T, alpha, gamma, epsilon, c, grid, rng,
                  eps_decay=None, memory="opponent", return_Q=False):
    """Stateful epsilon-greedy Q-learning.

    memory="opponent": state = opponents' previous prices (faithful to the paper text).
    memory="joint":    state = ALL firms' previous prices, incl. own (Calvano-style);
                       lets an agent punish relative to its own last price.

    Q-tables are initialised with negligible noise (~1e-6) purely to break argmax ties,
    so np.argmax can be used (fast) while remaining effectively a zero start.
    """
    if memory == "joint":
        nS = K ** n
        idx_self = True
    else:
        nS = K ** (n - 1)
        idx_self = False

    Q = [rng.normal(0.0, 1e-6, size=(nS, K)) for _ in range(n)]
    prev = rng.integers(0, K, size=n)

    def enc(i):
        s = 0
        for j in range(n):
            if idx_self or j != i:
                s = s * K + int(prev[j])
        return s

    states = [enc(i) for i in range(n)]
    price_hist = np.empty((T, n), dtype=np.int64)
    profit_hist = np.empty((T, n), dtype=np.float64)
    rand = rng.random
    randint = rng.integers

    for t in range(T):
        e = epsilon if eps_decay is None else epsilon / (1.0 + eps_decay * t)
        a0 = randint(K) if rand() < e else int(np.argmax(Q[0][states[0]]))
        a1 = randint(K) if rand() < e else int(np.argmax(Q[1][states[1]]))
        if n == 2:
            a = (a0, a1)
        else:
            a = [a0, a1] + [randint(K) if rand() < e else int(np.argmax(Q[i][states[i]]))
                            for i in range(2, n)]

        pr = step_profits(np.asarray(a), grid, c)

        # next states
        if idx_self:
            base = 0
            for j in range(n):
                base = base * K + a[j]
            nstates = [base] * n  # joint state identical for all (full price vector)
            # but each agent indexes the same full-vector state; ok since symmetric encoding
        else:
            nstates = []
            for i in range(n):
                s = 0
                for j in range(n):
                    if j != i:
                        s = s * K + a[j]
                nstates.append(s)

        for i in range(n):
            s, ai = states[i], a[i]
            Q[i][s, ai] += alpha * (pr[i] + gamma * Q[i][nstates[i]].max() - Q[i][s, ai])

        price_hist[t] = a
        profit_hist[t] = pr
        states = nstates

    if return_Q:
        return price_hist, profit_hist, Q
    return price_hist, profit_hist


def run_meanbased(n, K, T, epsilon, c, grid, rng):
    """Stateless epsilon-greedy mean-based learning (multi-armed bandit). No state, no gamma."""
    sums = [np.zeros(K) for _ in range(n)]
    counts = [np.zeros(K) for _ in range(n)]
    val = [np.zeros(K) for _ in range(n)]

    price_hist = np.empty((T, n), dtype=np.int64)
    profit_hist = np.empty((T, n), dtype=np.float64)

    for t in range(T):
        a = np.empty(n, dtype=np.int64)
        for i in range(n):
            a[i] = rng.integers(K) if rng.random() < epsilon else _randargmax(val[i], rng)

        pr = step_profits(a, grid, c)
        for i in range(n):
            ai = a[i]
            counts[i][ai] += 1
            sums[i][ai] += pr[i]
            val[i][ai] = sums[i][ai] / counts[i][ai]

        price_hist[t] = a
        profit_hist[t] = pr

    return price_hist, profit_hist


# ----------------------------------------------------------------------
# Classification & metrics
# ----------------------------------------------------------------------
def classify(price_hist, grid, T0=1000, H_star=1.0, p_star=2.0):
    """Shannon (log2) entropy of each agent's price distribution over the final T0 periods."""
    window = price_hist[-T0:]
    K = len(grid)
    n = window.shape[1]
    Hs = []
    for i in range(n):
        counts = np.bincount(window[:, i], minlength=K).astype(float)
        f = counts / counts.sum()
        f = f[f > 0]
        Hs.append(float(-(f * np.log2(f)).sum()))
    H_bar = float(np.mean(Hs))
    p_bar = float(grid[window].mean())
    if H_bar >= H_star:
        regime = "Chaotic"
    elif p_bar < p_star:
        regime = "Competitive"
    else:
        regime = "Collusive"
    return H_bar, p_bar, regime


def delta_index(profit_hist, grid, c, n, T0=1000):
    """Collusion index: 0 = Bertrand-Nash (zero profit), 1 = symmetric joint monopoly."""
    pi = float(profit_hist[-T0:].mean())                 # avg per-firm per-period profit
    pi_monopoly = (grid.max() - c) / n                   # split joint-monopoly profit
    return pi / pi_monopoly


# ----------------------------------------------------------------------
# Baseline config
# ----------------------------------------------------------------------
BASE = dict(n=2, K=41, T=5000, alpha=0.10, gamma=0.95, epsilon=0.10,
            c=1.0, p_min=0.0, p_max=10.0, T0=1000, H_star=1.0, p_star=2.0)
