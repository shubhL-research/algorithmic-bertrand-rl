"""
Single source of truth for the canonical baseline (fixes the code/paper mismatch).
Every script imports BASE from here; nothing redefines it.
"""

# Canonical baseline: a tractable price grid where memory-1 Q-learning converges
# within a feasible horizon. Grid resolution (K) and horizon (T) are themselves
# treated as objects of study (see the horizon ladder and grid sweep).
BASE = dict(
    n=2,            # firms
    K=7,            # price grid points (baseline; swept up to 41)
    T=30_000,       # learning horizon (baseline; swept up to 1,000,000)
    alpha=0.10,     # learning rate
    gamma=0.95,     # discount factor (patience)
    epsilon=0.10,   # initial exploration
    eps_decay=3e-4, # annealed schedule: eps_t = eps / (1 + eps_decay * t)
    c=1.0,          # marginal cost
    p_min=0.0,
    p_max=10.0,
    T0=2_000,       # classification / averaging window (final T0 periods)
)

# Classification thresholds. H_STAR is DATA-DERIVED (antimode of the entropy
# marginal); it is set here only as a fallback and is re-estimated in the
# classifier-validation step. p_star is the competitive/collusive price cut.
H_STAR_DEFAULT = 0.50   # on K-NORMALISED entropy in [0,1] (see classify_norm)
P_STAR_FACTOR = 2.0     # collusive if mean price >= P_STAR_FACTOR * c
