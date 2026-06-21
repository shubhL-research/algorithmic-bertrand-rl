# Learning Dynamics in Algorithmic Bertrand Competition

Reproducible code and paper for a computational study of reinforcement-learning agents in repeated
**homogeneous-good** Bertrand competition — the environment in which classical theory predicts the
sharpest competitive outcome (marginal-cost pricing).

**Author:** Shubh Lamba · **Paper:** [`paper.pdf`](paper.pdf)

## Summary of findings

Faithful ε-greedy Q-learning (temporal-difference update, discount factor, state = rivals' previous
prices) and a stateless mean-based (bandit) benchmark, 100 independent runs per configuration, classified
into competitive / collusive / chaotic regimes by a Shannon-entropy measure. Collusion is summarised by a
normalised index Δ ∈ [0,1] (0 = Bertrand–Nash, 1 = symmetric joint monopoly).

1. **Collusion is real but conditional.** 56% of baseline duopoly runs collude (Δ ≈ 0.43); 44% are
   chaotic (non-convergent). Collusion requires a tractable price grid, annealed exploration, and a long
   horizon.
2. **Sophistication is not the driver.** The *simpler* mean-based bandit colludes **more** (89%) than
   Q-learning (56%); the Q-learner's bootstrap makes convergence harder.
3. **Patience governs intensity.** Δ rises monotonically with the discount factor γ (0.23 → 0.45 as
   γ: 0.2 → 0.95) — the folk-theorem mechanism.
4. **Fragility.** Collusion collapses as the grid fines (57% → 3%), as exploration persists (→ 0%), as the
   horizon shortens (→ 10%), and as a third firm is added (→ 0%).

## Reproduce everything

```bash
pip install -r requirements.txt
cd src
python experiments.py        # ~15 min; regenerates all figures + results/results.json
```

Every result is produced from fixed random seeds; re-running yields identical numbers. Compile the paper
with `pdflatex paper.tex` (twice).

## Repository layout

```
src/simulation.py     core model: market, Q-learning (eq. 7), mean-based learner, entropy classifier, Δ
src/experiments.py    full experiment suite -> figures/ + results/
figures/              fig1–fig8 (all referenced in the paper)
results/results.json  all numbers (regime frequencies, SEs, Δ, sensitivity table)
paper.tex, paper.pdf  the write-up
```

## Model (baseline)

n = 2 firms, homogeneous good, marginal cost c = 1, inelastic unit demand, price grid [0, 10] with K = 7
points; α = 0.10, γ = 0.95, annealed ε (ε₀ = 0.10, δ = 3×10⁻⁴), T = 30,000; classification window
T₀ = 2,000; thresholds H\* = 1.0, p\* = 2.0. The Q-update is

```
Q(s,a) <- Q(s,a) + alpha * [ pi + gamma * max_a' Q(s',a') - Q(s,a) ],   s = rivals' previous prices.
```

## License

MIT — see [`LICENSE`](LICENSE).
