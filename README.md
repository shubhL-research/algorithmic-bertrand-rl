# The Two Faces of Algorithmic Collusion

Reproducible code and paper for a computational study of **agent memory** in homogeneous-good Bertrand
competition: when do pricing algorithms reach supra-competitive prices, and is that coordination *genuine*
(strategically punished) or *spurious* (a stateless artifact)?

**Author:** Shubh Lamba · **Paper:** [`paper.pdf`](paper.pdf) (10 pp.)

## Findings

A stateful ε-greedy **Q-learner** (state = rival's previous price) is contrasted with a stateless
**mean-based bandit**, on a homogeneous-good Bertrand grid, classified by a model-free,
**convergence-validated** Shannon-entropy diagnostic. 100 seeded runs per cell; horizon ladder to 10⁶.

1. **Reliability vs. intensity.** The *stateless* learner reaches supra-competitive prices immediately and
   almost always (90–96% of runs) but **weakly** (collusion index Δ≈0.13). The *stateful* learner
   **converges to competition** at normal horizons (2% collusive despite 89% convergence) and learns to
   collude only slowly as the horizon grows (0→25→44→50% at T = 3×10⁴ → 10⁶), but **intensely** (Δ→0.46).
   **Memory, not sophistication, is the operative margin.**
2. **Spurious vs. genuine.** A forced-deviation test: the Q-learner punishes a deviation sharply (rival
   price 5.61→4.05 vs. a 0.45 no-deviation band) — genuine, punishment-supported collusion. The mean-based
   learner cannot punish (response amplitude **0.00**) — structurally spurious coordination.
3. **No convergence artifact.** Because the stateless learner *converges* and still colludes, its advantage
   is not the usual "tabular Q-learning didn't finish learning" artifact.

Policy reading: the most reliable coordination is the most spurious and least detectable, so an
outcome-based, **algorithm-agnostic** screen (like the entropy diagnostic here) is the appropriate
instrument.

## Reproduce

```bash
pip install -r requirements.txt
cd src && python roadb_experiments.py     # ~50 min; regenerates results/roadb.json + figures/rb_fig1-5
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Every result is produced from fixed random seeds; re-running is identical.

## Layout

```
src/config.py            single source of truth for the baseline (K, T, alpha, gamma, eps, ...)
src/simulation.py        market, Q-learning (eq. 2), mean-based bandit, corrected discrete-Nash Delta,
                         convergence diagnostics, K-normalised bias-corrected entropy
src/roadb_experiments.py the full suite: convergence-conditional comparison, matched exploration,
                         split-gamma, classifier validation, impulse test, horizon ladder
figures/rb_fig1-5.png    the five figures in the paper
results/roadb.json       all numbers; results/citations.md = annotated bibliography
paper.tex, references.bib, paper.pdf
```

## Model (baseline)

n = 2 firms, homogeneous good, marginal cost c = 1, inelastic unit demand, price grid [0,10] with K = 21
(coarse contrast K = 7); α = 0.10, γ = 0.95, annealed ε (ε₀ = 0.10, δ = 3×10⁻⁴), T = 30,000 (swept to
10⁶); classification window T₀ = 2,000; entropy threshold H\* = 0.288 (data-derived antimode). The Q-update is

```
Q(s,a) <- Q(s,a) + alpha * [ pi + gamma * max_a' Q(s',a') - Q(s,a) ],   s = rival's previous price.
```

The collusion index Δ ∈ [0,1] is anchored on the true **discrete** Bertrand-Nash profit (not zero) and the
joint-monopoly profit, computed on converged runs.

## License

MIT — see [`LICENSE`](LICENSE).
