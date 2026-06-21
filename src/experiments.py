"""
Full experiment suite: generates every figure, table, and statistic in the paper
from the corrected simulation. Single entry point: `python experiments.py`.
All runs are seeded; re-running reproduces identical numbers.
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulation import (make_grid, run_qlearning, run_meanbased,
                        classify, delta_index)

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

# ---- locked baseline (where learning converges; homogeneous-good duopoly) ----
B = dict(n=2, K=7, T=30000, alpha=0.10, gamma=0.95, epsilon=0.10, decay=0.0003,
         c=1.0, p_min=0.0, p_max=10.0, T0=2000, H_star=1.0, p_star=2.0)
COL = {"Competitive": "#2a9d8f", "Collusive": "#e76f51", "Chaotic": "#577590"}


def one_run(seed, alg="q", **ov):
    cfg = dict(B); cfg.update(ov)
    grid = make_grid(cfg["K"], cfg["p_min"], cfg["p_max"])
    rng = np.random.default_rng(seed)
    if alg == "q":
        ph, prof = run_qlearning(cfg["n"], cfg["K"], cfg["T"], cfg["alpha"], cfg["gamma"],
                                 cfg["epsilon"], cfg["c"], grid, rng,
                                 eps_decay=cfg["decay"], memory="opponent")
    else:
        ph, prof = run_meanbased(cfg["n"], cfg["K"], cfg["T"], cfg["epsilon"],
                                 cfg["c"], grid, rng)
    H, p, reg = classify(ph, grid, cfg["T0"], cfg["H_star"], cfg["p_star"])
    d = delta_index(prof, grid, cfg["c"], cfg["n"], cfg["T0"])
    return dict(H=H, p=p, regime=reg, delta=d, ph=ph, grid=grid, T0=cfg["T0"])


def monte_carlo(n_runs, alg="q", seed0=0, **ov):
    rows = []
    for s in range(n_runs):
        r = one_run(seed0 + s, alg=alg, **ov)
        rows.append({k: r[k] for k in ("H", "p", "regime", "delta")})
    return rows


def freqs(rows):
    n = len(rows)
    out = {}
    for reg in ("Competitive", "Collusive", "Chaotic"):
        k = sum(1 for r in rows if r["regime"] == reg)
        p = k / n
        out[reg] = dict(pct=100 * p, se=100 * (p * (1 - p) / n) ** 0.5, count=k)
    out["mean_delta"] = float(np.mean([r["delta"] for r in rows]))
    out["n"] = n
    return out


results = {"baseline_config": {k: B[k] for k in B}}
t_start = time.time()


# ======================================================================
# FIGURE 1 — representative price trajectories for each regime
# ======================================================================
print("[1/7] representative runs ...", flush=True)
def find(regime, **ov):
    for s in range(60):
        r = one_run(s, **ov)
        if r["regime"] == regime:
            return r, s
    return one_run(0, **ov), -1

rep_col, sc = find("Collusive")                              # baseline
rep_comp, sp = find("Competitive", gamma=0.20)               # impatient -> competitive
rep_cha, sh = find("Chaotic", K=41)                          # fine grid -> chaotic
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
for a, (r, ttl) in zip(ax, [(rep_comp, "Competitive ($\\gamma$=0.20)"),
                            (rep_col, "Collusive (baseline)"),
                            (rep_cha, "Chaotic ($K$=41)")]):
    g = r["grid"]; ph = r["ph"]
    a.plot(g[ph[:, 0]], lw=0.6, alpha=0.8, label="Firm 1")
    a.plot(g[ph[:, 1]], lw=0.6, alpha=0.8, label="Firm 2")
    a.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    a.set_title(ttl); a.set_xlabel("Period")
ax[0].set_ylabel("Price"); ax[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Figure 1: Representative price trajectories (dashed = marginal cost)", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_regimes.png"), bbox_inches="tight"); plt.close(fig)


# ======================================================================
# FIGURE 2,3,4 — baseline Monte Carlo (100 runs)
# ======================================================================
print("[2/7] baseline Monte Carlo (100) ...", flush=True)
mc = monte_carlo(100, alg="q")
results["baseline_MC"] = freqs(mc)

H = np.array([r["H"] for r in mc]); P = np.array([r["p"] for r in mc])
creg = [COL[r["regime"]] for r in mc]
fig, a = plt.subplots(figsize=(6, 4.2))
a.scatter(H, P, c=creg, s=28, edgecolor="white", linewidth=0.4)
a.axvline(B["H_star"], color="k", ls="--", lw=0.8); a.axhline(B["p_star"], color="k", ls=":", lw=0.8)
a.set_xlabel("Mean price entropy  $\\bar H$ (bits)"); a.set_ylabel("Mean price  $\\bar p$")
a.set_title("Figure 2: Regime classification, $(\\bar H,\\bar p)$ across 100 runs")
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([0],[0],marker='o',ls='',mfc=COL[k],mec='w',label=k) for k in COL],
         fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_regime_map.png")); plt.close(fig)

fig, a = plt.subplots(figsize=(5.4, 4))
cnts = [results["baseline_MC"][k]["count"] for k in ("Competitive","Collusive","Chaotic")]
a.bar(["Competitive","Collusive","Chaotic"], cnts,
      color=[COL[k] for k in ("Competitive","Collusive","Chaotic")])
a.set_ylabel("Number of runs (of 100)"); a.set_title("Figure 3: Regime frequencies (baseline, 100 runs)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_regime_freq.png")); plt.close(fig)

fig, a = plt.subplots(figsize=(6, 4))
a.hist(P, bins=24, color="#8d99ae", edgecolor="white")
a.axvline(1.0, color="k", ls="--", lw=0.9, label="marginal cost")
a.set_xlabel("Mean price  $\\bar p$ (final window)"); a.set_ylabel("Frequency")
a.set_title("Figure 4: Distribution of mean prices (100 runs)"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_price_dist.png")); plt.close(fig)


# ======================================================================
# FIGURE 5 — Q-learning vs mean-based (100 each)
# ======================================================================
print("[3/7] Q-learning vs mean-based (100+100) ...", flush=True)
mc_q = mc
mc_m = monte_carlo(100, alg="mean")
results["q_vs_mean"] = {"q": freqs(mc_q), "mean": freqs(mc_m)}
labels = ["Competitive","Collusive","Chaotic"]; x = np.arange(3); w = 0.38
fig, a = plt.subplots(figsize=(6.4, 4))
a.bar(x-w/2, [results["q_vs_mean"]["q"][k]["count"] for k in labels], w, label="Q-learning", color="#e76f51")
a.bar(x+w/2, [results["q_vs_mean"]["mean"][k]["count"] for k in labels], w, label="Mean-based", color="#577590")
a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel("Number of runs (of 100)")
a.set_title("Figure 5: Q-learning vs mean-based learning"); a.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_q_vs_mean.png")); plt.close(fig)


# ======================================================================
# FIGURE 6,7 — three firms
# ======================================================================
print("[4/7] three-firm (100) ...", flush=True)
mc3 = monte_carlo(100, alg="q", n=3)
results["three_firm_MC"] = freqs(mc3)
fig, a = plt.subplots(figsize=(5.4, 4))
a.bar(labels, [results["three_firm_MC"][k]["count"] for k in labels], color=[COL[k] for k in labels])
a.set_ylabel("Number of runs (of 100)"); a.set_title("Figure 6: Regime frequencies, 3 firms (100 runs)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_three_firm.png")); plt.close(fig)

r3 = one_run(0, n=3)
fig, a = plt.subplots(figsize=(7.5, 3.6))
for i in range(3): a.plot(r3["grid"][r3["ph"][:, i]], lw=0.5, alpha=0.8, label=f"Firm {i+1}")
a.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
a.set_xlabel("Period"); a.set_ylabel("Price"); a.legend(fontsize=8, ncol=3)
a.set_title(f"Figure 7: Representative 3-firm run ({r3['regime'].lower()})")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_three_firm_run.png")); plt.close(fig)


# ======================================================================
# TABLE 1 — sensitivity analysis (30 runs/cell)
# ======================================================================
print("[5/7] sensitivity sweeps (this is the long one) ...", flush=True)
RUNS = 30
sens = {}
def cell(label, **ov):
    f = freqs(monte_carlo(RUNS, alg="q", **ov))
    sens[label] = f
    print(f"      {label:34s} comp/col/cha = "
          f"{f['Competitive']['count']:>2}/{f['Collusive']['count']:>2}/{f['Chaotic']['count']:>2}"
          f"  Delta={f['mean_delta']:+.2f}", flush=True)

for g in (0.20, 0.50, 0.80, 0.95): cell(f"gamma={g}", gamma=g)
cell("epsilon=0.10 (decaying, base)")
cell("epsilon=0.10 (constant)", decay=None)
cell("epsilon=0.30 (constant)", decay=None, epsilon=0.30)
for K in (7, 11, 21, 41): cell(f"grid K={K}", K=K)
for T in (5000, 15000, 30000, 60000): cell(f"horizon T={T}", T=T)
for n in (2, 3): cell(f"firms n={n}", n=n)
results["sensitivity"] = sens

# gamma figure (the headline mechanism)
print("[6/7] gamma mechanism figure ...", flush=True)
gs = (0.20, 0.50, 0.80, 0.95)
dl = [sens[f"gamma={g}"]["mean_delta"] for g in gs]
fig, a = plt.subplots(figsize=(5.6, 4))
a.plot(gs, dl, "o-", color="#e76f51", lw=2)
a.set_xlabel("Discount factor  $\\gamma$ (patience)"); a.set_ylabel("Collusion index  $\\Delta$")
a.set_title("Figure 8: Patience drives collusion ($\\gamma \\to \\Delta$)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_gamma_delta.png")); plt.close(fig)


# ======================================================================
print("[7/7] saving results.json ...", flush=True)
with open(os.path.join(RES, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# headline summary
b = results["baseline_MC"]; q = results["q_vs_mean"]
print("\n================ HEADLINE RESULTS ================")
print(f"Baseline (n=2,K=7,gamma=.95,annealed eps,T=30k), 100 runs:")
print(f"  Collusive {b['Collusive']['pct']:.0f}% (+/-{b['Collusive']['se']:.1f})  |  "
      f"Chaotic {b['Chaotic']['pct']:.0f}%  |  Competitive {b['Competitive']['pct']:.0f}%  |  "
      f"mean Delta={b['mean_delta']:+.2f}")
print(f"Q-learning collusive {q['q']['Collusive']['pct']:.0f}%  vs  "
      f"mean-based collusive {q['mean']['Collusive']['pct']:.0f}%")
print(f"3 firms: collusive {results['three_firm_MC']['Collusive']['pct']:.0f}%  "
      f"(chaotic {results['three_firm_MC']['Chaotic']['pct']:.0f}%)")
print(f"total runtime {time.time()-t_start:.0f}s")
print("==================================================")
