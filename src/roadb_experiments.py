"""
Road-B experiment suite. Rebuilds the analysis to survive the referee panel:
  A. baseline convergence-conditional Q vs mean (corrected Delta), K=7 and K=21, 100 runs
  B. matched exploration (both annealed / both constant)
  C. split gamma: collusive SHARE vs gamma and intensity DELTA vs gamma (separately)
  D. classifier validation: data-derived H*, (Delta,H) non-redundancy, threshold sensitivity
  E. impulse / forced-deviation test for BOTH learners + no-deviation counterfactual + CI
  F. horizon ladder to 1e6 periods (the convergence-artifact kill shot)

Results saved incrementally to results/roadb.json; figures to figures/.
Every run is seeded; identical on re-run.
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import BASE, P_STAR_FACTOR
from simulation import (make_grid, run_qlearning, run_meanbased, mean_entropy,
                        is_converged, delta_index, regime, nash_profit, monopoly_profit)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
OUT = {}
T_START = time.time()


def save():
    with open(os.path.join(RES, "roadb.json"), "w") as f:
        json.dump(OUT, f, indent=2)


def one(learner, seed, K=BASE["K"], T=BASE["T"], gamma=BASE["gamma"],
        eps_decay=BASE["eps_decay"], eps=BASE["epsilon"], track=True):
    grid = make_grid(K, BASE["p_min"], BASE["p_max"])
    rng = np.random.default_rng(seed)
    if learner == "q":
        res = run_qlearning(K, T, BASE["alpha"], gamma, eps, BASE["c"], grid, rng,
                            eps_decay=eps_decay, track_conv=track)
    else:
        res = run_meanbased(K, T, eps, BASE["c"], grid, rng, eps_decay=eps_decay, track_conv=track)
    return res, grid


def metrics(res, grid, K, T0=BASE["T0"], H_star=0.5):
    Hbar, Hn = mean_entropy(res["price"], K, T0)
    p_bar = float(grid[res["price"][-T0:]].mean())
    d = delta_index(res["price"], res["profit"], grid, BASE["c"], 2, T0)
    conv = is_converged(res) if "pol_stable" in res else None
    return dict(Hbar=Hbar, Hnorm=Hn, p_bar=p_bar, delta=d, converged=conv,
                regime=regime(Hn, p_bar, BASE["c"], H_star, P_STAR_FACTOR))


def se_prop(k, n):
    p = k / n if n else 0.0
    return 100 * (p * (1 - p) / n) ** 0.5 if n else 0.0


def find_antimode(vals, lo=0.0, hi=1.0, bins=40):
    """Antimode of a (hopefully bimodal) distribution on [lo,hi]: the lowest-density
    point between the two largest peaks. Falls back to 0.5 if not clearly bimodal."""
    vals = np.asarray(vals)
    h, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    cen = 0.5 * (edges[:-1] + edges[1:])
    sm = np.convolve(h, np.ones(3) / 3, mode="same")
    # peaks: local maxima
    peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 0]
    if len(peaks) < 2:
        return 0.5
    peaks = sorted(peaks, key=lambda i: sm[i])[-2:]
    a, b = sorted(peaks)
    valley = a + int(np.argmin(sm[a:b + 1]))
    return float(cen[valley])


# ======================================================================
print("[D0] calibration pool -> data-derived H* ...", flush=True)
cal = []
for s in range(60):
    r, g = one("q", s); cal.append(metrics(r, g, BASE["K"])["Hnorm"])
    r, g = one("mean", 1000 + s); cal.append(metrics(r, g, BASE["K"])["Hnorm"])
H_STAR = find_antimode(cal)
OUT["H_star_data_derived"] = H_STAR
OUT["nash_profit_K7"] = nash_profit(make_grid(7, 0, 10), 1.0, 2)
OUT["monopoly_profit"] = monopoly_profit(make_grid(7, 0, 10), 1.0, 2)
print(f"      data-derived H* (K-normalised entropy) = {H_STAR:.3f}", flush=True)
save()


def cell(learner, n_runs, seed0=0, **kw):
    K = kw.get("K", BASE["K"])
    rows = [metrics(*one(learner, seed0 + s, **kw), K=K, H_star=H_STAR) for s in range(n_runs)]
    coll = [r for r in rows if r["regime"] == "Collusive"]
    conv = [r for r in rows if r["converged"]]
    conv_coll = [r for r in conv if r["regime"] == "Collusive"]
    return dict(
        n=n_runs,
        collusive_pct=100 * len(coll) / n_runs, collusive_se=se_prop(len(coll), n_runs),
        conv_rate=100 * len(conv) / n_runs,
        collusive_pct_conv=(100 * len(conv_coll) / len(conv)) if conv else None,
        collusive_se_conv=se_prop(len(conv_coll), len(conv)) if conv else None,
        delta_conv=float(np.mean([r["delta"] for r in conv])) if conv else None,
        delta_conv_se=float(np.std([r["delta"] for r in conv]) / max(1, len(conv) ** 0.5)) if conv else None,
        rows=[{k: r[k] for k in ("Hnorm", "p_bar", "delta", "converged", "regime")} for r in rows],
    )


# ======================================================================
print("[A] baseline convergence-conditional Q vs mean (K=7, K=21; 100 runs) ...", flush=True)
A = {}
for K in (7, 21):
    A[f"q_K{K}"] = cell("q", 100, K=K)
    A[f"mean_K{K}"] = cell("mean", 100, seed0=10_000, K=K)
    for who in ("q", "mean"):
        c = A[f"{who}_K{K}"]
        print(f"      {who} K={K}: collusive all={c['collusive_pct']:.0f}% "
              f"conv-only={c['collusive_pct_conv']}% conv-rate={c['conv_rate']:.0f}% "
              f"Delta(conv)={c['delta_conv']}", flush=True)
OUT["A_baseline"] = {k: {kk: v[kk] for kk in v if kk != "rows"} for k, v in A.items()}
save()

# Figure: Q vs mean, all vs converged-only, both grids
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for j, K in enumerate((7, 21)):
    lab = ["Q-learning", "Mean-based"]; x = np.arange(2); w = 0.35
    allp = [A[f"q_K{K}"]["collusive_pct"], A[f"mean_K{K}"]["collusive_pct"]]
    cnvp = [A[f"q_K{K}"]["collusive_pct_conv"] or 0, A[f"mean_K{K}"]["collusive_pct_conv"] or 0]
    ax[j].bar(x - w/2, allp, w, label="all runs", color="#8d99ae")
    ax[j].bar(x + w/2, cnvp, w, label="converged only", color="#e76f51")
    ax[j].set_xticks(x); ax[j].set_xticklabels(lab); ax[j].set_title(f"K = {K}")
    ax[j].set_ylabel("collusive share (%)"); ax[j].legend(fontsize=8)
fig.suptitle("Collusive share: all runs vs convergence-conditional", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rb_fig1_conv_conditional.png"), bbox_inches="tight"); plt.close(fig)

# ======================================================================
print("[B] matched exploration ...", flush=True)
B = {}
for sched, dec in (("annealed", BASE["eps_decay"]), ("constant", None)):
    B[f"q_{sched}"] = cell("q", 100, K=21, eps_decay=dec)
    B[f"mean_{sched}"] = cell("mean", 100, seed0=10_000, K=21, eps_decay=dec)
    print(f"      {sched}: Q={B['q_'+sched]['collusive_pct']:.0f}% "
          f"mean={B['mean_'+sched]['collusive_pct']:.0f}%", flush=True)
OUT["B_matched_exploration"] = {k: {kk: v[kk] for kk in v if kk != "rows"} for k, v in B.items()}
save()

# ======================================================================
print("[C] split gamma (share vs intensity) ...", flush=True)
C = {}
for g in (0.20, 0.50, 0.80, 0.95):
    C[f"g{g}"] = cell("q", 100, K=21, gamma=g)
    print(f"      gamma={g}: share={C[f'g{g}']['collusive_pct']:.0f}% "
          f"Delta(conv)={C[f'g{g}']['delta_conv']}", flush=True)
OUT["C_gamma"] = {k: {kk: v[kk] for kk in v if kk != "rows"} for k, v in C.items()}
gs = [0.20, 0.50, 0.80, 0.95]
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()
ax1.plot(gs, [C[f"g{g}"]["collusive_pct"] for g in gs], "o-", color="#577590", label="collusive share")
ax2.plot(gs, [C[f"g{g}"]["delta_conv"] or 0 for g in gs], "s--", color="#e76f51", label="Delta (intensity)")
ax1.set_xlabel("discount factor gamma"); ax1.set_ylabel("collusive share (%)", color="#577590")
ax2.set_ylabel("Delta on converged runs", color="#e76f51")
ax1.set_title("Patience: share (convergence) vs intensity are different")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rb_fig2_gamma_split.png")); plt.close(fig)
save()

# ======================================================================
print("[D] classifier validation: (Delta,H) scatter + threshold sensitivity ...", flush=True)
pool = A["q_K21"]["rows"] + A["mean_K21"]["rows"]
Hn = np.array([r["Hnorm"] for r in pool]); Dl = np.array([r["delta"] for r in pool])
# threshold sensitivity: collusive share as H* sweeps (honesty about dependence)
sweep = np.linspace(0.2, 0.9, 15)
share_vs_Hstar = []
for hs in sweep:
    sh = np.mean([(r["Hnorm"] < hs) and (r["p_bar"] >= P_STAR_FACTOR * BASE["c"]) for r in pool]) * 100
    share_vs_Hstar.append(float(sh))
OUT["D_classifier"] = dict(H_star=H_STAR, share_vs_Hstar=list(zip(sweep.tolist(), share_vs_Hstar)),
                           hnorm_min=float(Hn.min()), hnorm_max=float(Hn.max()))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(Hn, Dl, s=14, alpha=0.6, color="#577590")
ax[0].axvline(H_STAR, color="k", ls="--", lw=1, label=f"data-derived H*={H_STAR:.2f}")
ax[0].set_xlabel("K-normalised entropy H"); ax[0].set_ylabel("collusion index Delta")
ax[0].set_title("(H, Delta): entropy is not redundant with Delta"); ax[0].legend(fontsize=8)
ax[1].plot(sweep, share_vs_Hstar, "o-", color="#e76f51")
ax[1].axvline(H_STAR, color="k", ls="--", lw=1)
ax[1].set_xlabel("entropy threshold H*"); ax[1].set_ylabel("collusive share (%)")
ax[1].set_title("Threshold sensitivity (reported honestly)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rb_fig3_classifier.png")); plt.close(fig)
save()

# ======================================================================
print("[E] impulse test for BOTH learners + no-deviation counterfactual ...", flush=True)
def impulse(learner, n_target=30, horizon=25, settle=8, K=21):
    devs, rivs, base_dev, base_riv = [], [], [], []
    grid = make_grid(K, 0, 10); s = 0; used = 0
    while used < n_target and s < 600:
        res, _ = one(learner, s, K=K)
        m = metrics(res, grid, K, H_star=H_STAR)
        if m["regime"] != "Collusive" or (m["converged"] is False):
            s += 1; continue
        coll = int(np.bincount(res["price"][-BASE["T0"]:, 0], minlength=K).argmax())
        if learner == "q":
            rng = np.random.default_rng(s)
            r2 = run_qlearning(K, BASE["T"], BASE["alpha"], BASE["gamma"], BASE["epsilon"],
                               BASE["c"], grid, rng, eps_decay=BASE["eps_decay"], return_Q=True)
            Q = r2["Q"]
            def greedy(i, opp_prev): return int(np.argmax(Q[i][opp_prev]))
            def step(prev, force0):
                a0 = 0 if force0 else greedy(0, prev[1])
                a1 = greedy(1, prev[0]); return a0, a1
        else:
            val = res["val"]
            def step(prev, force0):
                a0 = 0 if force0 else int(np.argmax(val[0]))
                a1 = int(np.argmax(val[1])); return a0, a1
        for forced in (True, False):
            prev = [coll, coll]; p0, p1 = [], []
            for t in range(-settle, horizon):
                a0, a1 = step(prev, forced and t == 0)
                p0.append(grid[a0]); p1.append(grid[a1]); prev = [a0, a1]
            (devs if forced else base_dev).append(p0)
            (rivs if forced else base_riv).append(p1)
        used += 1; s += 1
    return (np.array(devs), np.array(rivs), np.array(base_dev), np.array(base_riv), used)

E = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); xx = np.arange(-8, 25)
for j, lr in enumerate(("q", "mean")):
    dv, rv, bdv, brv, used = impulse(lr)
    def band(arr):
        m = arr.mean(0); n = len(arr)
        lo = np.percentile([arr[np.random.default_rng(b).integers(0, n, n)].mean(0) for b in range(200)], 2.5, 0)
        hi = np.percentile([arr[np.random.default_rng(b).integers(0, n, n)].mean(0) for b in range(200)], 97.5, 0)
        return m, lo, hi
    rm, rlo, rhi = band(rv); bm, blo, bhi = band(brv)
    ax[j].plot(xx, dv.mean(0), "-", color="#e76f51", lw=1.4, label="deviator (forced undercut)")
    ax[j].plot(xx, rm, "-o", ms=2, color="#577590", lw=1.4, label="rival (response)")
    ax[j].fill_between(xx, rlo, rhi, color="#577590", alpha=0.2)
    ax[j].plot(xx, bm, ":", color="#2a9d8f", lw=1.2, label="rival (NO deviation)")
    ax[j].fill_between(xx, blo, bhi, color="#2a9d8f", alpha=0.15)
    ax[j].axvline(0, color="k", ls=":", lw=0.8)
    ax[j].set_title(f"{'Q-learning' if lr=='q' else 'Mean-based'} (n={used})")
    ax[j].set_xlabel("period rel. to deviation"); ax[j].legend(fontsize=7)
    E[lr] = dict(n=used, rival_pre=float(rv[:, :8].mean()), rival_trough=float(rv.mean(0)[8:].min()),
                 nodev_band_amp=float(brv.mean(0).max() - brv.mean(0).min()))
ax[0].set_ylabel("price")
fig.suptitle("Forced-deviation response: does the rival actually punish?", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rb_fig4_impulse_both.png"), bbox_inches="tight"); plt.close(fig)
OUT["E_impulse"] = E
print(f"      Q: rival pre={E['q']['rival_pre']:.2f} trough={E['q']['rival_trough']:.2f} "
      f"nodev-amp={E['q']['nodev_band_amp']:.2f} | mean: pre={E['mean']['rival_pre']:.2f} "
      f"trough={E['mean']['rival_trough']:.2f} nodev-amp={E['mean']['nodev_band_amp']:.2f}", flush=True)
save()

# ======================================================================
print("[F] HORIZON LADDER to 1e6 (the convergence-artifact kill shot) ...", flush=True)
ladder = {"T": [], "q_share": [], "q_se": [], "m_share": [], "m_se": [],
          "q_conv": [], "m_conv": [], "q_delta": [], "m_delta": []}
T_LIST = [30_000, 100_000, 300_000, 1_000_000]
SEEDS = {30_000: 24, 100_000: 24, 300_000: 16, 1_000_000: 10}
for T in T_LIST:
    ns = SEEDS[T]; t0 = time.time()
    qc = cell("q", ns, K=21, T=T)
    mc = cell("mean", ns, seed0=20_000, K=21, T=T)
    ladder["T"].append(T)
    ladder["q_share"].append(qc["collusive_pct"]); ladder["q_se"].append(qc["collusive_se"])
    ladder["m_share"].append(mc["collusive_pct"]); ladder["m_se"].append(mc["collusive_se"])
    ladder["q_conv"].append(qc["conv_rate"]); ladder["m_conv"].append(mc["conv_rate"])
    ladder["q_delta"].append(qc["delta_conv"]); ladder["m_delta"].append(mc["delta_conv"])
    print(f"      T={T}: Q share={qc['collusive_pct']:.0f}% (conv {qc['conv_rate']:.0f}%) "
          f"mean share={mc['collusive_pct']:.0f}% ({time.time()-t0:.0f}s)", flush=True)
    OUT["F_horizon_ladder"] = ladder; save()

fig, ax = plt.subplots(figsize=(6.5, 4.3))
lt = np.array(ladder["T"])
ax.errorbar(lt, ladder["q_share"], yerr=ladder["q_se"], marker="o", lw=1.6, color="#e76f51", label="Q-learning")
ax.errorbar(lt, ladder["m_share"], yerr=ladder["m_se"], marker="s", lw=1.6, color="#577590", label="Mean-based")
ax.set_xscale("log"); ax.set_xlabel("horizon T (log)"); ax.set_ylabel("collusive share (%)")
ax.set_title("Horizon ladder: does Q-learning catch up?"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rb_fig5_horizon_ladder.png")); plt.close(fig)
save()

print(f"\nDONE in {time.time()-T_START:.0f}s. H*={H_STAR:.3f}. results/roadb.json + 5 figures.", flush=True)
