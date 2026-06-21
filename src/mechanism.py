"""
Mechanism evidence + inferential statistics for the paper (journal-level rigor).

1. Forced-deviation impulse response: on a CONVERGED collusive run we freeze the
   learned policy (greedy, no learning), confirm steady collusive play, force one
   firm to undercut for a single period, and trace the price path. This is the
   standard test for a reward-punishment ("price war + recovery") strategy.
2. Two-proportion z-test for the Q-learning vs mean-based collusion-rate gap.
3. Per-regime descriptive statistics.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from simulation import make_grid, run_qlearning, run_meanbased, classify, delta_index

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

B = dict(n=2, K=7, T=30000, alpha=0.10, gamma=0.95, epsilon=0.10, decay=0.0003,
         c=1.0, T0=2000, H_star=1.0, p_star=2.0)
grid = make_grid(B["K"], 0.0, 10.0)
KSZ = B["K"]


def greedy(Qi, opp_prev):
    return int(np.argmax(Qi[opp_prev]))


def impulse_for_seed(seed, horizon=25, settle=8):
    """Return price path of the deviator and the rival around a forced undercut,
    or None if the run did not converge to collusion."""
    rng = np.random.default_rng(seed)
    ph, prof, Q = run_qlearning(2, KSZ, B["T"], B["alpha"], B["gamma"], B["epsilon"],
                                B["c"], grid, rng, eps_decay=B["decay"],
                                memory="opponent", return_Q=True)
    H, p, reg = classify(ph, grid, B["T0"], B["H_star"], B["p_star"])
    if reg != "Collusive":
        return None
    # collusive price = modal action of firm 0 over the converged window
    coll = int(np.bincount(ph[-B["T0"]:, 0], minlength=KSZ).argmax())
    # freeze policy; start both at collusive state
    prev = [coll, coll]
    path0, path1 = [], []
    for t in range(-settle, horizon):
        if t == 0:
            a0 = 0                      # forced undercut by firm 0 (lowest price)
        else:
            a0 = greedy(Q[0], prev[1])
        a1 = greedy(Q[1], prev[0])
        path0.append(grid[a0]); path1.append(grid[a1])
        prev = [a0, a1]
    return np.array(path0), np.array(path1)


def main():
    print("impulse responses (scanning collusive seeds) ...", flush=True)
    dev, riv = [], []
    s = 0
    while len(dev) < 40 and s < 400:
        out = impulse_for_seed(s)
        if out is not None:
            dev.append(out[0]); riv.append(out[1])
        s += 1
    dev = np.array(dev); riv = np.array(riv)
    n_used = len(dev)
    md, mr = dev.mean(0), riv.mean(0)
    x = np.arange(-8, 25)

    fig, a = plt.subplots(figsize=(7.2, 4.2))
    a.axvline(0, color="k", ls=":", lw=1, alpha=0.6)
    a.axhline(grid[int(np.bincount([0]).argmax())], color="none")
    a.plot(x, md, "-o", ms=3, lw=1.6, color="#e76f51", label="Deviator (forced undercut at $t{=}0$)")
    a.plot(x, mr, "-o", ms=3, lw=1.6, color="#577590", label="Rival (response)")
    a.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5, label="marginal cost")
    a.set_xlabel("Period relative to deviation"); a.set_ylabel("Price")
    a.set_title(f"Figure 9: Reward–punishment response ($n{{=}}{n_used}$ collusive runs, frozen policy)")
    a.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_impulse.png")); plt.close(fig)

    # ---- magnitude of punishment + recovery ----
    pre = mr[:8].mean()
    trough = mr[8:].min()
    recovered = mr[-1]
    print(f"  used {n_used} collusive runs; rival pre={pre:.2f} trough={trough:.2f} end={recovered:.2f}")

    # ---- two-proportion z-test: Q vs mean-based collusion rate ----
    print("z-test on Q vs mean-based (100 runs each) ...", flush=True)
    def coll_count(alg, nr=100):
        k = 0
        for sd in range(nr):
            rng = np.random.default_rng((0 if alg == "q" else 10_000) + sd)
            if alg == "q":
                ph, pf = run_qlearning(2, KSZ, B["T"], B["alpha"], B["gamma"], B["epsilon"],
                                       B["c"], grid, rng, eps_decay=B["decay"], memory="opponent")
            else:
                ph, pf = run_meanbased(2, KSZ, B["T"], B["epsilon"], B["c"], grid, rng)
            if classify(ph, grid, B["T0"], B["H_star"], B["p_star"])[2] == "Collusive":
                k += 1
        return k
    kq, km, N = coll_count("q"), coll_count("mean"), 100
    pq, pm = kq / N, km / N
    pbar = (kq + km) / (2 * N)
    z = (pm - pq) / np.sqrt(pbar * (1 - pbar) * (2 / N))
    pval = 2 * (1 - norm.cdf(abs(z)))
    print(f"  Q collusive {kq}/100, mean-based {km}/100, z={z:.2f}, p={pval:.2e}")

    out = dict(impulse=dict(n_runs=n_used, rival_pre=float(pre), rival_trough=float(trough),
                            rival_end=float(recovered),
                            deviator_path=md.tolist(), rival_path=mr.tolist(), x=x.tolist()),
               q_vs_mean_test=dict(k_q=kq, k_mean=km, N=N, p_q=pq, p_mean=pm,
                                   z=float(z), p_value=float(pval)))
    with open(os.path.join(RES, "mechanism.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/mechanism.json")


if __name__ == "__main__":
    main()
