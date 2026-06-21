# 1. Annotated Bibliography (verified papers only)

**Dropped:** None. All 13 supplied references carry `verified:true`; nothing was excluded.

---

**Calvano, Calzolari, Denicolò & Pastorello (2020).** "Artificial Intelligence, Algorithmic Pricing, and Collusion." *American Economic Review*, 110(10), 3267–3297.
- *What it shows:* Q-learners in a differentiated-product (logit) Bertrand oligopoly autonomously learn supra-competitive prices sustained by genuine reward-punishment strategies.
- *How we relate:* Canonical anchor; we move to the harder homogeneous-good coarse-grid case, add a stateless comparator, and condition on convergence — without claiming "algorithms collude" as our novelty.

**Klein (2021).** "Autonomous algorithmic collusion: Q-learning under sequential pricing." *RAND Journal of Economics*, 52(3), 538–558.
- *What it shows:* Two memory-1 Q-learners pricing *sequentially* learn reward-punishment-supported collusive focal prices (coarse grid) or supra-competitive Edgeworth cycles (fine grid).
- *How we relate:* Corroborates our Q-learning arm's strategic collusion, but he uses sequential moves and never contrasts a stateless bandit; we hold the game simultaneous and add that learner-memory contrast.

**Asker, Fershtman & Pakes (2024).** "The impact of artificial intelligence design on pricing." *Journal of Economics & Management Strategy*, 33(2), 276–304.
- *What it shows:* The update's information structure — synchronous (counterfactual) vs. asynchronous (realized-action-only) — drives competitive vs. near-monopoly pricing, not sophistication.
- *How we relate:* Both our learners are asynchronous (so AFP explains *why* both reach high prices); we hold that fixed and isolate the orthogonal stateful-vs-stateless reliability/intensity margin.

**Banchio & Mantegazza (2023).** "Adaptive Algorithms and Collusion via Coupling." *Proceedings of the 24th ACM Conference on Economics and Computation (EC '23)*, p. 208. (Preprint: arXiv:2202.05946.)
- *What it shows:* Stateless, myopic, mean-based learners can periodically coordinate on supra-competitive actions via endogenous "spontaneous coupling," predictable from primitives.
- *How we relate:* **High-threat.** We adopt "spontaneous coupling" as the named mechanism for our bandit result and contribute a forced-deviation diagnostic that certifies it as *non-punishable*, distinguishing it from our Q-learner's enforceable collusion.

**Abada & Lambin (2023).** "Artificial Intelligence: Can Seemingly Collusive Outcomes Be Avoided?" *Management Science*, 69(9), 5042–5065.
- *What it shows:* In storable-good arbitrage, simple ML agents reach seemingly collusive outcomes from *imperfect exploration* (not sophistication), correctable by raising exploration or intervening in learning.
- *How we relate:* **High-threat / closest in spirit.** We move from "it can be spurious" to an operational *when*-it-is-spurious test in canonical Bertrand, plus the reliability-vs-intensity decomposition they lack.

**Abada, Lambin & Tchakarov (2024).** "Collusion by mistake: Does algorithmic sophistication drive supra-competitive profits?" *European Journal of Operational Research*, 318(3), 927–953.
- *What it shows:* Supra-competitive Q-prices are often an under-exploration "mistake," not equilibrium; larger epsilon restores competition — sophistication does not drive collusion.
- *How we relate:* **High-threat.** Motivates our convergence-conditioning and 1e6 horizons; we find their verdict fits the *stateless bandit* but not *converged* memory-1 Q-learning, recasting their dichotomy as reliability-vs-intensity.

**den Boer, Meylahn & Schinkel (forthcoming, 2024 accepted).** "Artificial Collusion: Examining Supracompetitive Pricing by Q-learning Algorithms." *Management Science*, DOI 10.1287/mnsc.2024.08557 (in press).
- *What it shows:* Q-learning's collusion is fragile, knife-edge, and learned only on economically irrelevant timescales; not a genuine cartel-risk threat.
- *How we relate:* **High-threat but supportive.** We treat their critique as the methodological standard we meet (convergence-conditioned, horizon-robust), then draw the stateful-vs-stateless and detection distinction they do not address.

**Hartline, Long & Zhang (2024).** "Regulation of Algorithmic Collusion." *Proceedings of the 2024 Symposium on Computer Science and Law (CSLAW '24)*, 11 pp. (Also arXiv:2401.15794.)
- *What it shows:* A code-free statistical audit; no-swap/calibrated regret guarantees competition while external/mean-based no-regret is *insufficient* and can sustain supra-competitive prices.
- *How we relate:* **High-threat (framing only).** Their theorem does *not* say "mean-based ⇒ competitive"; it permits supra-competitive mean-based pricing, so our concrete stateless-bandit result is consistent with — not refuted by — them.

**Dolgopolov (2024).** "Reinforcement learning in a prisoner's dilemma." *Games and Economic Behavior*, 144(C), 84–103.
- *What it shows:* For memoryless RL in a 2×2 PD, under vanishing epsilon-greedy exploration the unique stochastically stable outcome is competitive defection; cooperation needs logit exploration.
- *How we relate:* **High-threat.** Reinforces our spurious-bandit verdict; our positive Q-collusion escapes his theorem because our Q-learner is *stateful* (memory-1), where punishment is feasible.

**Waltman & Kaymak (2008).** "Q-learning agents in a Cournot oligopoly model." *Journal of Economic Dynamics and Control*, 32(10), 3275–3293.
- *What it shows:* Q-learning Cournot firms reach supra-competitive outcomes even when memoryless (no trigger strategies), a bias arising from Q-value updating itself.
- *How we relate:* Their "collusion without punishment" *is* the spurious regime our forced-deviation test isolates; we differ in market (Bertrand), the stateless-vs-stateful contrast, and an explicit deviation test.

**Calvano, Calzolari, Denicolò & Pastorello (2023).** "Algorithmic collusion: Genuine or spurious?" *International Journal of Industrial Organization*, 90(C), 102973.
- *What it shows:* Genuine collusion (reward-punishment-supported) vs. spurious supra-competitive pricing (exploration-mode/non-equilibrium artifact); with random exploration, Q-learning collusion is genuine.
- *How we relate:* Supplies the genuine-vs-spurious frame our finding (3) operationalizes; we add *statelessness itself* as a structural (not exploration-driven) source of spuriousness.

**Possnig (2023, working paper; R&R *Theoretical Economics*).** "Reinforcement Learning and Collusion" (revised as "Monitoring, Market Primitives, and the Stability of Algorithmic Collusion"). University of Waterloo working paper.
- *What it shows:* Via stochastic-approximation/ODE analysis, whether RL escapes competitive Nash depends on the *state variables/monitoring* the algorithms condition on, not raw sophistication.
- *How we relate:* Theoretical backbone for our state-vs-no-state contrast; we provide the honest computational instantiation plus reliability-vs-intensity and the deviation diagnostic.

**Bischi & Lamantia (2022).** "Evolutionary oligopoly games with cooperative and aggressive behaviors." *Journal of Economic Interaction and Coordination*, 17(1), 3–27.
- *What it shows:* A replicator/imitation oligopoly with cooperative vs. aggressive types yields convergence, mixed coexistence, or periodic/chaotic regimes depending on memory and intensity.
- *How we relate:* Venue-fit "house" paper anchoring our regime taxonomy; but their cooperation is hand-wired into objectives with no learning agents, so it cannot pre-empt our discovered-collusion or deviation-test results.

---

# 2. "Closest related work" paragraph (≈315 words)

Our study sits within the algorithmic-collusion literature founded by Calvano et al. (2020) and Klein (2021), who establish that Q-learning agents autonomously learn reward-punishment-supported supra-competitive pricing in differentiated-product and sequential Bertrand settings respectively. We depart from this canon in three ways: we study the harder homogeneous-good case on a coarse memory-1 grid; we add a model-free, convergence-validated entropy classifier that sorts runs into competitive, collusive, and chaotic regimes; and we contrast a stateful Q-learner against a stateless mean-based bandit. The closest and most demanding neighbors warn that supra-competitive pricing may be an artifact rather than strategy. Abada and Lambin (2023) and Abada, Lambin and Tchakarov (2024) trace it to imperfect exploration and under-convergence ("collusion by mistake"); den Boer, Meylahn and Schinkel (forthcoming) argue Q-learning collusion is fragile and emerges only on economically irrelevant timescales. Rather than contest these critiques, we adopt them as our methodological baseline — conditioning every collusion statistic on validated convergence over horizons up to 1e6 periods — and then go further than a debunking by drawing two distinctions they do not. First, reliability-versus-intensity: the stateless bandit reaches supra-competitive prices *more often* but at *lower* normalized collusion index, while the Q-learner colludes *less often* but *harder*. Second, a genuine-versus-spurious test indexed by learner class: a forced-deviation diagnostic shows the bandit's coordination — consistent with Banchio and Mantegazza's (2023) "spontaneous coupling" and Calvano et al.'s (2023) spurious category — is structurally non-punishable, whereas converged Q-learning is reward-punishment-supported. Crucially, our mean-based result is *not* refuted by Hartline, Long and Zhang (2024): their theorem shows external/mean-based no-regret is *insufficient* for competition and can permit supra-competitive prices — it never implies mean-based learners go competitive. Our finding is a concrete, finite-horizon, tabular, epsilon-greedy instantiation of exactly that permitted regime, and Dolgopolov's (2024) stateless-defection theorem applies to our bandit, not to our memory-enabled Q-learner. We thus complement Asker-Fershtman-Pakes (2024) and Possnig (2023) by isolating learner memory, not update information, as the operative margin.

---

# 3. BibTeX

```bibtex
@article{calvano2020artificial,
  author  = {Calvano, Emilio and Calzolari, Giacomo and Denicol{\`o}, Vincenzo and Pastorello, Sergio},
  title   = {Artificial Intelligence, Algorithmic Pricing, and Collusion},
  journal = {American Economic Review},
  year    = {2020},
  volume  = {110},
  number  = {10},
  pages   = {3267--3297},
  doi     = {10.1257/aer.20190623}
}

@article{klein2021autonomous,
  author  = {Klein, Timo},
  title   = {Autonomous algorithmic collusion: {Q}-learning under sequential pricing},
  journal = {RAND Journal of Economics},
  year    = {2021},
  volume  = {52},
  number  = {3},
  pages   = {538--558},
  doi     = {10.1111/1756-2171.12383}
}

@article{AskerFershtmanPakes2024,
  author  = {Asker, John and Fershtman, Chaim and Pakes, Ariel},
  title   = {The impact of artificial intelligence design on pricing},
  journal = {Journal of Economics \& Management Strategy},
  year    = {2024},
  volume  = {33},
  number  = {2},
  pages   = {276--304},
  doi     = {10.1111/jems.12516}
}

@inproceedings{banchioMantegazza2023,
  author    = {Banchio, Martino and Mantegazza, Giacomo},
  title     = {Adaptive Algorithms and Collusion via Coupling},
  booktitle = {Proceedings of the 24th ACM Conference on Economics and Computation (EC '23)},
  year      = {2023},
  pages     = {208},
  publisher = {ACM},
  doi       = {10.1145/3580507.3597726},
  note      = {Preprint: arXiv:2202.05946; preprint titled ``Artificial Intelligence and Spontaneous Collusion''}
}

@article{abada_lambin_2023,
  author  = {Abada, Ibrahim and Lambin, Xavier},
  title   = {Artificial Intelligence: Can Seemingly Collusive Outcomes Be Avoided?},
  journal = {Management Science},
  year    = {2023},
  volume  = {69},
  number  = {9},
  pages   = {5042--5065},
  doi     = {10.1287/mnsc.2022.4623}
}

@article{abada2024collusion,
  author  = {Abada, Ibrahim and Lambin, Xavier and Tchakarov, Nikolay},
  title   = {Collusion by mistake: Does algorithmic sophistication drive supra-competitive profits?},
  journal = {European Journal of Operational Research},
  year    = {2024},
  volume  = {318},
  number  = {3},
  pages   = {927--953},
  doi     = {10.1016/j.ejor.2024.06.006}
}

@article{denBoerMeylahnSchinkel2024,
  author  = {den Boer, Arnoud V. and Meylahn, Janusz M. and Schinkel, Maarten Pieter},
  title   = {Artificial Collusion: Examining Supracompetitive Pricing by {Q}-learning Algorithms},
  journal = {Management Science},
  year    = {2024},
  note    = {Forthcoming / in press; Tinbergen Institute Discussion Paper No. 22-067/VII (2022)},
  doi     = {10.1287/mnsc.2024.08557}
}

@inproceedings{HartlineLongZhang2024,
  author    = {Hartline, Jason D. and Long, Sheng and Zhang, Chenhao},
  title     = {Regulation of Algorithmic Collusion},
  booktitle = {Proceedings of the 2024 Symposium on Computer Science and Law (CSLAW '24)},
  year      = {2024},
  pages     = {1--11},
  publisher = {ACM},
  address   = {Boston, MA, USA},
  doi       = {10.1145/3614407.3643706},
  note      = {arXiv:2401.15794}
}

@article{dolgopolov2024reinforcement,
  author  = {Dolgopolov, Arthur},
  title   = {Reinforcement learning in a prisoner's dilemma},
  journal = {Games and Economic Behavior},
  year    = {2024},
  volume  = {144},
  pages   = {84--103},
  doi     = {10.1016/j.geb.2024.01.004}
}

@article{WaltmanKaymak2008,
  author  = {Waltman, Ludo and Kaymak, Uzay},
  title   = {{Q}-learning agents in a {C}ournot oligopoly model},
  journal = {Journal of Economic Dynamics and Control},
  year    = {2008},
  volume  = {32},
  number  = {10},
  pages   = {3275--3293},
  doi     = {10.1016/j.jedc.2008.01.003}
}

@article{calvano2023genuine,
  author  = {Calvano, Emilio and Calzolari, Giacomo and Denicol{\`o}, Vincenzo and Pastorello, Sergio},
  title   = {Algorithmic collusion: Genuine or spurious?},
  journal = {International Journal of Industrial Organization},
  year    = {2023},
  volume  = {90},
  pages   = {102973},
  doi     = {10.1016/j.ijindorg.2023.102973}
}

@techreport{possnig2023reinforcement,
  author      = {Possnig, Clemens},
  title       = {Reinforcement Learning and Collusion},
  institution = {Department of Economics, University of Waterloo},
  year        = {2023},
  type        = {Working Paper},
  note        = {Dated 24 July 2023; revised as ``Monitoring, Market Primitives, and the Stability of Algorithmic Collusion'', R\&R at Theoretical Economics},
  url         = {https://uwaterloo.ca/economics/sites/default/files/uploads/documents/reinforcement-learning-and-collusion.pdf}
}

@article{bischiLamantia2022,
  author  = {Bischi, Gian Italo and Lamantia, Fabio},
  title   = {Evolutionary oligopoly games with cooperative and aggressive behaviors},
  journal = {Journal of Economic Interaction and Coordination},
  year    = {2022},
  volume  = {17},
  number  = {1},
  pages   = {3--27},
  doi     = {10.1007/s11403-020-00298-y}
}
```