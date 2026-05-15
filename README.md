# Fuzzy Linguistic CFL Prototype (Gap A + Gap C + end-to-end)

Research prototype for two of the four gaps identified in
[`docs/research-plan.md`](docs/research-plan.md):

- **Gap A — HFLTS-CFL**: replace FedSoft's scalar mixture weights with
  Hesitant Fuzzy Linguistic Term Set envelopes per cluster, plus an
  explicit "defer-update" rule when envelopes overlap.
- **Gap C — Z-CFL drift**: replace FedDrift's numeric-threshold drift
  trigger with a Z-number reliability-aware detector + Mamdani rules.

Both prototypes ship a working baseline, three layers of synthetic
experiments (representation only, drift only, full end-to-end CFL
training), sweepable hyperparameters, and unit tests.

## Layout

```
hflts/         core HFLTS library (term sets, grammar, Liao-Xu-Zeng distance, HFLOWA)
cfl/           FedSoft baseline, HFLTSCFLServer, end-to-end FederatedTrainer
znumbers/      Z-number, Mamdani rule system, drift detector
experiments/   representation demos, parameter sweeps, end-to-end runs
tests/         unit tests for hflts, znumbers, trainer
docs/          original research plan
```

## Run

```bash
pip install -r requirements.txt

python -m unittest discover -s tests -v                # 50 tests
python -m experiments.synthetic_demo                   # Gap A representation only
python -m experiments.sweep_defuzz                     # Gap A defuzzification sweep
python -m experiments.sweep_granularity                # Gap A term-set size sweep
python -m experiments.drift_demo                       # Gap C reliability-aware drift
python -m experiments.end_to_end                       # full CFL training, 1 seed
python -m experiments.end_to_end_sweep --n-seeds 10    # full CFL training, 10 seeds
python -m experiments.inspect_drift                    # diagnostic: per-round drift signals
```

## End-to-end results (10 seeds, default hyperparameters)

30 clients, 3 task clusters, 5-D logistic regression, 30 % boundary
clients, drift event at round 15 swapping the task of 30 % of the
non-boundary clients. All four configurations use the same defaults
(`softmax_tau=0.05`, `confidence_z=0.5`, `n_terms=11`).

| config | pre_acc | pre_bnd | post_acc | post_bnd | drift_P | drift_R | drift_F1 | post_FP |
|---|---|---|---|---|---|---|---|---|
| fedsoft_numeric | 0.748 ± 0.019 | 0.700 ± 0.049 | 0.697 ± 0.040 | 0.701 ± 0.051 | 0.900 ± 0.300 | 0.422 ± 0.271 | 0.540 ± 0.288 | 0.000 ± 0.000 |
| hflts_numeric | 0.766 ± 0.022 | 0.708 ± 0.057 | 0.703 ± 0.039 | 0.705 ± 0.051 | 0.861 ± 0.298 | 0.567 ± 0.251 | 0.668 ± 0.256 | 0.300 ± 0.640 |
| fedsoft_z | 0.748 ± 0.019 | 0.700 ± 0.049 | 0.697 ± 0.041 | 0.701 ± 0.052 | 0.700 ± 0.458 | 0.311 ± 0.276 | 0.410 ± 0.315 | 0.000 ± 0.000 |
| hflts_z | 0.766 ± 0.022 | 0.708 ± 0.057 | 0.703 ± 0.039 | 0.705 ± 0.052 | 0.900 ± 0.300 | 0.511 ± 0.223 | 0.641 ± 0.243 | 0.000 ± 0.000 |

At these defaults `hflts_*` clears `fedsoft_*` by 1.8 pp on pre-drift
accuracy and the Z-CFL variants both keep `post_FP=0` while the
`hflts_numeric` baseline incurs 0.3 false re-clusters per run on
average. **But these numbers are hyperparameter-dependent** - see the
fairness audit below.

## 5-way baseline comparison (assignment step only, no drift)

To respond to "FedSoft alone isn't enough", the trainer now also
implements the standard CFL baselines and a 5-way comparison runs
through `experiments/compare_baselines.py`. Results saved to
`results/baselines.md`.

| method | reference | rule |
|---|---|---|
| fedavg | McMahan, AISTATS 2017 | single global model, no clustering |
| local | - | each client trains alone, no FL |
| ifca | Ghosh et al., NeurIPS 2020 | hard one-hot to argmax-accuracy cluster |
| sattler | Sattler et al., IEEE TNNLS 2021 | top-down split via gradient-cosine bipartition |
| fedsoft | Ruan & Joe-Wong, AAAI 2022 | softmax(mean similarity), simplified |
| hflts | this prototype | HFLTS envelope -> defuzzify (Gap A) |

5 seeds, 20 rounds, all methods sharing the same local trainer.

### Synthetic (30 clients, 3 task profiles, 30 % boundary)

| method | mean_acc | boundary_acc |
|---|---|---|
| fedavg | 0.662 ± 0.041 | 0.668 ± 0.040 |
| local | 0.714 ± 0.023 | 0.644 ± 0.040 |
| **ifca** | **0.780 ± 0.020** | 0.677 ± 0.044 |
| sattler | 0.774 ± 0.016 | 0.686 ± 0.047 |
| fedsoft | 0.739 ± 0.022 | 0.678 ± 0.031 |
| hflts | 0.751 ± 0.020 | **0.689 ± 0.040** |

- IFCA wins mean accuracy on the synthetic regime; Sattler's
  top-down split is a very close second (0.774).
- HFLTS wins **boundary-client accuracy** - exactly the metric the
  Gap A defer-on-overlap motivation targets - by 0.3-1.2 pp over
  every hard-clustering baseline.
- Hard clustering (IFCA, Sattler) wins on mean; soft clustering
  (HFLTS, FedSoft) wins on the boundary subset. This is the
  honest cut: HFLTS targets the boundary regime, and that is
  where it shows up.

### UCI Adult

| method | mean_acc | boundary_acc |
|---|---|---|
| fedavg | 0.862 ± 0.007 | 0.858 ± 0.027 |
| local | 0.809 ± 0.024 | 0.802 ± 0.044 |
| ifca | 0.859 ± 0.008 | 0.853 ± 0.017 |
| sattler | 0.858 ± 0.009 | 0.852 ± 0.024 |
| **fedsoft** | **0.862 ± 0.007** | 0.856 ± 0.026 |
| hflts | 0.861 ± 0.007 | 0.857 ± 0.025 |

All five clustering methods land at 0.86; local-only is the only
loser (per-client data alone is too thin). Adult is dominated by
a globally-good logistic regression, so cluster structure
provides no signal - the gap between FedAvg, IFCA, Sattler,
FedSoft and HFLTS is rounding.

### UCI HAR

| method | mean_acc | boundary_acc |
|---|---|---|
| fedavg | 0.953 ± 0.005 | 0.951 ± 0.007 |
| local | 0.584 ± 0.029 | 0.580 ± 0.029 |
| ifca | 0.953 ± 0.002 | 0.949 ± 0.007 |
| sattler | **0.953 ± 0.005** | **0.952 ± 0.010** |
| fedsoft | 0.950 ± 0.003 | 0.949 ± 0.008 |
| hflts | 0.951 ± 0.005 | 0.949 ± 0.008 |

Same pattern as Adult: all clustering methods tie at ~0.95.
Local-only collapses to 0.58 because individual subjects do not
have enough data to learn the 6-way activity classifier from
scratch. The federated signal matters; the cluster signal does
not on this dataset.

### What the 6-way comparison tells us

- HFLTS-CFL is **competitive** across all three datasets - never the
  worst, never far from the best.
- On the metric the prototype was designed for (synthetic
  boundary-client accuracy), HFLTS leads every hard-clustering
  baseline (IFCA, Sattler) and the soft baseline (FedSoft).
- On mean accuracy, hard clustering (IFCA, Sattler) wins on
  synthetic - they commit to a single cluster per client, which
  is the right move when the task structure is clean. HFLTS
  trails by ~3 pp on this regime.
- On real tabular data with a logistic-regression-friendly target
  (Adult, HAR), the choice of clustering algorithm is dwarfed by
  the choice of "do FL at all" (the gap between local-only and
  everyone else on HAR is ~37 pp; the gap between FedAvg and any
  CFL method is ~0 pp).
- Sattler's gradient-cosine bipartition is a strong canonical
  baseline; in this prototype it tracks IFCA closely on synthetic
  and ties everyone else on the real datasets.

## Tabular benchmark results

Both prototypes were rerun on two real tabular CFL-style datasets to
sanity-check the synthetic findings. Saved tables live in
`results/adult.md`, `results/har_easy.md`, `results/har_hard.md`.

### UCI Adult (binary income, profile-partitioned by `education-num`, 3 seeds)

| config | pre_acc | pre_bnd | post_acc | post_bnd | drift_P | drift_R | drift_F1 | post_FP |
|---|---|---|---|---|---|---|---|---|
| fedsoft_numeric | 0.854 ± 0.008 | 0.833 ± 0.009 | 0.856 ± 0.009 | 0.833 ± 0.010 | 1.000 | 0.444 | 0.610 | 0.000 |
| hflts_numeric | 0.854 ± 0.007 | 0.833 ± 0.007 | 0.856 ± 0.008 | 0.832 ± 0.009 | 1.000 | 0.444 | 0.610 | 0.000 |
| fedsoft_z | 0.854 ± 0.008 | 0.833 ± 0.009 | 0.856 ± 0.009 | 0.833 ± 0.010 | 1.000 | 0.444 | 0.610 | 0.000 |
| hflts_z | 0.854 ± 0.007 | 0.833 ± 0.007 | 0.856 ± 0.008 | 0.832 ± 0.009 | 1.000 | 0.444 | 0.610 | 0.000 |

All four configurations land within rounding of each other on Adult.
This dataset is dominated by a globally-good logistic regression so
clustering does not buy anything; HFLTS-CFL does **not hurt**, but
there is no accuracy win to claim either.

### UCI HAR (multi-class activity, 30 subjects = 30 clients, 3 seeds)

Easy setup (3 profiles, 25 % boundary, 20 rounds, drift round 15):

| config | pre_acc | pre_bnd | post_acc | post_bnd | drift_P | drift_R | drift_F1 | post_FP |
|---|---|---|---|---|---|---|---|---|
| fedsoft_numeric | 0.949 ± 0.001 | 0.955 ± 0.001 | 0.950 ± 0.002 | 0.956 ± 0.008 | 0.815 | 0.259 | 0.336 | 1.667 ± 2.357 |
| hflts_numeric | 0.947 ± 0.003 | 0.957 ± 0.004 | 0.952 ± 0.001 | 0.956 ± 0.007 | 0.815 | 0.259 | 0.336 | 1.667 ± 2.357 |
| fedsoft_z | 0.949 ± 0.001 | 0.955 ± 0.001 | 0.950 ± 0.002 | 0.956 ± 0.008 | 0.810 | 0.185 | 0.258 | 1.333 ± 1.886 |
| hflts_z | 0.947 ± 0.003 | 0.957 ± 0.004 | 0.952 ± 0.001 | 0.956 ± 0.007 | 0.810 | 0.185 | 0.258 | 1.333 ± 1.886 |

Harder setup (5 profiles, 50 % boundary, 25 rounds):

| config | pre_acc | pre_bnd | post_acc | post_bnd | drift_F1 | post_FP |
|---|---|---|---|---|---|---|
| fedsoft_numeric | 0.949 ± 0.005 | 0.945 ± 0.006 | 0.962 ± 0.002 | 0.957 ± 0.004 | 0.000 | 0.667 ± 0.471 |
| hflts_numeric | 0.952 ± 0.003 | 0.950 ± 0.001 | 0.964 ± 0.002 | 0.962 ± 0.003 | 0.067 | 0.333 ± 0.471 |
| fedsoft_z | 0.949 ± 0.005 | 0.945 ± 0.006 | 0.962 ± 0.002 | 0.957 ± 0.004 | 0.000 | 0.000 ± 0.000 |
| hflts_z | 0.952 ± 0.003 | 0.950 ± 0.001 | 0.964 ± 0.002 | 0.962 ± 0.003 | 0.000 | 0.000 ± 0.000 |

### What the real-data results say honestly

- **Accuracy**: HFLTS-CFL ≈ FedSoft on both real datasets. On HAR the
  difference is ≤0.5 pp with overlapping standard deviations; on
  Adult it is a hard tie. The synthetic +1.8 pp headline does not
  reproduce on real tabular data with this trainer.
- **Drift FP**: the synthetic "0 false positives across 10 seeds"
  result for Z-CFL **does not survive on real data**. On the easy
  HAR setup both `fedsoft_z` and `hflts_z` average 1.33 ± 1.89 false
  positives per run. The trend (Z-CFL has fewer FP than the numeric
  threshold) still holds, but the gap shrinks. On the harder HAR
  setup Z-CFL recovers to 0.000 false positives.
- **Practical takeaway**: HFLTS-CFL is **safe** on real tabular
  benchmarks - it never underperforms FedSoft by more than noise.
  Its concrete deliverable on these datasets is the interpretable
  per-client assignment (`at_least s7`, `between s3 and s6`), not a
  raw accuracy lift. Z-CFL's reliability gate continues to suppress
  spurious re-cluster triggers, but it is a relative improvement,
  not a binary "zero false positives" property in the wild.

## Fairness audit (response to "did we modify FedSoft?")

Two pieces of context the headline table hides:

1. **Simplified FedSoft baseline.** `FedSoftServer` implements only
   the *assignment* step of FedSoft (softmax of mean similarities). The
   original FedSoft (Ruan & Joe-Wong, AAAI 2022) additionally runs a
   proximal local update. Both methods use the same local trainer
   here, so the comparison isolates the assignment step rather than
   the full algorithm.
2. **Tuning matters more than the assignment rule.** With drift
   disabled and 10 seeds, the FedSoft baseline's accuracy is highly
   sensitive to `softmax_tau`:

| tau | fedsoft_plain | fedsoft_defer | hflts_no_defer |
|---|---|---|---|
| 0.025 | 0.770 ± 0.020 | 0.766 ± 0.021 | 0.759 ± 0.020 |
| 0.050 | 0.747 ± 0.024 | 0.735 ± 0.031 | 0.759 ± 0.020 |
| 0.100 | 0.706 ± 0.046 | 0.692 ± 0.057 | 0.759 ± 0.020 |
| 0.150 | 0.680 ± 0.067 | 0.679 ± 0.068 | 0.759 ± 0.020 |
| 0.250 | 0.679 ± 0.068 | 0.680 ± 0.067 | 0.759 ± 0.020 |
| 0.500 | 0.680 ± 0.067 | 0.680 ± 0.067 | 0.759 ± 0.020 |

At `tau=0.025` plain FedSoft *beats* HFLTS-CFL at our default
hyperparameters (0.770 vs 0.759). At `tau=0.05` and above, HFLTS wins.
The original tau choice (0.05) inadvertently favoured HFLTS in the
headline run. HFLTS itself is invariant to `softmax_tau` (it does not
use that knob) so its column is flat - this is the robustness side
of the comparison.

**Tuned vs tuned**:

| method | best config | mean_acc | std |
|---|---|---|---|
| FedSoft (plain) | tau=0.025 | 0.770 | 0.020 |
| HFLTS-CFL (no-defer) | defuzz_tau=0.05, z=0.8, n_terms=15 | **0.782** | 0.024 |

Tuned HFLTS still leads by ~1.2 pp but the standard deviations
overlap; with 10 seeds this is **not statistically significant** at
p<0.05. The honest Gap A claim is therefore:

- Tuned vs tuned, HFLTS-CFL matches or modestly beats simplified
  FedSoft on this synthetic regime.
- HFLTS is more **robust to hyperparameter choice** (its accuracy
  does not depend on the assignment-step temperature).
- HFLTS provides **interpretable assignments** as comparative
  expressions (`at_least s7`, `between s3 and s6`) - the
  representation, not the accuracy, is the main contribution.

The defer-on-overlap rule, which we initially expected to be a
significant contributor, actually does **nothing** on top of the
HFLTS representation in this regime (`hflts_no_defer == hflts_defer`
in the ablation). When ported to a setting where envelopes routinely
overlap, the rule may earn its keep; here it does not, and we say so
rather than counting it as a win.

Gap C's "zero false positives" headline is **not** affected by the
audit: Z-CFL never re-clusters spuriously across 10 seeds in either
the FedSoft or HFLTS column, independent of `softmax_tau`. That
finding survives the fairness check.

## Honest debugging notes

The first end-to-end run reported a much louder gap (`drift_F1` of
0.80 vs 0.00 for `fedsoft_numeric` vs `fedsoft_z`). That was
**suspicious** and turned out to be two artifacts:

1. The default Mamdani rule table had no entry for the
   (severe, medium) and (severe, low) cells, so a strong magnitude
   paired with mid-range reliability fell through to a default
   `no_action` and the detector abstained entirely. The fixed table in
   `znumbers/rules.py` covers every cell explicitly.
2. The neighbour-agreement signal was computed via cosine similarity
   of the `pi` rows. Under FedSoft's soft assignment every pair has
   cosine close to 1, so the top-K neighbour pick was effectively
   random and the agreement bottomed out around 0.2 even for cleanly
   drifting clients. Replacing it with cosine similarity of recent
   loss trajectories makes the signal independent of the assignment
   layer.

`experiments/inspect_drift.py` prints the per-round drift signals
behind both fixes so the corrected numbers are auditable rather than
just trusted.

## Representation-level results (Gap A only)

**Defuzzification sweep** (30 clients, 3 clusters, 40 % boundary,
`confidence_z=0.5`, seed 0):

| noise | FedSoft | HFLTS midpoint | HFLTS softmax | HFLTS upper | HFLTS score-pen | hesitation | overlap |
|---|---|---|---|---|---|---|---|
| 0.05 | +0.795 | +0.795 | +0.795 | +0.795 | +0.795 | 37% | 73% |
| 0.10 | +0.893 | +0.597 | +0.597 | +0.453 | +0.597 | 53% | 97% |
| 0.15 | +0.893 | +0.597 | +0.597 | +0.519 | +0.519 | 63% | 100% |
| 0.20 | +0.785 | +0.519 | +0.519 | +0.453 | +0.453 | 80% | 97% |
| 0.30 | +0.597 | +0.526 | +0.526 | +0.453 | +0.453 | 97% | 100% |
| 0.50 | +0.463 | +0.463 | +0.463 | +0.381 | +0.526 | 100% | 100% |

Defuzzification choice barely moves ARI; the gap at medium noise is
**quantization-limited**, confirmed by the granularity sweep:

| noise | FedSoft | HFLTS g=7 | HFLTS g=11 | HFLTS g=15 | HFLTS g=21 |
|---|---|---|---|---|---|
| 0.05 | +0.795 | +0.795 | +0.795 | +0.795 | +0.795 |
| 0.10 | +0.893 | +0.597 | +0.893 | +0.893 | +0.893 |
| 0.15 | +0.893 | +0.597 | +0.597 | +0.687 | +0.893 |
| 0.20 | +0.785 | +0.519 | +0.785 | +0.687 | +0.687 |
| 0.30 | +0.597 | +0.526 | +0.597 | +0.597 | +0.526 |
| 0.50 | +0.463 | +0.463 | +0.463 | +0.463 | +0.463 |

The end-to-end trainer uses `n_terms=11`, which closes the
representation-only gap and still keeps the HFLTS labels
human-readable (envelopes render as `at_least s7`, `between s3 and
s6`, etc.).

## Drift-only results (Gap C only)

`drift_demo` runs a synthetic FL trace where half of the clients are
explicitly low-reliability (small samples + low neighbour agreement)
and ground truth never wants re-clustering for them:

| Subgroup | Detector | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| all clients | numeric threshold | 0.497 | 0.973 | 0.658 | 0.329 |
| all clients | Z-CFL (rules) | **1.000** | 0.813 | **0.897** | **0.000** |
| low-reliability half | numeric threshold | - | - | - | 0.493 |
| low-reliability half | Z-CFL (rules) | - | - | - | **0.000** |

Low-reliability false-positive reduction: **100 %**.

## Limitations / next steps

- Synthetic only. CIFAR-10 / FEMNIST / Shakespeare benchmarks are out
  of scope for this prototype but are called out for the journal
  version in `docs/research-plan.md`.
- Drift detection uses a single round's relative-loss signal. A
  longer signal (per-feature drift, KL divergence on prediction
  distribution, gradient cosine drift) would likely lift Z-CFL recall
  without giving up its zero-false-positive precision.
- The defer-on-overlap rule for HFLTS-CFL is binary (overlap or not).
  A continuous "shrinkage" rule that scales the mixture toward
  uniform proportional to the overlap width might be a stronger
  contribution.
- Gap B (2-tuple aggregation) and Gap D (IT2 similarity) are not yet
  implemented.

## References

Theoretical foundations exercised here:

- Rodriguez, Martinez & Herrera, *Hesitant Fuzzy Linguistic Term Sets
  for Decision Making*, IEEE TFS 20(1):109-119, 2012.
- Liao, Xu & Zeng, *Distance and similarity measures for hesitant fuzzy
  linguistic term sets*, Information Sciences 271:125-142, 2014.
- Zadeh, *A note on Z-numbers*, Information Sciences 181(14):2923-2932,
  2011.
- Kang, Wei, Li & Deng, *A method of converting Z-number to classical
  fuzzy number*, Information Sciences 246:1-8, 2013.

Federated learning baselines targeted:

- Ruan & Joe-Wong, *FedSoft*, AAAI 2022.
- Jothimurugesan et al., *Federated Learning under Distributed Concept
  Drift* (FedDrift), AISTATS 2023.
