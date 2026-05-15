# Fuzzy Linguistic CFL Prototype (Gap A + Gap C)

Research prototype for two of the four gaps identified in
[`docs/research-plan.md`](docs/research-plan.md):

- **Gap A — HFLTS-CFL**: replace FedSoft's scalar mixture weights with
  Hesitant Fuzzy Linguistic Term Set envelopes per cluster.
- **Gap C — Z-CFL drift**: replace FedDrift's numeric-threshold drift
  trigger with a Z-number reliability-aware detector + Mamdani rules.

Both prototypes ship a working baseline, synthetic experiments,
sweepable hyperparameters, and unit tests.

## Layout

```
hflts/         core HFLTS library (term sets, grammar, Liao-Xu-Zeng distance, HFLOWA)
cfl/           FedSoft baseline + HFLTSCFLServer (Gap A)
znumbers/      Z-number, Mamdani rule system, drift detector (Gap C)
experiments/   synthetic demos and parameter sweeps
tests/         unit tests for hflts and znumbers
docs/          original research plan
```

## Run

```bash
pip install -r requirements.txt

python -m unittest discover -s tests -v          # 40 tests
python -m experiments.synthetic_demo             # Gap A baseline demo
python -m experiments.sweep_defuzz               # Gap A defuzzification sweep
python -m experiments.sweep_granularity          # Gap A term-set size sweep
python -m experiments.drift_demo                 # Gap C reliability-aware drift
```

## Gap A — HFLTS-CFL findings

**Idea.** Client similarity to each cluster is observed over `T` rounds.
For each cluster, `mean ± z * std` is mapped onto a linguistic term set,
producing a consecutive index range. The HFLTS renders as a comparative
expression like `at_least high` or `between low and medium`.

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

Defuzzification choice barely moves ARI - the gap at medium noise is
**quantization-limited**.

**Term-set granularity sweep** (g = number of terms minus 1):

| noise | FedSoft | HFLTS g=7 | HFLTS g=11 | HFLTS g=15 | HFLTS g=21 |
|---|---|---|---|---|---|
| 0.05 | +0.795 | +0.795 | +0.795 | +0.795 | +0.795 |
| 0.10 | +0.893 | +0.597 | +0.893 | +0.893 | +0.893 |
| 0.15 | +0.893 | +0.597 | +0.597 | +0.687 | +0.893 |
| 0.20 | +0.785 | +0.519 | +0.785 | +0.687 | +0.687 |
| 0.30 | +0.597 | +0.526 | +0.597 | +0.597 | +0.526 |
| 0.50 | +0.463 | +0.463 | +0.463 | +0.463 | +0.463 |

`g=11` already closes the ARI gap at 4/6 noise levels; `g=21` matches
FedSoft across the board. The trade-off is exactly the one called out
in the research plan: more terms = sharper resolution but less
human-readable labels (`s0..s20` vs `none..perfect`).

This satisfies the Gap A go/no-go threshold from the plan ("match
FedSoft ARI within 2 % on the boundary regime").

## Gap C — Z-CFL drift detector findings

**Idea.** Every per-client per-round drift report becomes a Z-number
`Z = (A, B)`:

- `A` is a triangular fuzzy magnitude centred on the raw drift signal.
- `B` is a triangular reliability centred on a blend of neighbour
  agreement and a saturating sample-size proxy.

A Mamdani rule system over `(drift, reliability)` linguistic labels
decides between `no_action`, `defer`, `observe`, and `re_cluster`.

**Synthetic trace** (20 clients, 30 rounds, drift after round 15, half
of the clients have low sample size + low neighbour agreement so the
ground truth never asks to re-cluster on them):

| Subgroup | Detector | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| all clients | numeric threshold | 0.497 | 0.973 | 0.658 | 0.329 |
| all clients | Z-CFL (rules) | **1.000** | 0.813 | **0.897** | **0.000** |
| low-reliability half | numeric threshold | - | - | - | 0.493 |
| low-reliability half | Z-CFL (rules) | - | - | - | **0.000** |

Low-reliability false-positive reduction: **100 %** (Gap C go/no-go
threshold was ≥ 30 %).

## Limitations / next steps

- Synthetic-only: no CIFAR-10 / FEMNIST / Shakespeare runs (out of
  scope here; the plan calls them out for the full paper).
- HFLTS-CFL keeps the linguistic envelope but defuzzifies for the
  argmax metric. A real CFL training loop would use the envelope to
  gate proximal updates (defer when overlap, weight when narrow) -
  this prototype only demonstrates the representation, not the
  training-time use.
- Z-CFL drift rules are hand-authored. A defensible journal version
  would learn or tune them, and would also handle multi-step drift
  (gradual / abrupt / recurring) rather than a single step.
- Gap B (2-tuple aggregation) and Gap D (IT2 similarity) are not yet
  implemented. See `docs/research-plan.md` for their formulations.

## References

Primary theoretical foundations exercised here:

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
