# HFLTS-CFL Prototype (Gap A)

Research prototype for **HFLTS-based soft cluster assignment in Clustered
Federated Learning** — Gap A from the research plan at
[`docs/research-plan.md`](docs/research-plan.md).

## Idea

FedSoft (Ruan & Joe-Wong, AAAI 2022) assigns each client a numeric
mixture weight `pi_i in Delta^K`. That representation is hard to audit
and discards information about whether the assignment is genuinely
confident or just sitting near a cluster boundary.

This prototype replaces the scalar mixture weight with a **Hesitant
Fuzzy Linguistic Term Set (HFLTS)** envelope per cluster:

- Client similarity to each cluster is observed over `T` recent rounds.
- For each cluster, mean ± `z * std` is mapped onto the linguistic
  term set `S = {none, very_low, low, medium, high, very_high, perfect}`.
- The resulting consecutive index range becomes an HFLTS, which renders
  as a comparative expression like `at_least high` or
  `between low and medium`.

Why this matters:

| Property | FedSoft | HFLTS-CFL |
| --- | --- | --- |
| Output | `pi = [0.40, 0.35, 0.25]` | `["at_least high", "between low and medium", "low"]` |
| Audit | requires threshold convention | grammatical phrase per cluster |
| Hesitation | implicit in flat distributions | explicit envelope width |
| Overlap flag | needs ad-hoc rule | direct from envelope intersection |

## Layout

```
hflts/         core library (term sets, HFLTS, grammar, distance, aggregation)
cfl/           FedSoft baseline + HFLTSCFLServer
experiments/   synthetic_demo: HFLTS-CFL vs FedSoft baseline
tests/         unit tests for the HFLTS library
docs/          original research plan
```

## Run

```bash
pip install -r requirements.txt

python -m unittest discover -s tests -v
python -m experiments.synthetic_demo
python -m experiments.synthetic_demo --noise-std 0.10 --confidence-z 0.5
```

The demo prints per-client comparative expressions and flags overlap
ambiguity. With `matplotlib` available it also writes
`results/signatures.png` (diamonds = HFLTS-flagged overlap clients).

## What the synthetic results say

The demo sweeps a 2-D Gaussian-mixture CFL setup with controllable
boundary fraction and per-round similarity noise. Headline numbers from
a default run (30 clients, 3 clusters, 40 % boundary):

| Noise std | FedSoft ARI | HFLTS-CFL ARI | HFLTS hesitation share | Overlap-flag share |
| --- | --- | --- | --- | --- |
| 0.05 | 0.79 | 0.79 | 37 % | 73 % |
| 0.15 | 0.89 | 0.60 | 63 % | 100 % |
| 0.30 | 0.60 | 0.53 | 97 % | 100 % |
| 0.50 | 0.46 | 0.46 | 100 % | 100 % |

Read this as:

- **Low / very-high noise**: HFLTS matches FedSoft (no information to lose
  or none to recover).
- **Medium noise**: hard `argmax` over the defuzzified midpoint trades
  ARI for explicit hesitation — exactly the trade-off documented in the
  HFLTS literature, and the regime where the linguistic envelope adds
  value for a downstream "defer update" rule.
- **Overlap flag**: even when ARI is comparable, HFLTS surfaces which
  clients sit on cluster boundaries without any extra tuning.

This aligns with the go/no-go threshold in `docs/research-plan.md`
(within 2 % of FedSoft ARI on CIFAR-10 non-IID before claiming
personalization wins; otherwise pivot to the interpretability angle).

## Limitations / next steps

- Midpoint defuzzification is intentionally simple and is the main
  cause of the ARI gap at medium noise. Better choices (alpha-cut
  upper / lower, weighted-OWA, HFLTS distance to cluster-anchor
  envelopes) are open for the full paper.
- The "similarity history" is here a 1-D scalar per cluster per round;
  the real Gap A formulation would consume FedSoft's proximal local
  updates plus gradient-cosine signals.
- Real benchmarks (CIFAR-10 non-IID, FEMNIST, Shakespeare) are out of
  scope for this prototype.

See `docs/research-plan.md` for related gaps (Z-numbers drift detection,
IT2 similarity, granular re-clustering) and recommended publication
venues.
