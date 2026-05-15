# Baseline implementation fidelity

This prototype implements five baselines alongside HFLTS-CFL. None
of them are line-for-line ports of the original papers - they are
all simplified so they can share the same NumPy logistic-regression
trainer with no PyTorch dependency. This document lists exactly
what was kept and what was dropped, so the comparison can be read
honestly.

## fedavg — McMahan et al., AISTATS 2017

**Original.** All clients share one global model; the server
averages local updates (FedAvg).

**This prototype.** Set `pi` uniformly across all K cluster slots at
init and keep it uniform every round. The aggregation step then
collapses every cluster centroid to the global average, which IS
FedAvg up to bookkeeping. Faithful in spirit.

## local — no FL baseline

**Original.** N independent local trainers, no aggregation.

**This prototype.** Skip `_aggregate_clusters`; route
`personalized_model()` directly to the latest `local_W`. Faithful.

## ifca — Ghosh, Chung, Yin, Ramchandran, NeurIPS 2020

**Original (Algorithm 1).** Each round, server broadcasts all K
cluster models. Each client picks `k̂ = argmin_k L_i(θ_k)` from its
local cross-entropy loss on cluster model `θ_k`. The server then
runs FedAvg within each chosen cluster set.

**This prototype.**

- We use `argmax(accuracy)` on the client's data as a proxy for
  `argmin(cross-entropy)`. The two agree on well-fit linear
  classifiers but can differ on noisy classifiers in early rounds.
- Single-round signal, no windowing - matches the paper.
- Server aggregation: weighted average with one-hot `pi` weights -
  identical to FedAvg-within-cluster up to floating-point order.

**Honest deviation.** Loss-vs-accuracy proxy may yield slightly
different selections in the first few rounds. We swept with both
forms (windowed and per-round) and the headline numbers move by
< 0.005 on synthetic.

## sattler — Sattler, Mueller, Samek, IEEE TNNLS 2021

**Original.** Top-down bipartitioning of a cluster when convergence
slows. Split decision uses:

1. Gradient norm tests `max_i ‖∇L_i‖ > ε_1` and `mean_i ‖∇L_i‖ > ε_2`
   to decide whether any client is still making real progress.
2. Pairwise cosine matrix `α_ij` over recent gradient updates.
3. The optimal bipartition `α^max_cross` is found via a search over
   the rows / columns of `α`; the split is committed when
   `α^max_cross < cos(γ_min)` for a threshold `γ_min`.

**This prototype.**

- We replace the `(ε_1, ε_2)` gradient-norm gate with a fixed warmup
  schedule (`sattler_warmup_rounds=5`, attempt every
  `sattler_split_every=3` rounds). Splits never "wait for
  convergence" in the original sense.
- We replace `α^max_cross` with a single threshold: split when the
  minimum off-diagonal cosine drops below
  `-sattler_split_threshold` (=0.2 by default).
- The actual bipartition uses the sign of the leading eigenvector
  of `α` (standard spectral relaxation of the 2-way cut). The paper
  uses an exact bipartition search; spectral is the textbook
  efficient approximation.

**Honest deviation.** This is the spirit of Sattler, not a port.
Real Sattler would be more conservative about splitting (gradient
norm gate prevents premature splits) but the split mechanism is
equivalent in principle. Differences may matter when the cohort
has not converged yet.

## fedsoft — Ruan & Joe-Wong, AAAI 2022

**Original.** Each client receives mixture weight `π_i` from the
server; clients run a proximal local update
`min L_i(θ) + (μ/2) Σ_k π_{i,k} ‖θ - θ_k‖²` and the server
re-estimates `π_i` from the optimum. Includes an EM-style
posterior update.

**This prototype.**

- Assignment step only: `π_i = softmax(mean_T sim_{i,k} / τ)` over a
  window of T similarity rounds.
- No proximal regularizer in the local update; clients just run
  vanilla gradient descent from their mixture model.
- Optional `defer_max_pi` fallback (added for the fairness audit -
  not in the paper) sets `π_i` uniform when no cluster reaches the
  threshold. Disabled by default.

**Honest deviation.** This is a stripped-down FedSoft that
isolates the assignment rule. The proximal term in the original
paper is part of what makes the soft assignment converge cleanly;
without it the cluster centroids can drift apart faster. The
comparison in this prototype is therefore "assignment rules sharing
a vanilla trainer", not "full FedSoft vs full HFLTS-CFL".

## hflts — this prototype (Gap A)

**Original.** N/A - this is the prototype being evaluated.

The HFLTS envelope is computed from
`mean ± confidence_z · std` of the similarity history, defuzzified
with one of four methods (`midpoint`, `midpoint_softmax`, `upper`,
`score_penalty`). The defer-on-overlap rule is an HFLTS-specific
fallback to a uniform mixture when at least two per-cluster
envelopes share `hflts_overlap_min` consecutive terms.

## What this means for the headline tables

Any "X beats Y by Δ" between methods that share a row in this
document carries the implicit caveat "in their stripped-down,
trainer-shared forms". The relative ordering on synthetic and
real tabular data may differ if either:

- A more faithful FedSoft (with proximal update) is wired in, or
- Real Sattler (with the convergence gate) is wired in, or
- The local trainer is upgraded from logistic regression to a
  multi-layer network where gradient cosine becomes more informative.

The HFLTS-CFL contribution stays the same regardless: an
interpretable per-client linguistic envelope rendered as
comparative expressions, plus the defer-on-overlap rule. The
question the comparison answers is "is this representation
competitive in a controlled, apples-to-apples assignment-step
study?" - not "does HFLTS-CFL win against production
implementations of FedSoft and Sattler?".
