# RANCH under exact inference — model card

`ranch` v0.1.0 · branch `exact-inference-reboot` · tables of record: jobs 44027063 (2026-09-17: infants,
deterministic configurations) and 44122930 (2026-09-18/20: adults) · companion note: *RANCH under exact
inference*, working note v3.2 (2026-09-20)

## What this is

RANCH is a rational-learner model of habituation and dishabituation (Raz, Cao, Saxe & Frank; eLife reviewed
preprint). A learner watches a stimulus through noisy perceptual samples, infers a Gaussian concept and its
own perceptual noise, and keeps looking in proportion to how informative the next sample is expected to be.
This package computes that model **exactly** — the published implementation approximated its inference on a
five-point random grid, and that approximation turned out to be load-bearing — and it contains the pipeline
that regenerates every number in the companion note from the public data.

Per embedding dimension (three principal components of ResNet-50 embeddings, treated independently):

```
(mu, sigma^2) ~ Normal-inverse-gamma(mu0 = 0, nu, alpha, beta)      the concept
y_k | mu, sigma^2 ~ N(mu, sigma^2)                                  the exemplar shown on presentation k
z_kt | y_k ~ N(y_k, eps^2)                                          the t-th perceptual sample of it
```

The learner infers `eps` (prior N(0.001, sd_eps), truncated to a box); the world draws samples with noise
`sigma_true`. Exemplars and the concept mean integrate out in closed form; the remaining posterior over
(sigma^2, eps) is a fixed quadrature (`ranch.settings.QUADRATURE`). Looking stops by a Luce rule,
P(look away) = w / (value + w), on a decision variable.

## The named configurations (`ranch.canonical`)

| name | world | learner's noise | decision variable | status |
|---|---|---|---|---|
| `CANONICAL` | noisy (`sigma_true` .2 infants, .1 adults) | inferred | **concept EIG**, I(z'; mu, sigma^2 \| data), eps a nuisance — the paper's own equation | the model of record |
| `PUBLISHED_CORRECTED` | noiseless | fixed | realized gain (the published quantity) | the published implementation made exact; coherent, but not the model the paper describes |
| `PUBLISHED_SPEC` | noiseless | inferred | realized gain | the published *specification* under exact inference: degenerate (eps collapses on identical samples; about one sample per stimulus) |
| `TOTAL_EIG_REFERENCE` | noisy | inferred | total EIG, information about (mu, sigma^2, eps) | fails: a familiar stimulus is as informative about eps as a novel one |

Two coherent forms exist — `CANONICAL` and `PUBLISHED_CORRECTED`. What the model is **not**: the published
code's quantity is a *realized* information gain evaluated at the true stimulus, not an expected one, and the
published behaviour depended on the coarse grid.

Canonical parameters: prior nu = 3, alpha = 1, beta = .1 and sd_eps = 1 in both populations; infants
sigma_true = .2, w = 1e-5; adults sigma_true = .1, w = 3.2e-5. Other registered decision variables: total EIG
(`mi`), KL between successive posteriors (`kl`), surprisal (`surprisal_b`), the published realized gain
(`eig_code`).

## Evaluation

Squared correlation between model and human condition means (invariant to the affine map from samples to
seconds, LT = a + b x samples, b >= 0), with 95% Monte-Carlo intervals; selection by the paper's rule (best
cross-validated RMSE on Experiment 1), every parameter then carried unchanged to Experiment 2.

| | concept EIG | published model, rescored identically |
|---|---|---|
| Infants, Exp. 1 (15 conditions; 93 infants) | .75 [.72, .78] | .74 |
| Adults, Exp. 1 (21 conditions; 470 adults) | .89 [.85, .91] | .87 |
| Infants, Exp. 2 (5 violation types) | .51 [.46, .56] | .66 |
| Adults, Exp. 2 (18 conditions) | .78 [.71, .81] | .72 |

Predicted ordering in Exp. 2, both populations: animacy > identity > number > pose. In the same noisy world
the other decision variables give, for infants / adults in Exp. 1: surprisal .58 / .83 (adult interval
[.68, .86]), KL .23 / .30, total EIG .23 / .23, published realized gain .08 / .22.

Protocol (the note's §4). Infants: the grid's best cell re-evaluated on 32 fresh rollouts of each of the 480
stimulus sequences; the stopping time is integrated analytically. Adults: a grid (96 stimulus pairs x 1
rollout per cell) shortlists the 10 best cells per variable and rule; these are re-evaluated on **every**
stimulus pair of the experiment (1180 pairs x 4 rollouts in Exp. 1; 845 pairs x 12 in Exp. 2); the selected
cell is reported on independent seeds. Error bars in the figures are standard errors over stimulus units.

## Limitations

- **Amplitude.** In its own units the model habituates and dishabituates less than people do (infants: familiar
  looking falls to .87 of its initial value against .57; adults .85 against .64), and the fitted affine map
  stretches the pattern. R^2 is blind to this. For adults the amplitude depends on the prior on the concept
  variance: at beta = .01 the model's amplitudes are close to the adults' (.68 / 1.33 against .64 / 1.31) at
  R^2 = .86 (exploratory extension; tables `*_selfcons_beta.csv`).
- **The adults' first trial.** Familiar looking falls by 8% after the first trial in model units; adults' falls
  by 31%. Neither a weaker prior on the mean (nu < 1) nor a tighter one on the variance fixes it. The same gap
  appears in Exp. 2 as too little looking on trial 1.
- **One fitted parameter more than the published model**: the world's sample noise, fitted per population.
- **Selection on the scored data**, as in the paper; Exp. 2 is the out-of-sample test. Infant Exp. 2 has five
  condition means and uses the six stimulus pairs per type of the published analysis.
- **Scope.** Two experiments, one stimulus set (218 stimuli, three principal components of ResNet-50
  embeddings, mu0 = 0: the model is not translation-invariant in embedding space). Parameter values are fitted
  constants, not measurements of infants' or adults' perception. Not a model of individual participants.
- **Numerics are part of the model's behaviour.** Monte-Carlo noise attenuates R^2; a coarse quadrature over
  eps biased the adult fits; six stimulus pairs were not the experiment's 1180. Each of these changed reported
  numbers before it was found (see Corrections). New configurations should be checked against a finer eps
  axis on identical seeds (`sherlock/diagnostics/`).

## Corrections and retractions

1. **The July 2026 note** (first exact-inference analysis): an infant data-handling bug (exposure-phase rows
   included; a non-unique subject key) invalidated its within-subject result and its infant fits; its claim
   that a proper EIG gives zero dishabituation was wrong (the sigma^2 channel is distance-sensitive); its
   "corrected model" used a functional that was not the published code's. Corrected in the August audit (v2).
2. **Retracted 2026-09-15:** "a noisy world revives the described model with the total EIG". The result came
   from a formula slip in our own closed-form EIG (1/2 log E[v] where the entropy needs E[1/2 log v]); fixed,
   the total EIG fails (R^2 .23). The concept EIG — the paper's equation, with eps a nuisance — is what works.
3. **Numerical corrections to the adult results:** v3 (2026-09-17, morning) under-reported them from too few
   rollouts; v3.1 computed them on a 30-node linear eps axis (+.03 in R^2) and on six stimulus pairs (-.06,
   irregular curves, a compressed difference between pose and identity violations); v3.2 is the regeneration on 120 log-spaced nodes and every
   stimulus pair. Infant results were unaffected throughout (the infant cell moves by .005 on the finer axis).

## Reproducing the results

```bash
# layout: <root>/RANCH_model (this repository), <root>/pkbb_paper_writing, <root>/RANCH_cluster  (or set RANCH_ROOT)
pip install -r requirements-lock.txt && pip install -e . --no-deps     # the record's environment (Python 3.12)
ranch check                                  # the 15 input files hash as pinned in granch_fast/tests/data_manifest.json
cd granch_fast && python -m pytest           # fast suite (~110 tests); `-m slow` for the statistical ones
sbatch sherlock/pipeline.sbatch              # every table: about 6 h (infants, deterministic) + 2 days (adults) on 24 cores
python -m ranch figures && Rscript granch_fast/plot_paper_panels.R     # the note's figure data and figures
```

Every stochastic stage takes explicit seeds (documented in `ranch.pipeline` / `ranch.selection`); golden
tests pin the tables of record; the pipeline was verified bit-for-bit against the scripts that produced the
earlier results (`sherlock/logs/step3f_verdict_2026-09-17.txt`). The suite passes on Python 3.12 / numpy
1.26 (the cluster, where the record was computed) and on Python 3.13 / numpy 2.5 (a laptop). Continuous integration runs the fast suite on
every push (`.github/workflows/tests.yml`). Install editably: the pipeline reads and writes tables next to the
code. Exploratory scripts at the top level of `granch_fast/` (outside the package and the pipeline) predate
the package and are kept for the record only.

## Data and licence

Input data are read from two public repositories at pinned commits — `GalRaz/pkbb_paper_writing` (human
looking times, the published model's plot data) and `anjiecao/RANCH_cluster` (stimulus pairs) — and the
stimulus embeddings in this repository. The human data belong to the paper's authors. **No licence has been
chosen yet for this repository or the data repositories; until one is, reuse needs the authors' permission.**

## Provenance

The exact-inference analysis, this package and the companion note were produced by an AI assistant (Claude,
Anthropic) working with Michael C. Frank between July and September 2026; the first analysis (July) was audited
independently in August, which corrected it (above); the companion note lists three derivations that a human should still re-derive independently. Contact:
Michael C. Frank. Please cite the paper (Raz, Cao, Saxe & Frank) and this repository.
