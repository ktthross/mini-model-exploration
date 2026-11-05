# Plan: interpretable models on the solubility tiers

> **Superseded in part, 2026-11-04.** The tier framing misread the floor: 1.5 bits per measurement is a continuous-channel quantity, and binning to four tiers discarded ~0.5 bits before any model ran. `notebooks/11_continuous.ipynb` rescores the same models on logS itself (predictive density = replicate noise KDE ⊛ N(0, τ²) + ε wide, τ and ε calibrated on validation; score = h(Y) − cross-entropy). Floor 1.42 bits; repeat measurement 1.48 (97% of its population's floor); forest on 152 descriptors 1.28 random / 1.18 cluster (90% / 83%); EBMs 1.19–1.21 / 1.10–1.13; equations 0.3–0.4. Error bar of the best model ±1.75 logS at 90% vs ±1.48 for a measurement alone, with coverage verified. Sections below describe the tier work; GOSDT/CORELS remain tier-only.
>
> **Per-compound bars and confidence, 2026-11-04.** `notebooks/12_confidence.ipynb` calibrates (τ, ε) per quintile of the forest's tree spread: 1.37 bits (96% of floor) on random splits vs 1.25 uniform, 1.29 vs 1.15 on cluster, 0.07 vs −0.33 on learned; coverage holds per bin on random/cluster. Confidence c(x) = (h(Y) − h(q_x)) / (h(Y) − h(e)) clipped to [0, 1]; 0 = quoting the dataset's distribution, 1 = only measurement noise left. Forest-152 per compound: median 0.92, lowest quintile ≈ 0.25 (90% bar ±3 logS). Caveat: the pooled floor is conservative for easy compounds, whose replicate noise is narrower; a per-bin floor is the upgrade. Hand-off: `data/aqsoldb_predictions.csv`.

Status: planned 2026-11-03; environments ready 2026-11-03 (see prerequisites — all done). All four notebooks (`07_additive`, `08_equations`, `09_trees_and_rules`, `10_front`) executed on the project venv and read 2026-11-03; skills sections F–I added; sections J–M (splits, features, learning to split, kNN) backfilled 2026-11-03, entries 53–73. Headline: the front is readable up to 0.950 bits (20-pair EBM) against the forest's 1.009; an equation or tree of ~20 numbers carries ~0.6.

## Goal

Three model families that are readable by construction — symbolic equations (PySR), optimal sparse trees and rule lists (GOSDT, CORELS), and additive shape functions (InterpretML EBMs) — scored on exactly the same footing as the random forest, so each lands as a point on one axis the project already uses (**information in bits about the measured tier**) plus one new axis (**description length**: how much model it took). The forest on the 152 descriptors carried down from the features notebook is the reference at 1.009 bits (random CV) / 0.941 (cluster); a second measurement is 0.963 bits on the benchmark population.

The question each answers: *how many bits can a model a person can read in full carry, and what does each bit cost in model size?*

## Prerequisites

1. ~~Re-authenticate the JFrog index~~ Done. The 401 was a stale token embedded in the index URL of `~/.config/uv/uv.toml`, which takes precedence over `UV_INDEX_JFROG_PYPI_*`; the URL now carries no credentials and `.bashrc` sets the uv variables from `JFROG_USER`/`JFROG_TOKEN`.
2. ~~Install~~ Done: `interpret` 0.7.8 and `pysr` 2.7.2 (Julia runtime fetched) are in the main venv. **`gosdt` has no wheel for Python 3.13** (source build needs TBB + CMake) and **`corels` does not build on 3.13**.
3. ~~Side environment~~ Done: both solvers live in `tools/trees_and_rules/` (`setup.sh` rebuilds it from scratch: Python 3.9, `gosdt` 1.0.4 from the prebuilt wheel, `corels` 1.1.29 from a vendored sdist with two one-line patches, linked against brew's GMP, numpy pinned `<2` because the pre-generated Cython C++ predates numpy 2). The notebook drives it through `tools/trees_and_rules/.venv/bin/python` on a script that writes trees, rule lists and per-compound predictions to files; nothing from it is imported into the main kernel.
4. ~~Extract the scoring harness~~ Done: `scripts/aqsoldb_harness.py`, first consumer `06_knn.ipynb`.

## Shared protocol (identical for every method)

- **Data.** `data/aqsoldb_tiers.csv` (one row per vote, flagged votes weight 0 and set aside), `data/aqsoldb_features.csv` (130 descriptors), `data/aqsoldb_splits.csv` (random `cv_*`, cluster `cluster_*`), `data/aqsoldb_splits_ls.csv` (learned `ls_*`, core flag).
- **Fit** on the train side; **choose every tunable thing on the validation side** (λ, σ for regressors, tree depth or regularization, equation complexity); **score once on the test side**. Random CV for selection, cluster and LS as the harder audits — the same three-scheme table as the features and LS notebooks.
- **Score.** Information = H(T) − log-loss of the calibrated tier distribution; hard-tier accuracies alongside. Two populations: all compounds, and compounds with 2+ unflagged votes (benchmark population).
- **Description length.** Declare it per method before fitting (below), report it beside the information, and draw the information–description-length front across all methods including the forest (whose length is, honestly, "unreadable").
- **Inputs.** Two tiers of input sets for each method: ESOL's 4 (the readable baseline the forest scores 0.769 on) and the 130 descriptors; for PySR, additionally the top importance groups from the features notebook (MolLogP, the MolWt group's representative, NumAromaticRings group, PEOE_VSA6) because symbolic regression needs few inputs to produce something readable.

## Method notes

### Symbolic regression — PySR

- **Target.** Each compound's consensus logS (median of unflagged votes), vote-weighted so every compound counts once. Regression, not tiers: an equation for logS is the readable object.
- **Bridge to tiers.** A regressor gives a point, the harness needs a distribution. Use $p(\text{tier} \mid x) = \int_{\text{tier}} \mathcal{N}(y;\, f(x), \sigma^2)\,dy$ with σ chosen on the validation side by maximizing information — σ plays the role λ plays for the forests, and the same floor logic applies (σ ≥ some small value so no tier gets probability zero).
- **Reference.** Delaney's published ESOL equation, $\log S = 0.16 - 0.63\,c\log P - 0.0062\,MW + 0.066\,RB - 0.74\,AP$, uses the same four inputs. Score it through the same bridge *before* running PySR: it is the equation to beat, and it costs nothing to fit.
- **Description length.** Equation complexity as PySR counts it (nodes in the expression tree). Report the whole Pareto front PySR returns (loss vs complexity), scored by information on validation, then the chosen equation on test.
- **Settings to declare.** Operator set (+ − × ÷, and whether to allow exp/log/pow), `maxsize`, number of iterations, and the seed; multiple seeds to see whether the same structure recurs.
- **Reading.** Does PySR rediscover ESOL's structure (a logP term dominating, a size term with a negative sign)? How many bits does its best readable equation carry versus the forest on the same inputs?

### Optimal sparse decision trees — GOSDT

- **Labels.** GOSDT is binary. Use the two **Gray-code bits** from the tiers hand-off as two binary targets: a one-tier error flips exactly one bit, so this is the natural factorization. One optimal tree per bit; the tier distribution is the product of the two bit distributions (state the independence assumption; check it on validation by comparing with a direct 4-class alternative such as one-vs-rest if needed).
- **Features.** GOSDT needs binarized features: use its reference-model threshold guessing (a gradient-boosting model proposes thresholds) rather than hand-chosen cuts, and record how many thresholds survive.
- **Weights.** Verify whether the package accepts sample weights; if not, replicate rows in proportion to weight (or drop the 1/n weighting and say so — the first thing to check).
- **Description length.** Number of leaves (and depth), with the regularization parameter chosen on validation.
- **Reading.** The tree itself, printed. Which descriptors appear at the top? Does the first split land on MolLogP, as the importances predict?

### Rule lists — CORELS

- **Same framing as GOSDT**: one rule list per Gray bit, binarized features (reuse GOSDT's thresholds so the two methods see the same inputs).
- **Environment.** Side venv on Python 3.9; a small driver script writes `rules_bit_1.txt`, `rules_bit_2.txt` and the per-compound predictions as CSV; the notebook reads them and scores them through the harness. Exact optimality is the point of CORELS, so prefer this over `imodels`' approximations; use `imodels` only if the side venv proves unworkable.
- **Description length.** Number of rules, and total antecedent terms.

### Additive models — InterpretML EBM

- **Drop-in.** `ExplainableBoostingClassifier` does multiclass, accepts `sample_weight`, returns `predict_proba`: it scores through the harness unchanged.
- **Variants.** Main effects only, then with interactions (multiclass EBMs limit interactions — note the constraint and the version used).
- **Description length.** Number of features with a non-flat shape function, and total shape-function degrees of freedom (bins); with interactions, their count.
- **Reading.** The shape functions for MolLogP and the MolWt group are the direct counterpart of the permutation-importance table in the features notebook: the curve, not just the bits lost under shuffling. Also the first place to look at *how* solubility depends on size — monotone? saturating?

## Deliverables

- `scripts/aqsoldb_harness.py` — the shared scoring and evaluation code, with one small self-check.
- `notebooks/08_equations.ipynb` (PySR, with ESOL's equation as the reference row), `notebooks/09_trees_and_rules.ipynb` (GOSDT + CORELS), `notebooks/07_additive.ipynb` (EBM). Each ends with the same three-scheme table and a row in the shared front.
- `notebooks/10_front.ipynb` (or a final section of the last one): the information–description-length front across forest, kNN, PySR, GOSDT, CORELS, EBM, with the repeat benchmark as the horizontal reference.
- `docs/skills_from_notebooks.md` — new section per notebook, as before.

## Open questions to settle on the way (not now)

- Whether the Gray-bit factorization loses information relative to a direct 4-class model (measure it on validation with the forest first: train two per-bit forests, combine, compare with the 4-class forest's 1.006 bits).
- Whether the regression bridge's Gaussian is the right noise shape for an equation's residuals (the prep notebook found replicate noise heavy-tailed; a Student-t bridge with ν chosen on validation is the fallback).
- How to count description length comparably across families; the honest answer may be "report each family's own count and don't pretend they are commensurable," with the forest as "not readable."
