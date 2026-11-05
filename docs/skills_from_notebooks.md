# Skills used in the notebooks

A study sheet, one entry per technique. Each is self-contained: what the skill is, where it appears, why it mattered there, what to watch for, and a source to read further. Sections A–D come from `01_prep_and_analysis.ipynb`; section E from `02_tiers.ipynb`; F–I from the interpretable-model notebooks and the front; J–M (added last, so numbered last) from the four notebooks between them in pipeline order: `03_splits`, `04_features`, `05_learning_to_split`, `06_knn`. The process skills (D, and the last entries of E) are the ones that generalize furthest beyond chemistry.

---

## A. Information theory

### 1. Shannon entropy as "average yes/no questions"
**What.** $H = -\sum_k p_k \log_2 p_k$ is the average number of bits needed to state which of $K$ classes occurred; $\log_2 K$ is the worst case (all classes equally likely).
**Where.** Goal cell: the 90/5/4/1 toy giving 0.605 bits.
**Why.** Sets the unit everything else is measured in, and gives the ceiling that later tier-based scores are compared against.
**Watch for.** Entropy is about the *distribution*, not any one outcome; a skewed distribution needs fewer bits than a flat one even with the same $K$.
**Source.** Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal 27. Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), Ch. 2.

### 2. The Gaussian-channel floor, $\tfrac{1}{2}\log_2(\operatorname{Var}(Y)/\operatorname{Var}(\text{noise}))$
**What.** For $Y = \mu + e$ with Gaussian $\mu$ and $e$, the mutual information $I(\mu;Y) = \tfrac12 \log_2(1 + \text{SNR})$, and $\operatorname{Var}(Y) = \operatorname{Var}(\mu) + \operatorname{Var}(e)$ makes it $\tfrac12\log_2(\operatorname{Var}(Y)/\operatorname{Var}(e))$. It answers "how many distinguishable levels does a measurement resolve?" ($2^{\text{bits}}$).
**Where.** Goal cell; used as the reference computation in "The floor, recomputed without a family".
**Why.** The cleanest possible floor, and the right thing to *start* from — but it assumes a family.
**Watch for.** Exact only when both signal and noise are Gaussian. With heavy-tailed noise the variance is dominated by a few extreme values and the formula misleads; the notebook shows an 84-vote (< 1%) gross-error population moving it by more than 0.1 bits.
**Source.** Cover & Thomas, Ch. 9 (The Gaussian Channel).

### 3. Mutual information for additive noise, computed numerically: $I(\mu;Y) = h(Y) - h(e)$
**What.** For $Y = \mu + e$ with $e$ independent of $\mu$, $h(Y \mid \mu) = h(e)$, so $I = h(Y) - h(e)$. Both differential entropies can be computed on a grid from any density, so no distribution family is needed: estimate the noise density from data, convolve it with the empirical distribution of $\mu$ to get $f_Y$, integrate $-f \log_2 f$ for each.
**Where.** `floor_bits()` in "The floor, recomputed without a family".
**Why.** Removes the Gaussian assumption that the data had already failed (no family fit both the spike and the spread).
**Watch for.** Differential entropies can be negative and depend on units — only the *difference* is meaningful. Grid step must be well below the noise scale; grid range must cover the tails. Plug-in entropy of a KDE is biased for small samples (here ~1–2k residuals, fine).
**Source.** Cover & Thomas, Ch. 8 (Differential Entropy), especially the translation-invariance and additive-noise identities.

---

## B. Noise estimation and robust statistics

### 4. Pooled within-group variance (one-way ANOVA's "within" mean square)
**What.** $\operatorname{Var}(\text{noise}) = \sum_i\sum_j (y_{ij}-\bar y_i)^2 \big/ (N - G)$: replicate scatter around each group's own mean, pooled across groups, with one degree of freedom lost per group.
**Where.** Goal cell formula; `pooled_formula()` in the floor section.
**Why.** The standard way to estimate measurement noise from replicates without knowing the true values.
**Watch for.** It is a *variance* — quadratic in deviations — so a handful of gross errors dominate it. And it assumes the replicates are independent measurements: copies silently shrink it.
**Source.** Snedecor, G. W. & Cochran, W. G., *Statistical Methods* (8th ed., 1989), one-way classification.

### 5. Checking a distributional family by whether one scale fits two tail probabilities
**What.** Fit the family's scale once to a near-zero quantile (the spike) and once to a far quantile (the spread). If the family is right, the two scales agree and each fit predicts the other's tail. A gross mismatch (here: predicted 0–1% beyond 0.5 logS vs observed 19%) rules the family out *without* a formal goodness-of-fit test.
**Where.** "Can measurement noise explain the spike?"
**Why.** Fast, visual, and it tested four families at once (Gaussian, Laplace, t, Cauchy); all failed, which is the evidence that the spike is not noise.
**Watch for.** This rejects families; it does not confirm one. Pair it with an ECDF overlay (the 2×2 plot) so the reader sees *where* the shape fails.
**Source.** The logic is that of probability plotting: Wilk, M. B. & Gnanadesikan, R. (1968). *Probability plotting methods for the analysis of data.* Biometrika 55:1–17.

### 6. Heavy-tailed noise families (Student-t, Laplace, Cauchy)
**What.** Alternatives to the Gaussian whose tails decay polynomially or exponentially, so a few large deviations do not dominate the fit. Student-t with low degrees of freedom is the workhorse of robust modelling.
**Where.** `FAMILIES` in the spike/spread fits and the truncated fits.
**Why.** Replicate differences in compiled data have between-lab offsets and occasional gross errors; Gaussian scale estimates are useless there.
**Watch for.** $t_\nu$ has infinite variance for $\nu \le 2$ — exactly where the best fits landed — which is a second reason the variance-based floor formula was abandoned.
**Source.** Lange, K. L., Little, R. J. A. & Taylor, J. M. G. (1989). *Robust Statistical Modeling Using the t Distribution.* JASA 84:881–896.

### 7. Truncated maximum likelihood
**What.** When only observations beyond a cutoff $C_0$ are trusted, fit the model to those alone but divide the density by $P(|d| > C_0)$ (the survival function) so the likelihood is that of the truncated distribution. The fit can then be *projected inward* to estimate how many genuine observations the cutoff hid.
**Where.** `fit_beyond()` in "Choosing a cutoff".
**Why.** The copies contaminate the near-zero region; fitting only beyond it, correctly, gives a copy-free estimate of the genuine-difference shape.
**Watch for.** The projection inward inherits the family's shape; the notebook flags this as "likely low" and brackets it with an empirical reference (dataset A's own repeats).
**Source.** Cohen, A. C. (1991). *Truncated and Censored Samples: Theory and Applications.* Marcel Dekker.

### 8. Robust scale from the MAD: $1.4826 \times \operatorname{median}|r|$
**What.** The median absolute deviation, scaled by 1.4826 so it equals the standard deviation for Gaussian data, is a scale estimate that ignores the tails entirely (50% breakdown point).
**Where.** "noise SD (robust)" column in the votes/flags sweep table.
**Why.** A scale summary that is not hijacked by gross errors, so the sweep shows how the *typical* disagreement moves with the copy band.
**Watch for.** Efficiency is low (~37% of the SD for Gaussian data) — fine as a summary, not as the final estimator.
**Source.** Rousseeuw, P. J. & Croux, C. (1993). *Alternatives to the Median Absolute Deviation.* JASA 88:1273–1283.

### 9. Outlier identification against a leave-one-out robust centre
**What.** Flag a value when it lies more than a threshold from the *median of the other* values. Using the median (not the mean) keeps the outlier from pulling the centre toward itself; leaving the value out keeps it from voting on its own fate.
**Where.** `flag_errors()` in "Votes, gross errors and consensus".
**Why.** Separates gross errors (unit slips, wrong salt form) from ordinary between-source disagreement, which the hump analysis showed dies off by ~1.5 logS.
**Watch for.** Two traps the notebook hit and fixed: (a) **sequential updating cascades** — recomputing the centre after each flag lets it drift through one side of a split population, so update all flags simultaneously per round; (b) a **majority guard** — if half or more would be flagged, there is no consensus, so flag nothing. Also: two values can never resolve which one is wrong.
**Source.** Davies, L. & Gather, U. (1993). *The Identification of Multiple Outliers.* JASA 88:782–792 (the "Hampel identifier" family). Hampel, F. R. (1974). *The Influence Curve and Its Role in Robust Estimation.* JASA 69:383–393.

### 10. Leverage correction for leave-one-out residuals: multiply by $\sqrt{(K-1)/K}$
**What.** The residual of one value against the mean of the other $K-1$ has variance $\sigma^2(1 + 1/(K-1)) = \sigma^2 K/(K-1)$. Scaling by $\sqrt{(K-1)/K}$ restores variance $\sigma^2$, so residuals from compounds with different $K$ can be pooled into one noise distribution.
**Where.** `residuals` in `analyze()`.
**Why.** Without it, compounds with few votes contribute inflated residuals and the noise density is too wide.
**Watch for.** Assumes independent, equal-variance votes within a compound — the copy collapse is what makes that approximately true.
**Source.** Cook, R. D. & Weisberg, S. (1982). *Residuals and Influence in Regression.* Chapman & Hall (deletion residuals, Ch. 2).

### 11. Kernel density estimation with Scott's bandwidth
**What.** Smooth an empirical sample into a density by placing a Gaussian kernel on each point; bandwidth $\propto n^{-1/5}$ (Scott's rule, `scipy.stats.gaussian_kde` default).
**Where.** `floor_bits()`, on the symmetrized residuals.
**Why.** Gives the "no family" noise density that the numeric mutual information needs.
**Watch for.** KDE over-smooths sharp peaks and under-represents heavy tails; symmetrizing the residuals ($r$ and $-r$) is a cheap way to impose the one structural assumption we *are* willing to make (noise has no sign preference).
**Source.** Scott, D. W. (1992). *Multivariate Density Estimation.* Wiley. Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis.* Chapman & Hall.

### 12. Cluster (group) bootstrap
**What.** To get a confidence interval for a statistic computed over grouped data, resample the *groups* (compounds) with replacement, not the individual values, then recompute the statistic. Preserves the within-group dependence.
**Where.** The 100-replicate bootstrap for the error rate and the floor.
**Why.** Residuals within a compound are not independent (they share a consensus), so a plain bootstrap over residuals would be overconfident.
**Watch for.** 100 replicates gives rough percentile intervals; it is enough to say "1.42–1.67", not to resolve a third decimal.
**Source.** Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife.* Annals of Statistics 7:1–26. Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application.* CUP, §3.8 (hierarchical data).

### 13. Estimating a mixing fraction from a tail statistic
**What.** If a population is a mixture of two known reference populations, the share of any tail (here: pairs beyond 0.5 logS) is a linear blend: $s = (1-f)\,s_{\text{ref1}} + f\,s_{\text{ref2}}$. Solve for $f$.
**Where.** The unspecified-vs-specified stereo diagnostic: $f \approx 5\%$ behave like different compounds.
**Why.** Turns "keep them separate" from an assumption into a costed judgement, using only tail shares already in the table.
**Watch for.** Needs references that genuinely bracket the mixture; here the copy spike inflates both the test population and the replicate reference alike, so cross-source rows were used for the sharper comparison.
**Source.** The same move as estimating a null proportion from the tail of a p-value distribution: Storey, J. D. (2002). *A Direct Approach to False Discovery Rates.* JRSS-B 64:479–498.

### 14. Recognizing an unidentifiable mixture model
**What.** A latent-component model (EM for "genuine noise + gross error") needs the components to live at separable scales. When the data's scales form a continuum, the likelihood is maximized by degenerate solutions — one component collapses onto the smallest scale and the other absorbs everything else — and the fitted labels are meaningless even though the fit "converges".
**Where.** Mentioned in the conclusion; the explicit threshold replaced it.
**Why.** Knowing *when* a flexible model cannot answer the question saves it from answering wrongly with confidence.
**Watch for.** Symptoms: parameters pinned at their bounds, a scale estimate far below any physically plausible value, the error fraction jumping to whatever cap you set. The cure is usually more structure (explicit thresholds, extra covariates like source identity), not more components.
**Source.** McLachlan, G. & Peel, D. (2000). *Finite Mixture Models.* Wiley, §2.2 (unbounded likelihood and spurious local maxima).

### 15. Single-linkage clustering for copy collapse
**What.** Sort values; split wherever consecutive values differ by more than $\delta$. Each run is a cluster. In one dimension this is single-linkage clustering, exact and $O(n\log n)$.
**Where.** `votes_of()`.
**Why.** Copies reprinted across compilations differ by rounding only; clustering collapses them into one vote without deciding which copy is "the original".
**Watch for.** Chaining: a sequence of values each within $\delta$ of the next forms one cluster even if its ends are far apart. Acceptable at $\delta = 0.1$ logS with few values per compound; it would not be at larger $\delta$ or denser data. Use the cluster *mean*, not an endpoint, as the representative.
**Source.** Sibson, R. (1973). *SLINK: An Optimally Efficient Algorithm for the Single-Link Cluster Method.* Computer Journal 16:30–34.

---

## C. Chemical identity

### 16. InChI / InChIKey as a canonical compound identity
**What.** A layered, canonical string for a molecule (formula, connectivity, hydrogens, charge, stereo, isotopes), hashed to a 27-character key. Regenerating it from each row's structure, rather than trusting a supplied key, makes identity a property of the structure.
**Where.** "What counts as the same compound": the baseline key, and the stable member keys the reviewed tautomer table is pinned to.
**Why.** The sources write the same chemistry many ways; a canonical key is what "same compound" is computed from.
**Watch for.** Standard InChI normalizes *some* tautomers (mobile hydrogens) but not keto–enol and friends; it does not unify salts with parents; the `/m` layer encodes mirror-image parity, which the notebook flips to merge enantiomers.
**Source.** Heller, S. R., McNaught, A., Pletnev, I., Stein, S. & Tchekhovskoi, D. (2015). *InChI, the IUPAC International Chemical Identifier.* J. Cheminformatics 7:23.

### 17. Structure standardization: neutralization, fragment handling, tautomer canonicalization
**What.** A pipeline of deterministic transforms (RDKit `MolStandardize`, descended from MolVS): neutralize charges, choose fragments, pick a canonical tautomer by a scoring function over an enumerated set.
**Where.** `representation_merges()` (isotopes, bare ions) and the tautomer stage.
**Why.** Each transform encodes one identity ruling; applying them in stages shows what each ruling is worth in compounds and pairs.
**Watch for.** Two documented behaviours that bit this notebook: the canonicalizer *strips stereocentres near tautomeric sites by default* (merging diastereomers), and its pick among near-tied tautomers *is not stable across processes*. Neutralize only lone ions, never across salt stoichiometries (NaHS vs Na₂S).
**Source.** RDKit documentation, `rdkit.Chem.MolStandardize`; Sitzmann, M., Ihlenfeldt, W.-D. & Nicklaus, M. C. (2010). *Tautomerism in large databases.* J. Comput.-Aided Mol. Des. 24:521–551.

### 18. The "representation vs substance" rule for merging
**What.** Merge two records only when the difference between them *cannot change what was measured* (atom order, tautomer drawn, isotope, lone ion vs neutral form, enantiomer in an achiral solvent). Keep them apart when it can (salt, hydrate, diastereomer, racemate vs enantiopure crystal, polymorph).
**Where.** The identity contract table.
**Why.** Gives every ruling a single test to pass, so the table is reviewable rather than a list of preferences.
**Watch for.** Some differences of substance are invisible to any structure key (polymorphs, racemate crystals): name them as irreducible noise rather than pretend a key can see them.
**Source.** On curating structures before modelling: Fourches, D., Muratov, E. & Tropsha, A. (2010). *Trust, but Verify.* J. Chem. Inf. Model. 50:1189–1204. On stereoisomer solubility: Jacques, J., Collet, A. & Wilen, S. H. (1981). *Enantiomers, Racemates, and Resolutions.* Wiley. On polymorph solubility ratios: Pudipeddi, M. & Serajuddin, A. T. M. (2005). *Trends in Solubility of Polymorphs.* J. Pharm. Sci. 94:929–939.

### 19. Distinguishing tautomers from positional isomers
**What.** A tautomer moves a hydrogen between heteroatom sites on an unchanged skeleton and equilibrates in water faster than any measurement (azo ↔ hydrazone, lactam ↔ lactim, 1,3-dione ↔ enol). A transform that moves a C=C or C=O *along the skeleton* produces a stable positional isomer that is stocked and measured as a different compound (allyl vs propenyl, α- vs β-ionone, aldose vs ketose, Δ4 vs Δ5(10) steroid).
**Where.** The reviewed-merges rule and the kept-separate list.
**Why.** The canonicalizer's transform set spans both; the chemistry decides which merges are real.
**Watch for.** Ketose/aldose and Δ-steroid pairs look like tautomers to software and are not, in practice, interconverting in a solubility vial.
**Source.** Sitzmann et al. (2010), above, on the scope of tautomer transform rules; any organic chemistry text on prototropic tautomerism vs. allylic rearrangement.

---

## D. Data provenance and process skills

### 20. Treating a too-good-to-be-true spike as a provenance signal
**What.** When a compiled dataset shows agreement far better than its measurement process could deliver (36% of replicate pairs within 0.005 logS), suspect copying between compilations before suspecting precision. Test it by whether agreement depends on *which sources* the two values came from (5% within one source vs up to 88% between some pairs).
**Where.** The first two analytical sections.
**Why.** Copies masquerade as replicates and make the data look more precise than it is; they would have halved the noise estimate.
**Watch for.** The dependence-on-source test is the decisive one; the family-fitting test alone only says "something is off".
**Source.** Sorkun, M. C., Khetan, A. & Er, S. (2019). *AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds.* Scientific Data 6:143 (the dataset's own description of its nine sources). Fourches et al. (2010), above.

### 21. Choosing a cutoff by marginal analysis
**What.** Instead of asking "what is merged at cutoff $c$?", ask "of the pairs *newly* merged by raising the cutoff from $c_1$ to $c_2$, what share are copies?" Keep raising while the marginal batch is mostly what you intend to catch. Bracket the answer with an optimistic and a pessimistic reference.
**Where.** "Choosing a cutoff" table (the "% copies, newly merged pairs" column).
**Why.** A total share hides the point where the next step starts doing more harm than good; the marginal share shows it directly (93–99% copies up to 0.02; 44–84% in 0.02–0.05).
**Watch for.** Each reference carries a known bias direction; state them ("likely low", "likely high").
**Source.** This is cost–benefit reasoning at the margin; the statistical cousin is the ROC/precision curve read step-by-step rather than at one operating point.

### 22. Flag, never delete (provenance preservation)
**What.** Suspect values stay in the dataset with a flag and the reason it was raised; downstream code decides what to do with them. Deleting destroys the evidence that a judgement was made.
**Where.** `flagged` column in the hand-off; the "every vote keeps its row" principle.
**Why.** The flagging rule has a threshold and a known error rate; a later analysis may want to revisit either.
**Watch for.** Flagging without a stated rule is just deletion with extra steps — the rule and its sensitivity must be in the notebook.
**Source.** The ALCOA+ data-integrity principles (attributable, legible, contemporaneous, original, accurate; plus complete, consistent, enduring, available): U.S. FDA (2018). *Data Integrity and Compliance With Drug CGMP — Guidance for Industry.*

### 23. Nominate-then-review for non-deterministic or over-eager tools
**What.** When a tool's output can drift between runs or over-reaches, do not key the result on the tool. Let it *nominate* candidates, apply the decision from an explicit reviewed list pinned to stable identifiers, and assert that every nomination has a ruling — so new behaviour fails the run instead of silently changing the data.
**Where.** `REVIEWED_MERGES` / `REVIEWED_SEPARATE` and the two asserts.
**Why.** The tautomer canonicalizer merged two steroid pairs in one run and not the next.
**Watch for.** The reviewed list must be keyed on something that *does not* depend on the flaky tool (here: pre-tautomer InChIKeys).
**Source.** The pattern is that of characterization (golden-master) tests: Feathers, M. (2004). *Working Effectively with Legacy Code.* Prentice Hall, Ch. 13.

### 24. Testing a judgement call against reference populations
**What.** For a ruling you cannot derive from first principles (are unspecified-stereo entries the same compound?), construct the population the ruling is about and compare it with two references that bracket the hypotheses (same-compound pairs; different-compound pairs). Report where it falls and what the ruling costs.
**Where.** "Testing the unspecified-stereo ruling".
**Why.** Converts "we assume" into "we measured, and the ruling costs ~5% of pairs".
**Watch for.** Make the comparison like-for-like (copies inflate both the test population and one reference; compare cross-source rows).
**Source.** General empirical-Bayes reasoning; see entry 13 for the mixing-fraction arithmetic.

### 25. Sensitivity grids over analyst choices
**What.** Where a result depends on a judgement (copy band δ, error threshold T), compute it across a small grid of defensible values and show the whole grid. The spread *is* the uncertainty attributable to the judgement, separate from sampling uncertainty (which the bootstrap gives).
**Where.** The δ × T tables for error rate, noise SD, and floor.
**Why.** One number would have hidden that the floor moves 1.3 → 1.9 bits with δ — the honest statement is the band.
**Watch for.** Choose the grid *before* seeing which setting flatters the result; mark one setting as primary with a reason.
**Source.** Steegen, S., Tuerlinckx, F., Gelman, A. & Vanpaemel, W. (2016). *Increasing Transparency Through a Multiverse Analysis.* Perspectives on Psychological Science 11:702–712. Simonsohn, U., Simmons, J. P. & Nelson, L. D. (2020). *Specification curve analysis.* Nature Human Behaviour 4:1208–1214.

### 26. Carrying an estimated error rate as a prior for the uncheckable majority
**What.** When quality can be checked only on a subset (compounds with 3+ votes), estimate the error rate there and *carry it as a prior* for the records that cannot be checked (single-vote compounds), rather than treating them as clean because nothing flagged them.
**Where.** The conclusion of "Votes, gross errors and consensus"; the "label as it arrives" floor.
**Why.** 8,366 of 9,946 compounds have one vote; at a 5–8% error rate, several hundred carry wrong labels that no structural model can fix. Pretending otherwise would make model errors on those compounds look like model failures.
**Watch for.** The checked subset is not a random sample (compounds measured often are better known); say so.
**Source.** Standard missing-data / selection reasoning; a readable treatment is Gelman, A. et al., *Bayesian Data Analysis* (3rd ed., 2013), Ch. 8 (modelling the data-collection process).

### 27. Compute every comparison number inside the document
**What.** When a conclusion compares against a simpler method, generate that number in the same run, with the reason it is being shown, instead of quoting a figure from elsewhere.
**Where.** `pooled_formula()` printed beside the distribution-free floor.
**Why.** A quoted number cannot be audited and goes stale; a generated one decomposes the difference (here: what gross errors do to a pooled variance vs what the Gaussian assumption does).
**Watch for.** Each comparison line should answer a specific "what does this simplification cost?" — not just show a second number.
**Source.** Reproducible-research practice; see Peng, R. D. (2011). *Reproducible Research in Computational Science.* Science 334:1226–1227.

---

## E. Discretizing a target and benchmarking it (`02_tiers.ipynb`)

### 28. Mutual information between two discretizations of a noisy quantity
**What.** Bin a continuous value into $K$ tiers and compute $I(T_1;T_2)$ between the tiers of two measurements of the same thing. It is bounded by $H(T) \le \log_2 K$, reached only when one measurement's tier fixes the other's and all tiers are equally likely. Finer bins always raise $I$ (toward the continuous $I(Y_1;Y_2)$) but lower the chance the two agree.
**Where.** The held-out curve over $K = 2 \ldots 8$.
**Why.** It answers "how many bits does a repeat measurement share with the first about the *label*?" — the quantity every later model is compared against.
**Watch for.** MI is invariant to relabelling, so a model that systematically swaps two tiers scores full MI (the toy table's last row). Pair it with an ordinal summary (same / one away / two-plus away).
**Source.** Cover & Thomas, *Elements of Information Theory* (2006), Ch. 2 (properties of MI; data-processing inequality for any function of $Y$, including binning).

### 29. Calibrating a metric on toy distributions before trusting it
**What.** Before scoring real data, evaluate the metric on hand-built joint distributions whose answer you already know (perfect agreement, unequal shares, neighbour confusion, independence, a one-to-one swap). The table of results is the metric's behaviour, made explicit.
**Where.** The toy joint-distribution table.
**Why.** It shows at a glance that MI rewards *any* one-to-one pattern and that unequal tier shares cap it below $\log_2 K$ — two facts that would otherwise be misread as model behaviour later.
**Watch for.** Keep the toys small and name each one by the property it tests; a toy you can't name isn't testing anything.
**Source.** This is the practice of unit-testing an evaluation metric; a readable statement of the principle is in Kuhn, M. & Johnson, K., *Applied Predictive Modeling* (2013), Ch. 11 (measuring performance in classification).

### 30. Cross-fitting: tune on one half, score on the other, both ways
**What.** Any parameter tuned to maximize a score on a dataset will look better on that dataset than on new data. Split the units (here compounds, so a compound's pairs never straddle the split), tune on one half, score on the other, swap, and repeat over random halvings. Report the held-out mean and its spread; show the in-sample number beside it so the optimism is visible.
**Where.** 20 halvings × 2 directions per $K$; benchmark 0.963 held-out vs 1.008 in-sample.
**Why.** The benchmark must be a number a *new* measurement would reproduce, not one the boundaries were fitted to.
**Watch for.** Split by the unit that carries the dependence (compound), not by row. The spread over halvings mixes boundary variability with sampling variability — fine for a tie-width, not a formal standard error.
**Source.** Hastie, T., Tibshirani, R. & Friedman, J., *The Elements of Statistical Learning* (2nd ed., 2009), §7.4 (optimism of the training error) and §7.10 (cross-validation).

### 31. Coordinate ascent over a candidate grid instead of brute force
**What.** When the objective (MI) is not decomposable but each parameter (a boundary) lives on a small grid, cycle through parameters, try every candidate for one while holding the rest fixed, keep improvements, and stop when a full pass changes nothing. Start from a sensible point (quantile cuts) and it converges in a few passes.
**Where.** `optimize()`; replaces an exhaustive search that would need $\binom{160}{7}$ evaluations at $K = 8$.
**Why.** Makes the whole $K$ sweep and cross-fit cheap enough to repeat 40 times per $K$.
**Watch for.** It finds a local optimum; here the landscape is smooth and the quantile start is near the answer, and the result is checked against the quantile reference anyway. Add random restarts if the two ever disagree sharply.
**Source.** Wright, S. J. (2015). *Coordinate descent algorithms.* Mathematical Programming 151:3–34.

### 32. Summed-area tables for O(1) block sums
**What.** Precompute the 2-D cumulative sum of a weight matrix; the total weight in any rectangle is then four lookups (inclusion–exclusion). With a zero row and column in front, no edge cases.
**Where.** `cumulative()` and `joint()`: every candidate boundary set's joint tier distribution is a handful of array lookups.
**Why.** It is what makes thousands of MI evaluations per fit affordable.
**Watch for.** Floating round-off from the subtraction leaves tiny negatives; clip them before taking logs.
**Source.** Crow, F. C. (1984). *Summed-area tables for texture mapping.* SIGGRAPH '84. Popularized as the "integral image" by Viola, P. & Jones, M. (2001), *Rapid object detection using a boosted cascade of simple features.*

### 33. Design weights so each unit counts once
**What.** A compound with $n$ votes has $n(n-1)$ ordered pairs; give each pair weight $1/(n(n-1))$ so every compound contributes one unit, however often it was measured. Likewise each vote gets weight $1/n$ downstream.
**Where.** `pairs["w"]`; the `weight` column in the hand-off.
**Why.** A few heavily measured compounds (up to 13 votes) would otherwise dominate the benchmark and the tier boundaries.
**Watch for.** Decide the unit of analysis explicitly (compound, not measurement) and carry the same weighting into every later score, or the numbers stop being comparable.
**Source.** Lohr, S. L., *Sampling: Design and Analysis* (2nd ed., 2010), Ch. 7 (weighting and unequal probabilities of selection).

### 34. A declared standard instead of a threshold on a smooth curve
**What.** When a performance curve has no knee (information keeps rising with $K$, each tier adding a little less), any rule based on marginal gain lands on a razor's edge. Declare, before reading the table, a standard phrased in terms of what the output is *for* — here, "a label must be reproduced by a repeat three times in four" — and read $K$ off it. Show the whole curve so a reader with a different standard can pick differently.
**Where.** `MIN_SAME_TIER = 75` and the reading of the curve.
**Why.** Turns an unavoidable judgement into a transparent, criticizable one, and separates "how much information" (rises with $K$) from "how reliable a label" (falls with $K$).
**Watch for.** Declare the rule before the table, and report where the alternatives would have landed (here, five tiers at 71%).
**Source.** On the cost of rules chosen after seeing the data: Gelman, A. & Loken, E. (2013). *The garden of forking paths.* (Columbia University, unpublished manuscript; widely cited.)

### 35. Always score the no-tuning baseline the same way
**What.** Whatever is tuned, also score the simplest untuned alternative (quantile boundaries) under the identical held-out protocol. The gap is what tuning is worth.
**Where.** "held-out MI, quantile boundaries" column.
**Why.** It revealed that tuning buys ≤ 0.03 bits and nothing at all beyond $K = 4$: the boundaries are weakly determined, which changes how much meaning anyone should read into them later.
**Watch for.** The baseline must be scored on the same held-out halves, with the same weights, or the comparison is not like-for-like.
**Source.** General practice; stated crisply in Kuhn & Johnson (2013), Ch. 4 (the importance of a null or naïve model).

### 36. Gray codes for ordinal labels
**What.** A reflected Gray code orders $2^b$ bit-vectors so that consecutive ones differ in exactly one bit (00, 01, 11, 10). Encoding ordered tiers this way means the most common confusion — neighbouring tiers — costs one bit flip, not two.
**Where.** `GRAY` and the `bit_1`, `bit_2` columns; justified by the joint distribution (20% of disagreements one tier apart, 2% two or more).
**Why.** The downstream models predict bits; the code should make the likely errors the cheap ones.
**Watch for.** Check the one-bit property with an assert, and re-derive it if $K$ changes — the first $K$ codes of a reflected Gray code keep the property for any $K$.
**Source.** Gray, F. (1953). *Pulse code communication.* U.S. Patent 2,632,058. Savage, C. (1997). *A survey of combinatorial Gray codes.* SIAM Review 39:605–629.

### 37. Ceilings via the data-processing inequality
**What.** If $T_1 \to S \to T_2$ (two measurements independent given the true value $S$), then $I(T_1;T_2) \le I(S;T_2)$. So the repeat-measurement benchmark is a *lower* bound on what a perfect model of $S$ could share with a measured tier, and $H(T)$ is an *unreachable* upper bound because the measured tier is itself noisy. The real ceiling lies between them.
**Where.** The two cautions at the end of the curve reading.
**Why.** Prevents two later misreadings: calling $H(T)$ "100% of the uncertainty removed" as if a model could reach it, and treating the repeat benchmark as the most any model could do.
**Watch for.** The inequality needs conditional independence of the two measurements given $S$; copies between compilations violate it — another reason the copy collapse had to come first.
**Source.** Cover & Thomas (2006), §2.8 (data-processing inequality).

### 38. Flag-weighted labels: zero weight, never deletion, decided downstream
**What.** Upstream flags a suspect record; the consumer decides what the flag means for its own purpose and encodes that as a weight (here 0 for the benchmark and the training weight), leaving the row and its flag in the file.
**Where.** The two "choices up front" and the `weight` column.
**Why.** A value 1.5 logS from its own compound's consensus should neither shape tier boundaries nor score a model, but a later analysis (e.g. of label error) will want exactly those rows.
**Watch for.** State the weighting rule in the notebook that applies it, not only the one that raised the flag — each consumer may legitimately choose differently.
**Source.** Same provenance principles as entry 22 (ALCOA+); on learning with label noise and instance weights, see Frénay, B. & Verleysen, M. (2014). *Classification in the presence of label noise: a survey.* IEEE TNNLS 25:845–869.

---

## F. Additive models and the information–length front (`07_additive.ipynb`)

### 39. Declare a description length before fitting, in one unit shared across model families
**What.** Before any model is fitted, write down how its size will be counted — for an EBM, bins × tiers summed over terms; for a forest, leaves × tiers summed over trees; for an equation, nodes; for a neighbour model, stored rows × dimensions — and report that count beside the held-out information. One common unit, "numbers needed to write the model down", lets every family land on the same chart even though the units are not equally hard to read.
**Where.** The "Description length" bullet of the framework and `ebm_length()` / `forest_length()`; the hand-off `data/aqsoldb_front_additive.csv`.
**Why.** Without a declared count, "interpretable" is a feeling; with one, the cost of readability is a number (here: thirteen times shorter for 0.07 bits).
**Watch for.** Say which counts a reader must actually take in (82 of 152 terms carry 90% of the importance) and mark unreadable models as such rather than omitting them — they are the reference for what length buys when readability is not required.
**Source.** Rissanen, J. (1978). *Modeling by shortest data description.* Automatica 14:465–471; Grünwald, P., *The Minimum Description Length Principle* (2007), Ch. 1.

### 40. Explainable boosting machines as the first fully readable model
**What.** An EBM is a generalized additive model fitted by cyclic gradient boosting on one feature at a time: each tier's log-odds is an intercept plus one step function per descriptor plus, optionally, a few pairwise step functions. The model is nothing but the sum of its drawings, so it can be read in full; the gap to a forest on the same inputs is the information that lives in interactions the EBM was not allowed.
**Where.** `ExplainableBoostingClassifier` with `interactions` chosen on validation from a declared grid (0, 5, 20); the three-scheme table against the forest.
**Why.** It answers "how much do the forest's bits depend on interactions?" directly — here very little — and gives the shape of each dependence, which permutation importance cannot.
**Watch for.** Multiclass EBMs in InterpretML strip pairs from the global explanation and cap pairs at the number of available feature pairs; both must be stated. Outside the training range every step function sits at its last value and the terms keep adding, so an EBM is over-confident on unfamiliar chemistry (the learned-split result).
**Source.** Lou, Y., Caruana, R. & Gehrke, J. (2012). *Intelligible models for classification and regression.* KDD '12; Lou, Y. et al. (2013). *Accurate intelligible models with pairwise interactions.* KDD '13; Nori, H. et al. (2019). *InterpretML: a unified framework for machine learning interpretability.* arXiv:1909.09223.

### 41. Draw shape functions as the median over folds with an interquartile band
**What.** Evaluate each term on a fixed grid of the descriptor (1st to 99th percentile) for every cross-validation fit, then plot the median contribution per tier with the interquartile range shaded. Centre the terms so one tier's curve against the others is the model's whole opinion of that descriptor.
**Where.** `eval_terms()` on grids; the six-panel shape figure.
**Why.** A narrow band says the curve is a property of the data, not of one fit; a wide band (ExactMolWt below 120 Da) says the model has few compounds there and the curve should not be read.
**Watch for.** Describe curves by their plateaus and transitions (where it saturates, where the step is), not by a slope — step functions have no slope, and the useful reading is "beyond logP 5 the model has nothing more to say".
**Source.** Hastie, T. & Tibshirani, R., *Generalized Additive Models* (1990), Ch. 6 (plotting fitted functions with pointwise bands); Caruana, R. et al. (2015). *Intelligible models for healthcare.* KDD '15 (reading shape functions for surprises).

### 42. Equal accuracy with less information means a confidence failure, not a ranking failure
**What.** Information $= H(T) - $ log-loss rewards both getting the tier right and knowing how sure to be. When two models have the same hard-tier accuracy but one carries much less information, the difference is in the spread of the distributions: the weaker one gives sharp distributions on wrong tiers. Report accuracy beside information so the two failure modes can be told apart.
**Where.** The learned-split rows: EBMs at −1.4 to −1.6 bits against the forest's −0.6 with same-tier accuracy within one point.
**Why.** The remedy differs: a ranking failure needs a better model; a confidence failure needs calibration that sees the shift (λ chosen on an in-distribution validation side cannot).
**Watch for.** Calibration tuned on a validation side drawn from the training distribution is blind to distribution shift; a model's confidence on unfamiliar chemistry is only as good as the chemistry its validation side covered.
**Source.** Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). *Probabilistic forecasts, calibration and sharpness.* JRSS B 69:243–268; Murphy, A. H. (1973). *A new vector partition of the probability score.* J. Appl. Meteorol. 12:595–600.

### 43. Selection stability across folds as the test of whether a term is real
**What.** When a model selects structure (here, which pairwise interactions to include), count how many of the cross-validation fits selected each item. Structure chosen in every fit is a property of the data; structure chosen once is a property of the fold.
**Where.** The pairs table: 110 distinct pairs over ten fits of twenty, none chosen in every fit, the most repeated in seven.
**Why.** It explains the small gain from interactions (0.014 bits) without a second experiment, and it stops any one fit's pair list being read as chemistry.
**Watch for.** Instability can mean the interactions are weak or that several near-equivalent descriptors (the correlation groups) can stand in for each other; the per-group representative choice upstream affects which pairs appear.
**Source.** Meinshausen, N. & Bühlmann, P. (2010). *Stability selection.* JRSS B 72:417–473.

---

## G. Equations by symbolic regression (`08_equations.ipynb`)

### 44. Bridge a point prediction to a distribution with a hedging parameter chosen on validation
**What.** A regressor gives a value; a log-loss score needs a distribution over tiers. Integrate a Gaussian centred on the value over each tier's interval, with σ chosen on the validation side by the information it yields, then apply the same shrinkage λ as every other model. σ is the equation's way of hedging, as λ is the forest's; its size relative to the replicate noise says whether the errors are the measurement's or the model's.
**Where.** `bridge()`, `choose_sigma()`, `SIGMAS`; every equation chose σ of 1.2–1.5 logS against replicate noise of about 0.5.
**Why.** It puts a published equation, a refit and a search result on exactly the same scale as classifiers, and it exposes over- and under-confidence as a number rather than a feeling.
**Watch for.** A σ far above the replicate noise means the bridge is covering model error; a Student-t bridge with ν on validation is the fallback if residuals are heavy-tailed. Recover the tier boundaries from the hand-off and assert they reproduce every vote's tier before using them.
**Source.** Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules, prediction, and estimation.* JASA 102:359–378; on ordinal thresholds over a latent Gaussian, McCullagh, P. (1980). *Regression models for ordinal data.* JRSS B 42:109–142.

### 45. Score the published reference and a fair refit of its form before searching
**What.** Before running a search, score the literature equation as published (nothing fitted but the bridge), then its form refitted on the train side — and if the target has heavy tails, refit with a robust loss as well as least squares. The three numbers separate what the form is worth, what the coefficients are worth and what the loss did.
**Where.** Delaney's line (0.595 bits), its least-squares refit (0.451) and Huber refit (0.555), with the coefficient table beside them.
**Why.** Here least squares lost to the published coefficients, which looked like a bug until the coefficient table showed the squared loss had halved the logP slope to chase a target spanning fifteen logS units; the Huber refit recovered the slopes to within 0.1 of Delaney's.
**Watch for.** When a refit on more data scores worse than the original, suspect the loss before the data. Show the coefficients, not just the score: a reader can see what moved.
**Source.** Huber, P. J. (1964). *Robust estimation of a location parameter.* Ann. Math. Statist. 35:73–101; Delaney, J. S. (2004). *ESOL: estimating aqueous solubility directly from molecular structure.* J. Chem. Inf. Comput. Sci. 44:1000–1005.

### 46. Declare the search budget as what actually stops the search, and check it
**What.** A search has several stopping rules (iterations, wall-clock, convergence). Declare the one that binds, measure that it binds, and run the search once with a budget ten times larger on one split to show what the budget costs. If the longer search moves the chosen result within its split-to-split spread, the budget is converged for that operator set and size.
**Where.** The "Is 200 iterations enough?" cell: ten times the iterations moved validation information by +0.002 and −0.008 bits.
**Why.** The first draft declared a 150-second time budget that never bound (searches took six seconds); a declared protocol that does not describe the run is worse than none.
**Watch for.** Time a search before writing the protocol. A search that converges in seconds says the ceiling is the form (operators and size), not the search — which is itself a result.
**Source.** Cranmer, M. (2023). *Interpretable machine learning for science with PySR and SymbolicRegression.jl.* arXiv:2305.01582.

### 47. Read a symbolic-regression result as recurring structure across splits, not as one equation
**What.** One run of an evolutionary search returns one equation; ten runs on ten splits return a distribution of equations. Read the stable part: which variables appear in how many, the sign of the derivative at the median compound, and the recurring functional form — and count description length in a canonical form, since the search leaves constants unfolded.
**Where.** The variable-recurrence table (MolLogP in 10 of 10 with negative slope; the recurring $\log(MW + (\log P - c)^4)$ term); `shortest()` for the node count.
**Why.** The stable part is the finding (a bend in logP, not a slope; size inside the same logarithm); the unstable part is the fold. It also lets a search result be compared with another family's picture of the same dependence — here the additive model's logP curve.
**Watch for.** A derivative at the median is a crude direction; forms like a ratio or a log change sign across the range. Count nodes on the shorter of the raw and simplified forms, or sizes are not comparable between a search and a hand-written line.
**Source.** Schmidt, M. & Lipson, H. (2009). *Distilling free-form natural laws from experimental data.* Science 324:81–85; on the front as the object to report, Cranmer (2023) §3.

---

## H. Optimal trees and rule lists (`09_trees_and_rules.ipynb`)

### 48. Factorize a multiclass target into binary targets and measure what the factorization costs first
**What.** When a solver is binary and the target is ordinal, encode the classes in Gray bits and fit one model per bit; multiply the bit probabilities into a class distribution. Before fitting the solver, measure the cost of the factorization with a model that can do both: here two per-bit forests multiplied against the four-class forest.
**Where.** The factorization check (0.992 against 1.009 bits on validation: 0.017 bits).
**Why.** It separates the limit of the encoding from the limit of the solver; without it a weak tree result could be blamed on either.
**Watch for.** The product assumes the bits are independent given the leaf; the check with the forest is on the whole model, so state the assumption and show the number rather than argue it.
**Source.** Frank, E. & Hall, M. (2001). *A simple approach to ordinal classification.* ECML 2001 (binary decompositions of ordinal targets); Gray-code entry 36 above.

### 49. Drive an incompatible solver from a side environment through files, never imports
**What.** When a package will not build on the project's Python, put it in its own environment with a rebuild script, and have the notebook call a driver there by subprocess that writes per-row predictions and the models as text and JSON. The notebook reads files back; nothing from the side environment is imported into the kernel.
**Where.** `tools/trees_and_rules/` (`setup.sh`, vendored patched sdist, `run.py`); `run_driver()` and `load_driver()` in the notebook.
**Why.** The project kernel stays one environment that the user's editor and nbconvert both run; the solver's numpy pin and Cython build never leak into it; the run is reproducible from the script.
**Watch for.** Every convention the driver relies on — which child of a node is the "true" branch, whether a binarized column means `x <= t` or `x > t` — must be asserted against the solver's own predictions, not assumed from documentation. The first pass here printed every tree with the wrong inequality because the threshold direction was assumed; the predictions were right because the traversal was asserted, and the text was wrong because the label was not.
**Source.** Standard reproducible-environment practice; the specific lesson is Entry 27's (compute, don't quote) applied to conventions.

### 50. Read a provably optimal small model for what it does not ask
**What.** An optimal sparse tree with a declared size penalty is the best model of that size; what it leaves out of a 152-descriptor set is therefore informative, not an omission. Report the descriptors used across folds and the ones never used, and read the threshold that recurs.
**Where.** The high-bit tree is the single split MolLogP ≤ 2.76 in all ten splits; no size, polar-surface or hydrogen-bond descriptor ever appears; the threshold sits in the middle of the EBM's steepest logP range.
**Why.** It gives a concrete, falsifiable statement — "with eight leaves to spend, only logP and one complexity measure pay" — and a cross-family check: two models with nothing in common but the data agreed on where solubility turns.
**Watch for.** The regularization grid should be wide enough that the chosen value is not at an edge; here both solvers chose the loosest setting and were still gaining, which is itself a finding (size, not optimality, is the limit) but should be said.
**Source.** Lin, J., Zhong, C., Hu, D., Rudin, C. & Seltzer, M. (2020). *Generalized and scalable optimal sparse decision trees.* ICML 2020; Angelino, E., Larus-Stone, N., Alabi, D., Seltzer, M. & Rudin, C. (2017). *Learning certifiably optimal rule lists for categorical data.* JMLR 18:1–78.

---

## I. The front across families (`10_front.ipynb`)

### 51. Draw the information–length front and read the slopes between its points
**What.** Collect one hand-off file per notebook with the same columns (model, family, inputs, scheme, subset, information, spread, length, readable); concatenate, merge the same model that several notebooks fitted under different names, and chart information against log length per scheme. Mark a model as on the front if nothing of the same or shorter length (rounded to two significant figures, so near-equal lengths compete directly) has more information, and report bits per decade between consecutive front points.
**Where.** `data/aqsoldb_front_*.csv`, `SAME_MODEL`, the chart and the front table.
**Why.** The slope is the price of the next bit in each region: cheap in the first 22 numbers, flat from an equation to a four-curve EBM, steep from four inputs to 152, and 0.075 bits per decade for giving up readability.
**Watch for.** An unreachable ceiling drawn on the chart compresses the data; print it and leave it off. Labels for points in the same decade need stacking. Say in words what the length unit hides (a curve is read faster than its bins), and weigh the front a second time in reading time.
**Source.** Rissanen (1978) and Grünwald (2007) as in entry 39; on readability as a cost that is not a number, Rudin, C. (2019). *Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.* Nature Machine Intelligence 1:206–215.

### 52. Check the three split schemes for a reversal of order
**What.** A ranking on random splits is one ranking; read the cluster and learned-split columns for whether it holds. A reversal (here the shortest models being the least over-confident on learned splits, and the EBMs the most) is a second axis of the comparison that the main chart cannot show.
**Where.** The learned-split column of the front table and point 3 of the reading.
**Why.** It turns "the forest is best" into "the forest is best on familiar chemistry, loses the most on unfamiliar, and is out-hedged by an equation on the hardest splits" — which is what a user choosing a model needs.
**Watch for.** Learned-split information is negative for every model; compare models by how negative, and pair it with accuracy to tell confidence failures from ranking failures (entry 42).
**Source.** Sheridan, R. P. (2013). *Time-split cross-validation as a method for estimating the goodness of prospective prediction.* J. Chem. Inf. Model. 53:783–790; Bates, S., Hastie, T. & Tibshirani, R. (2024). *Cross-validation: what does it estimate and how well does it do it?* JASA 119:1434–1445.

---

## J. Splits (`03_splits.ipynb`)

### 53. Group held-out units by what the features can see, not by what the records say
**What.** A test compound is only "unseen" if nothing with the same feature vector was in training. Descriptors are blind to stereochemistry, protonation and counter-ions, so every scheme groups compounds by *skeleton* — the formula-and-connectivity block of the InChIKey of the largest fragment — and deals whole skeletons to one side. Grouping only salts with parents, the usual habit, leaves stereo and protonation siblings straddling the split.
**Where.** The `skeleton` column; `assign_folds()` deals skeletons, not compounds; every later scheme (cluster, learned) inherits the rule.
**Why.** Several hundred compounds have a feature-identical sibling; a random split by row would score memorization as generalization.
**Watch for.** Derive the grouping key from the *representation's* blind spots, so a representation that does see stereochemistry (the graph sets) is grouped more strictly than it needs, never less. Keep the 2048-bit fingerprint check (entry 55) as a second net: positional isomers slip through skeleton grouping.
**Source.** Heller, S. R. et al. (2015). *InChI, the IUPAC International Chemical Identifier.* J. Cheminf. 7:23; on leakage through near-duplicates, Kapoor, S. & Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4:100804.

### 54. Sphere-exclusion clusters with a threshold rule declared before the sweep
**What.** To hold out neighbourhoods, cluster by the leader algorithm on Tanimoto distance between Morgan fingerprints: walk the compounds, make one a centre if it is further than a threshold from every centre so far, assign the rest to the nearest centre, and deal whole clusters to folds. Sweep the threshold and show, for each value, cluster count, fold balance and the nearest-training similarity it produces; pick by a rule written before the sweep — here the smallest distance at which no test compound has an identical fingerprint in training, with folds within a few percent of equal.
**Where.** `sphere_clusters()`, `THRESHOLDS`, `THRESHOLD`, the sweep table.
**Why.** The threshold is a dial that sets how hard the test is; declaring the rule first keeps the choice from drifting to whatever makes a later model look good. Sphere exclusion is also the project's answer to scaffold splits: it clusters on whole-molecule similarity rather than a ring-system key, so it does not throw a chain analogue and a ring analogue into different groups by accident.
**Watch for.** Each skeleton takes the cluster of its first member so entry 53 is preserved. Held-out neighbourhoods are also solubility neighbourhoods, so the tier mix of single folds wanders (9–19% in the least soluble tier against 14–18% at random): score over all ten splits, never one.
**Source.** Taylor, R. (1995). *Simulation analysis of experimental design strategies for screening random compounds as potential new drugs.* J. Chem. Inf. Comput. Sci. 35:59–67; Butina, D. (1999). *Unsupervised data base clustering based on Daylight's fingerprint and Tanimoto similarity.* J. Chem. Inf. Comput. Sci. 39:747–750.

### 55. Measure every split scheme's hardness on one yardstick
**What.** For each test compound, the Tanimoto similarity to its most similar training compound on the split where it is tested, plus the share of test compounds with an identical fingerprint in training. Report quartiles per scheme. Later schemes (learned splits) are measured the same way, so "how familiar is the test chemistry" is one axis for all of them.
**Where.** `nearest_train_similarity()`, reused verbatim in the learning-to-split notebook.
**Why.** It is what let the learning-to-split notebook say its hard splits are *not* hard through unfamiliarity (median similarity 0.55 against random 0.59 and cluster 0.47) — a finding that only exists because the yardstick was fixed here.
**Watch for.** Both cross-validation schemes test every compound exactly once, so each gives one number per compound; a scheme that tests a compound several times needs an explicit rule for which split counts.
**Source.** Sheridan, R. P. et al. (2004). *Similarity to molecules in the training set is a good discriminator for prediction accuracy in QSAR.* J. Chem. Inf. Comput. Sci. 44:1912–1928.

### 56. One file per split scheme, and a validation rule fixed with the folds
**What.** Split $k$ tests on fold $k$ and validates on fold $k+1$ (wrapping), so every compound is tested once and validated once. Each scheme writes its own columns (`cv_0…9`, `cluster_0…9`) and later schemes write their own file with the same layout, so rerunning this notebook never erases a split that took an hour to learn.
**Where.** `sides()`, `SIDES`, the hand-off `data/aqsoldb_splits.csv`; `data/aqsoldb_splits_ls.csv` from the next notebook.
**Why.** Every downstream score is "train on these rows, choose on those, audit on the rest"; writing the three sides to disk makes that protocol a file every notebook reads rather than a convention each re-implements.
**Watch for.** Training sets overlap heavily across the ten splits, so the spread of scores understates sampling uncertainty; the harness corrects for it (entry 59). Say which fold is validation in the file, not in prose.
**Source.** Hastie, Tibshirani & Friedman (2009), §7.10; same provenance principle as entry 22.

---

## K. Feature discovery (`04_features.ipynb`)

### 57. Compare representations under one fixed, untuned model and call the score a lower bound
**What.** Score every representation with the same random forest at the same settings, chosen once and never tuned. Differences are then differences between representations as *this forest extracts them* — a lower bound on what each representation holds, not a measurement of it.
**Where.** `RF_SETTINGS`, `fit()`, `run()`; the validation table.
**Why.** Tuning per representation would fold model capacity into the comparison; fixing it keeps the question "which features?" separate from "which model?".
**Watch for.** The bound bites: a radius-3 environment set contains everything the radius-1 set does and cannot hold less information, yet the untuned forest scored it lower (0.846 against 0.899) because it spread its splits over 4,069 sparse columns. When a superset scores below its subset, the statement is about the estimator, and the notebook must say so rather than rank the representations.
**Source.** Wu, Z. et al. (2018). *MoleculeNet: a benchmark for molecular machine learning.* Chem. Sci. 9:513–530 (representations compared under fixed models); on sparse high-dimensional inputs and forests, Hastie, Tibshirani & Friedman (2009), §15.3.

### 58. One-parameter calibration with a floor, chosen on validation
**What.** Shrink every predicted distribution toward uniform, $q' = (1-\lambda)q + \lambda/K$, with $\lambda$ picked on the validation side from a grid whose smallest value is 0.005. One parameter is enough to stop a forest's over-confidence being charged to a representation, and the floor keeps any probability from being exactly zero, which would cost one vote an infinite log-loss.
**Where.** `calibrate()` and `LAMBDAS` in `scripts/aqsoldb_harness.py`; every later notebook reuses them.
**Why.** The forests' probabilities were nearly usable as they came (chosen $\lambda$ 0.005–0.075), which is itself worth knowing; without the floor, a single unseen tier would have made a representation look infinitely bad.
**Watch for.** $\lambda$ chosen on an in-distribution validation side cannot hedge for a shifted test side (entries 42 and 68). The grid and floor are choices; state them in the framework.
**Source.** Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting good probabilities with supervised learning.* ICML 2005; Guo, C. et al. (2017). *On calibration of modern neural networks.* ICML 2017 (one-parameter post-hoc calibration).

### 59. Correct the standard error for overlapping training sets
**What.** Ten cross-validation scores are not ten independent draws: the training sets share most of their rows. Report the mean over the $J$ splits with the corrected standard error $\sqrt{(1/J + n_\text{test}/n_\text{train})\,s^2}$, where $s^2$ is the variance of the split scores.
**Where.** `corrected_se()` in the harness; `CORRECTION` and `test_to_train()`; every ± in every table.
**Why.** The naive $s/\sqrt{J}$ would make a 0.003-bit difference look significant; with the correction, the 131- and 152-descriptor sets are a tie, which is the honest reading.
**Watch for.** The correction is a heuristic (there is no unbiased estimator of cross-validation variance); it widens the interval by the right order, not exactly. Pair differences per split before applying it (entry 62).
**Source.** Nadeau, C. & Bengio, Y. (2003). *Inference for the generalization error.* Machine Learning 52:239–281; Bengio, Y. & Grandvalet, Y. (2004). *No unbiased estimator of the variance of K-fold cross-validation.* JMLR 5:1089–1105.

### 60. Reduce correlated descriptors on your own data, blind to the label
**What.** Group descriptors so every pair in a group has $|\rho_\text{Spearman}|$ at or above a chosen level, and let the member most correlated with the rest stand for the group. Run the grouping on all compounds: it never sees the target, so it does not leak into the test folds. Sweep the level so the cost of reduction is a number.
**Where.** `correlation_groups()`, `LEVELS`; RDKit at $|\rho| \ge 0.8$ (131), $0.9$ (152), full (187).
**Why.** Correlation depends on the dataset, so a published decorrelated list is a list for someone else's compounds. Here reduction from 187 to 131 cost nothing the data could detect.
**Watch for.** Spearman tolerates the heavy tails of RDKit descriptors where Pearson does not. The representative choice decides which name appears in every later table (which pair the EBM selects, which descriptor the tree splits on); carry the group membership along so a reader can expand a name.
**Source.** Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*, Ch. 3 (between-predictor correlations); Harrell, F. E. (2015). *Regression Modeling Strategies* (2nd ed.), §4.7 (variable clustering).

### 61. Unfolded substructure counts and random, untrained message passing as fixed graph representations
**What.** Two label-free graph representations a forest can be pointed at. (a) Morgan environments kept unfolded: one column per 32-bit environment identifier holding its count, duplicates of covered bonds retained so every bonded atom contributes one environment per radius, rare environments dropped by a declared cutoff; the expected number of hash collisions is printed. (b) Chemprop's directed-bond message passing with weights drawn at random and never trained, summed over atoms; repeated over several seeds and the between-seed spread reported.
**Where.** `environments()`, `describe_environment()`, `MIN_COMPOUNDS`; `message_passing()`, `draw_weights()`, `REPEATS`.
**Why.** Unfolded counts make every feature a readable substructure, which a folded fingerprint is not. Random message passing is a cheap test of whether *connectivity* (not just composition) carries information the descriptors lack: it did not, for this forest.
**Watch for.** Use the CIP label for chirality; RDKit's raw chiral tag depends on atom order in the SMILES. Random projections are not individually readable, so exclude them from importance tables and say so. Report the seed spread (0.02 bits here) or one lucky draw is a result.
**Source.** Rogers, D. & Hahn, M. (2010). *Extended-connectivity fingerprints.* J. Chem. Inf. Model. 50:742–754; Yang, K. et al. (2019). *Analyzing learned molecular representations for property prediction.* J. Chem. Inf. Model. 59:3370–3388; Rahimi, A. & Recht, B. (2007). *Random features for large-scale kernel machines.* NIPS 2007.

### 62. The one-standard-error rule, a size front, and a carry-forward rule fixed before the audit
**What.** Compare every set with the best through its per-split paired difference and the corrected standard error; sets within one SE are tied with the best. Then trace the information–size front: from fewest features up, a set joins only if it beats every smaller set by more than one SE. Decide three things on validation alone — which sets to inspect, which to audit on test, and which to carry downstream (the last step on the front) — and write the carry rule down before any test score is seen. Audit the chosen sets on the random test folds and on the cluster splits.
**Where.** `clearly_better()`, `INSPECT`, `CHOSEN`, `DOWNSTREAM`; the audit tables.
**Why.** Choosing on validation makes validation scores optimistic; the audit showed the chosen sets reproduced them to within 0.005 bits, so the choosing flattered nothing. The front turns "which is best?" into "what does each extra feature buy?" — the same shape the interpretable-models front takes later (entry 51).
**Watch for.** A set can join the front by the letter of the rule and not by anything the data show (152 over 131 by 0.003 bits with an SE of the same size); say so and show what the choice was worth on test (0.002 bits). Average seeds per split before comparing so a lucky draw is not rewarded.
**Source.** Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984). *Classification and Regression Trees*, §3.4.3 (the 1-SE rule); Hastie, Tibshirani & Friedman (2009), §7.10.

### 63. Permutation importance by correlated group, shuffled across compounds, with calibration held fixed
**What.** Shuffle a feature across compounds (not rows, so each compound still counts once) on the validation folds and measure the drop in information with the split's $\lambda$ unchanged. Shuffle correlated features together — features linked by $|\rho| \ge 0.8$ directly or by a chain form one group — because shuffling one alone lets the forest fall back on its partner and both look unimportant. List only groups whose loss clears two standard errors, with group size and members.
**Where.** `permutation_importance()`, `GROUPS`, `SHUFFLE_LEVEL`, `show_importance()`.
**Why.** The forest's built-in importance is measured on training rows and favours continuous features; permutation on held-out rows measures what the model actually uses. Group shuffling is what made "MolLogP and the size group do the heavy lifting" a defensible statement across every descriptor set.
**Watch for.** Losses do not add up to the model's information (groups overlap in what they say). A loss larger than the model's whole information (ESOL without MolLogP) means the model became confidently wrong, which information charges and accuracy would not. Chaining can make groups large; print the size.
**Source.** Breiman, L. (2001). *Random forests.* Machine Learning 45:5–32; Strobl, C. et al. (2008). *Conditional variable importance for random forests.* BMC Bioinformatics 9:307; Gregorutti, B., Michel, B. & Saint-Pierre, P. (2015). *Grouped variable importance with random forests.* Comput. Stat. Data Anal. 90:15–35.

### 64. Beating a benchmark on information while losing on accuracy is not a contradiction
**What.** On the compounds where the repeat-measurement benchmark exists, the descriptor forest shared more information with the measured tier than a second measurement does (0.99 against 0.963 bits) while agreeing with it less often (71–72% against 76.5%). A probability spread over the plausible tiers carries more information than one noisy draw that commits to a tier; the information score credits the hedge, the hard-tier score cannot.
**Where.** The benchmark rows of the validation and test tables; the "Against the benchmark" paragraph.
**Why.** It confirms the framework's caution (entry 37): the repeat benchmark is a floor for a good model, not a ceiling. It also fixes which score the project ranks by — the one the tier design was built on.
**Watch for.** Always report both; a reader who wants a hard label is entitled to the accuracy, and the two can rank models differently (entry 42).
**Source.** Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules, prediction, and estimation.* JASA 102:359–378; Cover & Thomas (2006), §2.8.

---

## L. Learning to split (`05_learning_to_split.ipynb`)

### 65. Reimplement a short published loop instead of installing an incompatible package
**What.** When the published code cannot respect a constraint you need (here: keep a skeleton on one side; handle a compound with votes in several tiers) and drags in dependencies that will not install, rewrite the loop and keep every default of the original — the same network for predictor and splitter, the same optimizer, losses and stopping rules — with each default noted in the code where it is set.
**Where.** `learning_to_split()`, `train_predictor()`, `train_splitter()`, `splitter_loss()`, `make_mlp()`; the framework bullet "Reimplemented, with the package's predictor".
**Why.** Fidelity to the published defaults keeps the result comparable to the paper; the rewrite is what made skeleton grouping and tier distributions possible at all.
**Watch for.** State what you changed on purpose (the gap measured in information rather than accuracy; the predictor's target is a tier distribution) separately from what you kept. The split found is the hardest for an MLP, so also score the model the features were chosen with.
**Source.** Bao, Y. & Barzilay, R. (2022). *Learning to split for automatic bias detection.* arXiv:2204.13749; code at github.com/YujiaBao/ls.

### 66. Adversarial splitting with three balanced losses, several seeds, and an overlap check
**What.** Each round: deal skeletons by the splitter's probabilities, train a fresh predictor on two thirds of the train side, measure the gap between inner-validation and test information, keep the round with the largest gap, then train the splitter on three equally weighted losses — gap (send what the predictor got right to train), size (hold the train share at 0.9) and tier balance (keep each side's tier mix at the overall mix). The result is random, so run several seeds and measure how much their test sides overlap by Jaccard index against the $q/(2-q)$ expected of independent random sides of share $q$.
**Where.** The algorithm section; `SEEDS`, `RATIO`, `SHARE`; the overlap table (0.46–0.54 observed against 0.13 expected).
**Why.** The balance loss is what stops "hard" meaning "a tier moved into test"; the overlap check is what turns five runs into one object (the core) worth examining.
**Watch for.** The size loss is one of three and does not pin the test share; report the shares actually found. Inner-validation information rises above any cross-validation score once the hard compounds are gone — that is a property of the method, not evidence of a better model.
**Source.** Bao & Barzilay (2022); on repeated-run agreement as the test of a found structure, Meinshausen & Bühlmann (2010) as in entry 43.

### 67. Negative information means confidently wrong; read it as miscalibration under shift as much as missing knowledge
**What.** Information is $H(T)$ minus log-loss; a model that gives the measured tier less probability than a uniform guess scores below zero. On the learned test sides the MLP reached −2.3 bits and the forest −0.6, with the top tier right in only 18–26% of cases, under the 32% of always guessing the commonest tier. The shrinkage $\lambda$ was chosen on a validation side carved from the (easy) train side, so it could not hedge for a test side the model disagrees with.
**Where.** The three-scheme table and its reading; `carve_validation()`.
**Why.** It is the honest situation of any model deployed on such compounds, and it is the reason every later family is scored on the learned splits too: the family that hedges best under shift (the equations, entry 52) is not the one with the most bits.
**Watch for.** Do not fix the negative sign by tuning $\lambda$ on the test side; that would be the leak the scheme exists to expose. Report accuracy beside information so a ranking failure and a confidence failure can be told apart (entry 42).
**Source.** Ovadia, Y. et al. (2019). *Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift.* NeurIPS 2019.

### 68. Rule out the simple explanations in order, then ask the multivariate question
**What.** Before interpreting a hard subset, test the three cheap explanations with the yardsticks already in the project: a shifted tier mix (compare to the overall mix), unfamiliar chemistry (entry 55's nearest-training similarity), and a shifted descriptor range (per-descriptor AUC of telling the subset from the rest; 0.5 means identical distributions). Only when all three fail, train a classifier to tell the subset from the rest on skeleton-grouped held-out folds; its AUC says whether the subset is a region of feature space.
**Where.** `test_auc()`, the per-descriptor AUC table (0.43–0.55), the core-membership forest (held-out AUC 0.77).
**Why.** The ordering produced the notebook's central finding: no single descriptor sees the core, the 152 together do, and the test chemistry is as familiar as random folds — so the core is a region where the solubility the descriptors imply and the solubility measured systematically part ways.
**Watch for.** A classifier two-sample test must be grouped by the same unit as every other split or it will find the skeleton, not the region. An AUC near 0.5 per feature does not rule out a multivariate signature; say which of the two was tested.
**Source.** Lopez-Paz, D. & Oquab, M. (2017). *Revisiting classifier two-sample tests.* ICLR 2017; Hanley, J. A. & McNeil, B. J. (1982). *The meaning and use of the area under a ROC curve.* Radiology 143:29–36.

### 69. Characterize a hard core by how many runs selected each compound, and carry the uncheckable error rate as a prior
**What.** Count, per compound, how many of the seeds' test sides it fell in; group by that count and ask the same questions of each group: is it hard on random splits too (the forest's correct share, 21% in the core against 91% for never-selected); is it near a tier boundary (median 0.43 against 0.74 logS); are its repeats discordant; does it carry a flagged vote. Where most of the core was measured once (809 of 912), the prep notebook's flagged-error rate (~6%) is the prior for how many hold a wrong label — the same order as the core's single-vote members.
**Where.** The core table; the PCA and logS-against-logP figures; the 24 drawn structures.
**Why.** It sharpens "hard" into two named hypotheses — the value is wrong, or a condition the structure does not record — and says what data would settle them (replicates, sources, a literature check), which is the hand-off the next notebook takes.
**Watch for.** Flags cannot explain a single-vote core by construction (they need 3+ votes); say that before reporting that flags are no more common in the core. Discordant repeats sit just outside the core because a compound split across tiers can be at most partly right, which the gap loss pushes to test.
**Source.** Northcutt, C. G., Athalye, A. & Mueller, J. (2021). *Pervasive label errors in test sets destabilize machine learning benchmarks.* NeurIPS 2021 Datasets & Benchmarks; entry 26 for the prior.

---

## M. Nearest neighbours as a probe (`06_knn.ipynb`)

### 70. Use a model that fits no function to separate the representation from the architecture
**What.** A $k$-nearest-neighbour model predicts a compound's tier distribution as the mean of its $k$ nearest training compounds' distributions. It asks a representation directly whether compounds that look alike have alike solubility: two compounds identical in a representation but in different tiers defeat every model on it, and neighbours extend that from identical to nearly identical. Run it on the descriptors and on structural fingerprints that see the molecule differently, under all three split schemes.
**Where.** `knn_proba()`, `neighbors()`, `REPRESENTATIONS`; the three-scheme table against the forest.
**Why.** It answered the learning-to-split notebook's open question: on the learned splits every fingerprint fell as far as the descriptors (−0.8 to −1.0 bits), so the hard core is not something the descriptors hide and a fingerprint sees.
**Watch for.** Neighbours are only as good as the metric; sweep $k$, weighting, scaling and distance on validation only (entry 71). A neighbour model with near-unanimous neighbours on the wrong tier is *more* confidently wrong than a forest; read the sign with entry 67.
**Source.** Cover, T. & Hart, P. (1967). *Nearest neighbor pattern classification.* IEEE Trans. Inf. Theory 13:21–27; Maggiora, G. M. (2006). *On outliers and activity cliffs — why QSAR often disappoints.* J. Chem. Inf. Model. 46:1535.

### 71. Sweep the metric on validation, and count what a representation cannot tell apart
**What.** For each representation, score every setting ($k$, 1/distance weights, and for descriptors the scaling and distance) on the validation sides of the random splits and keep the best; only then score test sides. Before any scoring, count per representation the compounds identical to at least one other — a representation's blind spots are part of its description.
**Where.** `KS`, `WEIGHTS`, `SCALINGS`, `DISTANCES`, `BEST`, `SETTING`; the identical-compound table; `map4_shingles()`.
**Why.** The chosen metric (Manhattan on z-scores, 1/distance, $k = 20$) is not the obvious one, and the sweep is what makes the neighbour verdict on a representation fair. MAP4 gives every simple inorganic salt and every single atom the same fingerprint; said up front, that limits a later reading instead of surprising it.
**Watch for.** When a reference implementation will not install (MAP4's `distutils` setup and `tmap` dependency), rebuild the fingerprint from its published definition with an in-library equivalent (RDKit's MHFP MinHash) and say which estimator of which similarity replaced which. Fit scalings on all compounds only if they do not see the label.
**Source.** Capecchi, A., Probst, D. & Reymond, J.-L. (2020). *One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome.* J. Cheminf. 12:43; Probst, D. & Reymond, J.-L. (2018). *A probabilistic molecular fingerprint for big data settings.* J. Cheminf. 10:66.

### 72. Classify each compound by what its neighbours say, and cross-tabulate two representations' verdicts
**What.** For every compound on its one random test side, compute agreement (the chance a random neighbour vote matches a random vote of the compound) and the neighbours' top-tier share. Three patterns follow: neighbours agree with it; neighbours agree on another tier (the compound is the odd one out); neighbours mixed (the region mixes tiers). Group by learned-split selection count to see the core's pattern, then cross-tabulate the descriptor pattern against each fingerprint's pattern for core compounds and check whether "another tier" is the *same* other tier.
**Where.** `vote()`, `PATTERNS`; the pattern table by selection count; the cross-tabulation and its printed same-tier check (93–95%).
**Why.** Averages over test sides cannot distinguish "the descriptors miss a difference the structure shows" from "the value disagrees with every view of the structure"; the two cells of the cross-tab can. Here only 18–41 of 508 core compounds had structural analogues backing their value.
**Watch for.** Report relative neighbour distance alongside: the core is *not* a sparse region (ratio 1.0), which rules out "no neighbours to learn from". Fingerprint neighbours are a noisier witness than descriptor neighbours; their verdict matters because it agrees, not on its own.
**Source.** Stumpfe, D. & Bajorath, J. (2012). *Exploring activity cliffs in medicinal chemistry.* J. Med. Chem. 55:2932–2942; on agreement across methods as evidence, Campbell, D. T. & Fiske, D. W. (1959). *Convergent and discriminant validation by the multitrait-multimethod matrix.* Psychol. Bull. 56:81–105.

### 73. When every representation and every model agree, name what is left and what data would settle it
**What.** Across four representations and three architectures (MLP, forest, neighbours), the core compounds are the odd one out: structurally like them by any measure, their neighbours sit in another tier. What remains is outside the structure as encoded — the value itself (an error the flags could not reach) or a condition the structure does not record (ionization state, crystal form, temperature). State the remaining hypotheses and the data that would separate them: the source of each value, replicate measurements, a literature check of a sample.
**Where.** The "Reading" section's third and fourth points; the hand-off to the interpretable-models plan with the neighbour model as one front point (0.86 bits).
**Why.** A notebook series that ends in "the model could be better" has not finished; one that ends in "the structure is exhausted; here is the measurement to make" has. It is also the honest boundary of a project that holds no data beyond the structure.
**Watch for.** Keep the caveat that the conclusion rests on agreement between imperfect witnesses, and keep the ceiling open (a learned notion of nearness may find more, as the forest's lead shows).
**Source.** Platt, J. R. (1964). *Strong inference.* Science 146:347–353 (exclude alternatives, then name the next decisive measurement).
