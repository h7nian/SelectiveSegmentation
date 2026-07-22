# Mechanism study: turning the current interpretation into evidence

## What the existing data establish

The current results establish an asymmetric empirical pattern, not yet its
cause. Across the ten target conditions, HD-M32 has higher Spearman agreement
with realized HD safety than Dice-M32 in every condition. For Dice safety,
Dice-M32, SDC, and foreground entropy are usually close. In nine non-pathological
cells, Dice-M32 and foreground entropy have score--score Spearman correlation
between 0.916 and 0.960; FIVES/CLIP-T is the outlier at 0.535. Dice-Exact changes
the Dice-risk Spearman coefficient by at most about 0.004 relative to Dice-M32.

These observations support three statements:

1. the HD result is not an integration-resolution artifact;
2. the Dice level-set score contains little rank information beyond strong
   single-map summaries in most cells;
3. FIVES/CLIP-T is a qualitatively different failure regime and should be shown,
   not hidden.

They do **not** by themselves prove that Dice errors are detection-level, that
the true boundary posterior is nested, or that HD superiority is geometrically
inevitable.

## Locked decisive experiment: change the posterior, not the action

The fixed action is the mask from the three-checkpoint mean probability at 0.5.
The only primary variable is the source of candidate masks:

- `LevelSet-Q`: 32 masks obtained by thresholding the mean probability map;
- `Ensemble-Q`: three masks obtained by thresholding the three member maps at
  0.5.

For each of Dice, HD, and HD95, both scores average the same loss to the same
fixed action. This separates posterior support from action quality. The analysis
contract is stored in
`configs/auxiliary/ensemble_posterior_analysis_v1.json`. Ten independent jobs
also compute threshold-IoU stability, all-member IoU, mean pairwise Dice, mutual
information, and probability variance.

### Presentation

Use one main mechanism figure and one compact table:

- **Figure: posterior support and consequence.** Left: a nested level-set chain;
  middle: non-nested ensemble masks; right: paired AURC difference
  `Ensemble-Q minus LevelSet-Q` for each of ten conditions, faceted by risk.
- **Table: methods as rows, datasets as columns.** Separate CLIP-T and DL-T
  panels. Report AURC x100 for LevelSet-Q, Ensemble-Q, SDC, threshold stability,
  pairwise Dice, and mutual information under all three risks. Keep the method
  set symmetric and mark top-1 with the existing blue style.
- Put all-member IoU, probability variance, Kendall tau, runtime, and complete
  bootstrap intervals in the appendix.

Interpretation is fixed before output inspection: at least 7/10 Dice-condition
wins plus a negative mean AURC difference supports the expressivity hypothesis.
Anything else is inconclusive; it does not authorize replacing datasets.

### Frozen outcome

The Dice expressivity gate failed. Ensemble-Q was lower in only 1/10 Dice
conditions; its mean and median AURC x100 differences relative to LevelSet-Q
were +2.763 and +0.427. The HD LevelSet-Q gate passed: LevelSet-Q was lower in
10/10 conditions, with mean Ensemble-Q-minus-LevelSet-Q difference +3.820.
For HD95, LevelSet-Q was lower in 9/10, with mean difference +4.391. Therefore
three hard masks from independently trained checkpoints are not a useful proxy
for the conditional annotation posterior here. This negative result does not
show that non-nested annotation support is irrelevant; it shows that raw
training-seed disagreement and annotation variability must not be conflated.

## Additional experiment 1: error morphology audit

This tests the claim that Dice failures lie outside a boundary-displacement
family.

For each image, partition the symmetric-difference error into pixels within and
outside a fixed boundary band. Use radii 1%, 2%, and 5% of the image diagonal,
with 2% designated primary. Also record missed truth components, spurious
prediction components, largest missed-component area, and the maximum distance
of an error component from the deployed boundary.

Report:

- the fraction of Dice error outside the boundary band;
- Dice-M32 minus SDC rank residual as a function of that fraction;
- AURC within prespecified low/middle/high morphology strata;
- the same analysis for Ensemble-Q.

The mechanism is supported only if LevelSet-Q degrades as off-boundary/component
error grows and Ensemble-Q disagreement captures those errors better. The
stratified results are descriptive because strata use labels.

## Additional experiment 2: does HD add spatial information?

The claim is not merely that HD-M32 correlates with HD risk; it is that it adds
spatial-extreme information beyond pixel uncertainty.

Use leave-one-dataset-out prediction of realized HD safety with nested model
families:

1. entropy, mean maximum probability, SDC, foreground fraction, and action size;
2. the same features plus HD-M32.

Fit only on four datasets and evaluate rank correlation and AURC on the held-out
dataset. A consistent held-out improvement from adding HD-M32 supports
incremental spatial information without fitting and testing on the same images.
Complement this with a synthetic matched-entropy intervention: hold the number
and probability values of uncertain pixels fixed while moving their component
progressively farther from the main object. Entropy should remain fixed while
HD-M32 and realized HD change.

Present a five-fold held-out improvement forest plot and a three-panel synthetic
example (near, intermediate, far).

## Additional experiment 3: calibration intervention with an unchanged action

Fit one scalar temperature per dataset/model on validation pixels only and set
`p_T = sigmoid(logit(p) / T)`. This preserves the 0.5 deployed mask exactly, so
any score change is not an action-quality effect.

Before looking at test AURC, record temperature, validation NLL objective, test
Brier/ECE, and the fixed list of scores. Then compare raw versus calibrated
Dice-M32, HD-M32, SDC, and entropy. If calibration improves Dice-M32 relative
to the baselines, the bottleneck was partly marginal calibration. If calibration
improves ECE but not the relative Dice ranking, the posterior-support explanation
becomes more credible.

Present calibration change and AURC change side by side; do not select between
raw and calibrated results on the test set.

## Additional experiment 4: qualitative disagreement cases

Select cases algorithmically, not visually: the top three images per dataset by
absolute normalized rank disagreement between HD-M32 and foreground entropy,
plus the top three between Ensemble-Q-Dice and LevelSet-Q-Dice. Show image,
truth, deployed mask, probability map, representative level sets, three ensemble
masks, and the three realized losses. This makes “far-away boundary excursion”
and “component on/off disagreement” inspectable.

## Additional experiment 5: a Dice-matched coupling ladder

Dice does not directly see connectivity or Euclidean position.  For a fixed
action (A), its candidate-mask loss depends only on

\[
  Z=|Y\cap A|,\qquad W=|Y\setminus A|.
\]

Consequently, connected-component switching is a plausible model of
region-level error, but it is not a mathematical consequence of Dice.  Test it
against a parameter-free coupling that is matched directly to these two
sufficient counts.  Every construction below must retain the same pixel
marginals: a pixel (i) belongs to a block (b(i)), each block receives an
independent (U_b\sim\mathrm{Unif}(0,1)), and
(Y_i=\mathbf 1\{p_i\ge U_{b(i)}\}).

Use the following prespecified ladder on the ten target conditions, always with
the same (0.5) action:

1. **Global level set:** one block, the current comonotone posterior.
2. **Action two-block:** (A) and (A^c) have independent thresholds.  This is
   the primary Dice-specific alternative because it directly separates (Z)
   from (W), has no spatial hyperparameter, and admits deterministic
   (32\times32) midpoint evaluation.
3. **Action components:** four-connected components of (A) and (A^c) receive
   independent thresholds.  This tests component-level switching without a
   low-probability proposal threshold.
4. **Proposal components:** four-connected components of
   \(\{p\ge0.1\}\), plus its complement, receive independent thresholds.  Fixed
   thresholds 0.05 and 0.20 are sensitivity analyses, not candidates from
   which to select the best test result.
5. **Independent pixels:** the other coupling endpoint, included as a negative
   control rather than a proposed segmentation posterior.

The primary estimand is Dice-risk AURC x100 for action-two-block minus global
level-set.  The interpretation gate is the same as the Ensemble-Q experiment:
at least 7/10 condition wins and a negative mean paired difference.  Component
constructions are secondary.  Report score correlation with SDC and foreground
entropy, Monte Carlo repeat error where applicable, number and size of blocks,
and runtime.  The claim “region coupling adds Dice information” is supported
only when an object-aligned component construction improves over both the
global and action-two-block constructions and a spatial-grid block control with
the same block-count distribution.

This experiment answers a different question from Ensemble-Q.  The coupling
ladder asks how much can be recovered from one probability map after changing
only its joint-mask model; Ensemble-Q estimates the additional information
available from training-run diversity.  Showing both places the single-map
method between a marginal-information baseline and a higher-budget empirical
posterior without calling either one the true conditional label posterior.

The frozen action-two-block result passed its directional gate (8/10), but its
mean and median AURC x100 gains were only 0.0195 and 0.0020, and its Spearman
correlation with Dice-M32 was at least 0.99936. Thus dependence can change the
expected random ratio, but merely decoupling inside and outside counts does not
create a practically different Dice ranking.

### More fundamental formulation and presentation

The common object is the Dice count pushforward

\[
  R_A=\mathcal L\bigl(|Y\cap A|,|Y\setminus A|\mid X\bigr),
\]

not a particular full-mask sampler.  A main mechanism figure should show, for
predeclared disagreement cases, the support induced in the ((Z,W)) plane:

- the global level-set construction is a one-dimensional chain
  \((z(t),w(t))\);
- the action-two-block construction is its Cartesian product
  \((z(t_j),w(t_k))\);
- Ensemble-Q contributes three empirical count pairs;
- the reference-label count pair is overlaid only as a post-analysis outcome.

Beside this support panel, report the ten paired Dice-AURC differences.  The
support plot establishes expressivity; only AURC establishes usefulness.  A
label-dependent support-distance diagnostic may quantify how far the observed
count pair lies from each support, but it must not define or tune confidence.

The two-block and Ensemble-Q outcomes make interpolation between their scores
unlikely to produce a substantive new ordering. The more principled method is
instead a **calibrated count posterior**. For fixed action \(A\), estimate

\[
  \widehat R_A
  \approx
  \mathcal L\bigl(Z=|Y\cap A|,W=|Y\setminus A|\mid X\bigr)
\]

from out-of-fold labeled images or genuine repeated annotations. Useful
predictors remain functions of the frozen probability map: inside/outside
probability mass, uncertainty, action size, and the component-mass spectrum.
The confidence is then the expectation of the Dice ratio under this calibrated
bivariate law, rather than a full synthetic mask posterior.

Because one map does not identify count dependence, the robust version should
expose this uncertainty:

\[
  C_{\rm robust}(x)=
  \inf_{R:\,W_{1,d_A}(R,\widehat R_A)\le\varepsilon_x}
  \mathbb E_R\left[\frac{2Z}{|A|+Z+W}\right].
\]

The count-Wasserstein result in Appendix A turns this into a certified lower
confidence whenever the true count pushforward lies in the ambiguity set.
This is the substantive distinction from SDC: SDC inserts only the two expected
counts into a ratio; the count posterior models the random numerator and
denominator, while the robust version acknowledges uncertainty about their
coupling. A validation-only scalar mixture remains a useful ablation, not the
proposed endpoint.

The immediate no-training experiment remains the partition ladder with a
matched spatial-grid control. If action/proposal components do not beat both
the two-block and grid controls by a meaningful rank margin, stop adding
handcrafted couplings and move to cross-fitted or multi-rater count-posterior
estimation.

## Controlled extension: a spatial Gaussian-copula posterior

The partition ladder changes dependence by assigning one uniform threshold to
each discrete block.  A complementary continuous family uses a standard-normal
latent field,

\[
  Z_i=\sqrt{\alpha}\,G+\sqrt{\beta}\,U_i
      +\sqrt{1-\alpha-\beta}\,\varepsilon_i,
  \qquad
  Y_i=\mathbf 1\!\left\{Z_i\leq\Phi^{-1}(p_i)\right\}.
\]

Here \(G\), every \(U_i\), and every \(\varepsilon_i\) have unit marginal
variance, while (U\) is spatially correlated.  The square roots are essential:
they make \(\alpha\) and \(\beta\) variance weights and guarantee
\(\operatorname{Var}(Z_i)=1\).  Consequently every setting preserves the
declared pixel marginals exactly, \(\Pr(Y_i=1)=p_i\).  The endpoints are the
global shared-threshold posterior at \((\alpha,\beta)=(1,0)\) and independent
Bernoulli sampling at \((0,0)\).

The first implementation should use a coordinate-only, standardized bilinear
Gaussian field whose knot spacing is expressed as a fraction of the image
diagonal.  This keeps the one-probability-map information budget and avoids an
unusable dense covariance matrix.  Image or encoder features are a separate,
higher-information extension and must not be silently introduced into the
matched-budget comparison.  Hard-fixing pixels outside an uncertainty band is
also excluded from the marginal-preserving experiment: unless their
probabilities are exactly zero or one, fixing them changes the marginals.

This is an ablation, not an automatic replacement for the global posterior.
It loses one-dimensional exact integration, introduces Monte Carlo error and
requires validation-only choices of variance weights and spatial scale.  With
single annotations, such choices optimize selective-ranking utility; they do
not identify the true annotation posterior.  Promote the spatial posterior
only if a prespecified setting improves matched Dice-risk AURC in at least
7/10 target conditions with a non-negligible macro effect, stable repeat error,
and acceptable runtime.  A null result strengthens the simpler global method.

Each condition--setting--repeat is one independent Slurm job.  A repeat job
records one fixed-seed Monte Carlo estimate; a separate aggregation step checks
sample identities, source hashes, posterior parameters, and repeat indices
before computing the mean and repeat standard deviation.  The same command is
controlled by arguments for posterior draws, repeat index, variance weights,
spatial knot spacing, device, and posterior batch size.  No setting-specific
Python or Slurm scripts are allowed.

## Baselines by information budget

The headline table remains a matched one-map, no-fitted-quality-predictor
comparison: SDC, mean maximum probability, mean pixel entropy, foreground
entropy, foreground/volume summaries, simple threshold stability, and the
risk-aligned scores.  Methods that consume more information are reported in
separate panels:

1. **One map plus validation fitting:** scalar temperature scaling, with the
   deployed 0.5 action held fixed.
2. **Multiple stochastic predictions or checkpoints:** MC dropout, test-time
   augmentation, and deep-ensemble disagreement.  The three-checkpoint
   Ensemble-Q experiment already belongs here.
3. **Additional labeled training:** learned quality prediction, reverse
   classification, or a learned stochastic segmentation posterior.

This separation prevents a method from appearing stronger merely because it
uses more forward passes, checkpoints, annotations, or fitted labels.  The
temperature intervention is the highest-priority missing experiment because it
changes probability calibration without changing the deployed action.  A
multi-annotator dataset is the strongest posterior-validation experiment, but
its annotation average estimates an empirical rater risk rather than the
unobservable population ideal risk.

## What not to run yet

- Do not replace FIVES, CLIP-T, or any adverse cell after seeing results.
- Do not put RCA in the matched-information table; it needs an annotated atlas
  and per-case reverse fitting.
- Do not add MC dropout or TTA unless the three-member Ensemble-Q result is
  inconclusive because of limited member diversity. They are useful follow-ups,
  not needed to answer the first posterior-support question.
- A repeat-annotation dataset is the highest-value future validation if the
  paper wants to claim approximation to a true conditional mask posterior. With
  one reference per image, all posterior diagnostics remain falsification tests,
  not posterior validation.

## Recommended paper order

Keep the main result as risk-aligned selective ranking. State the asymmetry
first: HD gains are broad and stable, whereas Dice is competitive but redundant
with strong single-map summaries. Use the fixed-action posterior experiment to
explain that asymmetry. Put calibration and morphology audits in the appendix
unless either materially changes the interpretation.
