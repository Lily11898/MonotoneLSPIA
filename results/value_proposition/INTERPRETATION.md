# Value-proposition validation

> **Historical pre-optimization result.** The current implementation replaces
> row-wise Python basis construction with SciPy's vectorized sparse design
> matrix and reuses the collocation matrix. Use
> `results/value_proposition_optimized_default/` for the final default-setting
> evidence. The accuracy values below remain valid, but the timing conclusions
> are superseded.

## Question

Can MonotoneLSPIA provide lower computational cost or a more transparent
constraint process than a constrained penalized B-spline and pyGAM, while
retaining acceptable recovery accuracy?

## Design

The primary experiment used 16 cubic spline basis functions and the shipped
MonotoneLSPIA tolerance of `1e-8`. Each of three monotone target functions was
observed at 120 points under noise standard deviations 0.02, 0.05, and 0.10,
with 100 paired trials per condition. The penalized B-spline selected its
roughness weight by deterministic four-fold cross-validation. pyGAM selected
its smoothing weight by the documented `gridsearch` workflow. The
predeclared reporting thresholds were an RMSE ratio no greater than 1.10 and
an end-to-end speedup of at least two.

A secondary, explicitly post-primary sensitivity experiment used 12 basis
functions and tolerance `1e-4`. It tests a practical stopping configuration;
it does not replace the primary result.

## Primary result: 16 basis functions, tolerance 1e-8

| Comparator | Median RMSE ratio (ours/comparator) | Median end-to-end speedup | Conditions meeting both criteria |
|---|---:|---:|---:|
| Penalized monotone B-spline | 1.081 | 1.213x | 0/9 |
| pyGAM | 1.113 | 2.897x | 3/9 |

This configuration does not support a general claim of substantially lower
cost at acceptable accuracy. It supports only a conditional pyGAM comparison.
With 10,000 observations, median end-to-end times were 0.396 s for
MonotoneLSPIA, 0.317 s for the penalized spline, and 0.357 s for pyGAM.

## Practical sensitivity: 12 basis functions, tolerance 1e-4

| Comparator | Median RMSE ratio (ours/comparator) | Median end-to-end speedup | Conditions meeting both criteria |
|---|---:|---:|---:|
| Penalized monotone B-spline | 1.000 | 1.790x | 0/9 |
| pyGAM | 1.004 | 4.959x | 6/9 |

The practical configuration supports a narrower statement: MonotoneLSPIA
can match the median recovery accuracy of both competitors and materially
reduce the small-sample end-to-end cost of pyGAM because it has no explicit
roughness-weight search. It still does not achieve the predeclared two-fold
speedup over the penalized spline. When the smoothing parameter is already
known, both competitors refit faster, so the advantage is parameter-selection
cost rather than a universally faster numerical kernel.

The result is target-dependent. On the sensor-log target, MonotoneLSPIA RMSE
was 11--26% above the penalized spline and 11--28% above pyGAM. On the dose-
response and CDF-plateau targets, it was generally within the 10% accuracy
threshold and was sometimes slightly more accurate.

## Projection transparency

For the practical audit, all 30 post-initialization iterations required an
active projection. Every recorded update was independently reconstructed
from the residual step followed by PAVA. The maximum pre-projection cone
violation was 0.00914 and the maximum post-projection violation was exactly
zero. The fitted model exposes the step size, residual and update histories,
constraint-violation history, and—when `store_history=True`—every coefficient
iterate.

This is an out-of-the-box auditability advantage, not a claim that competing
optimizers can never be instrumented. SciPy SLSQP can be supplied with a
callback, and pyGAM supports callbacks, but their default fitted objects do
not provide the same explicit projection identity and projected coefficient
path.

## Defensible conclusion

MonotoneLSPIA is not universally more accurate or faster. Its defensible
position is a specialized, exactly projected and auditable monotone B-spline
workflow with no explicit smoothing-weight search. Under the practical
configuration it offers competitive accuracy and substantially lower
small-sample end-to-end cost than pyGAM, while the constrained penalized
B-spline remains a strong and often faster competitor. Claims of universal
speed superiority, large-sample superiority, or uniformly better recovery
accuracy are not supported by these experiments.
