# Final value-proposition result

> **Timing update:** Accuracy, monotonicity, and projection results in this
> document remain current. Timing values in the original benchmark used
> unequal workflow boundaries and are superseded by `fair_timing_final/`.

The final default-setting evidence is in
`value_proposition_optimized_default/`. It uses 12 cubic spline basis
functions, MonotoneLSPIA tolerance `1e-8`, 100 paired trials per condition,
and scaling runs through 100,000 observations.

Across the nine target/noise conditions, the median MonotoneLSPIA RMSE ratio
was 1.000 relative to the penalized monotone B-spline and 1.004 relative to
pyGAM. Six of nine conditions were within the predeclared 10% accuracy margin
for each comparator. The three exceptions were all sensor-log conditions,
demonstrating the remaining model-complexity sensitivity of an unpenalized
fit with a fixed number of basis functions.

The original embedded timers reported 3.22x and 8.48x end-to-end speedups,
but the methods did not start and stop at identical workflow boundaries.
These numbers are retained only for traceability and must not be cited. The
dedicated fair benchmark reports the corrected total- and fixed-workflow
comparisons.

The projection audit reconstructed all 78 projected updates, with a maximum
pre-projection cone violation of 0.00914 and zero post-projection violation.

`value_proposition_optimized_practical/` is a secondary sensitivity analysis
using tolerance `1e-4`. Earlier folders are retained for traceability but
their timing results predate the sparse-basis optimization.
