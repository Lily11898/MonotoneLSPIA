# Final default-setting interpretation

> **Superseded timing:** Accuracy and convergence results in this folder
> remain valid. Timing values used unequal workflow boundaries and must not be
> used for publication. Use `../fair_timing_final/` for the final timing
> experiment, raw trials, IQR summaries, and interpretation.

This is the primary result after the mathematically equivalent sparse-basis
optimization. It uses the public defaults of 12 cubic B-spline coefficients
and tolerance `1e-8`, with 100 paired trials for each of nine target/noise
conditions.

| Comparator | Median RMSE ratio (ours/comparator) | Median end-to-end speedup | Joint accuracy-and-speed successes |
|---|---:|---:|---:|
| Penalized monotone B-spline | 1.000 | 3.22x | 6/9 |
| pyGAM | 1.004 | 8.48x | 6/9 |

The six dose-response and CDF-plateau conditions met both predeclared
criteria. The three sensor-log conditions did not meet the 10% RMSE margin;
their RMSE ratios ranged from 1.115 to 1.265 relative to the penalized spline
and from 1.112 to 1.285 relative to pyGAM. A separate prototype showed that
cross-validating the number of control coefficients reduces this gap, but it
adds model-selection cost and cannot guarantee uniform superiority. It is not
part of the core result.

Scaling medians were:

| Observations | MonotoneLSPIA | Penalized spline | pyGAM |
|---:|---:|---:|---:|
| 120 | 0.012 s | 0.042 s | 0.104 s |
| 1,000 | 0.019 s | 0.072 s | 0.119 s |
| 10,000 | 0.040 s | 0.198 s | 0.275 s |
| 100,000 | 0.290 s | 0.850 s | 1.854 s |

The speed improvement comes from vectorized sparse B-spline construction,
collocation reuse, and avoiding smoothing-weight selection. It is not a claim
that every individual numerical solve is faster. When the selected penalty
is already known, the SLSQP penalized spline refits faster than
MonotoneLSPIA.

All 900 MonotoneLSPIA fits converged. The projection audit independently
reconstructed all 78 projected updates; the maximum pre-projection cone
violation was 0.00914 and the maximum post-projection violation was zero.

The supported conclusion is therefore conditional but strong: the default
MonotoneLSPIA workflow provides competitive median recovery accuracy, exact
and auditable feasibility, and materially lower end-to-end cost over the
tested range. Uniform recovery superiority is neither claimed nor supported.
