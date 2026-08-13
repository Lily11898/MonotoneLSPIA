# Cross-implementation verification

During development, the numerical core of MonotoneLSPIA was cross-checked
against an earlier internal MATLAB prototype. The prototype was used for
algorithm exploration and numerical verification and was not released as a
standalone software product. MATLAB is not required to install, use, test, or
reproduce the Python package.

The comparison was performed using MATLAB R2021b on 2026-08-03. Both
implementations received identical input arrays and equivalent numerical
options.

| Case | Control points | Iterations | Maximum coefficient difference | Maximum prediction difference |
|---|---:|---:|---:|---:|
| Increasing, arbitrary predictor scale | 10 | 121 | `4.80e-14` | `4.97e-14` |
| Decreasing | 9 | 153 | `4.22e-15` | `5.11e-15` |
| Quadratic, custom knots | 5 | 52 | `5.11e-15` | `5.11e-15` |

The largest difference among RMSE, iteration count, and step size was
`4.16e-16`; iteration counts were identical in every case. These deviations
are consistent with floating-point and CSV serialization effects.

This comparison is a cross-implementation consistency check rather than an
indication that MATLAB forms part of the released package. The automated
Python test suite additionally compares the sparse B-spline design matrix with
`scipy.interpolate.BSpline.design_matrix` and tests both monotonicity
directions, custom knots, repeated predictors, invalid inputs, rank-deficiency
reporting, persistence, and bundled examples.
