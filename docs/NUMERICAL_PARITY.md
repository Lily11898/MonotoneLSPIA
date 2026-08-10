# Numerical parity with MATLAB 1.0.0

The Python release was compared directly with the MATLAB R2021b reference
implementation on 2026-08-03. Both implementations used identical input
arrays and default numerical options.

| Case | Control points | Iterations | Maximum coefficient difference | Maximum prediction difference |
|---|---:|---:|---:|---:|
| Increasing, arbitrary predictor scale | 10 | 121 | `4.80e-14` | `4.97e-14` |
| Decreasing | 9 | 153 | `4.22e-15` | `5.11e-15` |
| Quadratic, custom knots | 5 | 52 | `5.11e-15` | `5.11e-15` |

The largest difference among RMSE, iteration count, and step size was
`4.16e-16`; iteration counts were identical in every case. These deviations
are consistent with floating-point and CSV serialization effects.

The automated Python suite additionally compares the sparse B-spline design
matrix with `scipy.interpolate.BSpline.design_matrix` and exercises both
monotonicity directions, custom knots, repeated predictors, invalid inputs,
rank-deficiency reporting, and bundled example data.
