# MATLAB-to-Python migration map

The Python 1.0.0 release preserves the numerical conventions of the MATLAB
1.0.0 reference release.

| MATLAB | Python |
|---|---|
| `monotoneLspiaFit(x,y,...)` | `fit(x, y, ...)` |
| `monotoneLspiaPredict(model,x)` | `predict(model, x)` or `model.predict(x)` |
| `monotoneLspiaBasisMatrix(t,n)` | `basis_matrix(t, n)` |
| `monotoneLspiaPlot(model)` | `plot(model)` |
| `monotoneLspiaVersion` | `version()` |
| name-value `NumControlPoints` | keyword `num_control_points` |
| name-value `FixEndpoints` | keyword `fix_endpoints` |
| MATLAB sparse matrix | SciPy CSR matrix |
| MATLAB structure | `MonotoneLSPIAModel` dataclass |

For easier automated comparison, snake-case aliases beginning with
`monotone_lspia_` are also exported.

The test suite verifies partition of unity, equality with SciPy's B-spline
design matrix, increasing and decreasing fits, repeated predictors, custom
knots, deterministic runs, constant responses, rank-deficiency diagnostics,
history storage, extrapolation policy, metadata, and bundled data examples.
