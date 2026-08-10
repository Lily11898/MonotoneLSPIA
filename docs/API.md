# API reference

## `fit(x, y, **options)`

Fits a monotone B-spline and returns `MonotoneLSPIAModel`.

| Option | Default | Meaning |
|---|---:|---|
| `direction` | `"increasing"` | `"increasing"` or `"decreasing"` |
| `num_control_points` | up to 12 | Number of B-spline coefficients |
| `degree` | 3 | B-spline degree |
| `knots` | open-uniform | Knot vector on `[0, 1]` |
| `max_iterations` | 500 | Maximum projected iterations |
| `tolerance` | `1e-8` | Relative coefficient-update tolerance |
| `weight` | `"practical"` | `"practical"` or `"theoretical"` |
| `step_size` | `None` | Positive user-specified iteration step |
| `safety_factor` | `0.99` | Practical-step safety factor |
| `fix_endpoints` | `False` | Keep both endpoint coefficients exactly at compatible endpoint observations |
| `store_history` | `False` | Store each coefficient iterate |

The model stores coefficients, knots, the predictor normalization interval,
sorted training observations, fitted values, RMSE, and convergence/rank/
monotonicity diagnostics. `model.predict(x)` is equivalent to `predict(model,x)`.

With `fix_endpoints=True`, increasing fits require the left endpoint response
to be no greater than the right endpoint response; decreasing fits require the
reverse order. Incompatible endpoint observations raise `ValueError`.

## `predict(model, x_query, extrapolation="error")`

Evaluates a fitted model. The default rejects values outside the training
range. `extrapolation="clip"` applies constant boundary extension.

## `basis_matrix(parameters, num_control_points, degree=3, knots=None)`

Returns a SciPy CSR B-spline collocation matrix at normalized parameters.

## `plot(model)`

Returns `(figure, (fit_axes, history_axes))`. Matplotlib is an optional
dependency and is imported only when this function is called.

## `save(model, path)` and `model.save(path)`

Save a fitted model to the versioned, compressed MonotoneLSPIA NPZ format.
The format stores arrays without pickle and includes convergence, rank, and
monotonicity diagnostics plus the optional coefficient history.

## `load(path)`

Load a model written by `save`. The restored model can predict immediately
without access to the original training code or refitting.

## `version()`

Returns the software version string.
