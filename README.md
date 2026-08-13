# MonotoneLSPIA for Python

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21900832.svg)](https://doi.org/10.5281/zenodo.21900832)

MonotoneLSPIA is a native Python package for fitting increasing or decreasing
B-spline curves using Projected Least-Squares Progressive Iterative
Approximation. It is designed for direct use within the NumPy/SciPy scientific
computing ecosystem and provides sparse collocation, support for repeated and
unsorted predictor values, numerical and shape diagnostics, prediction,
plotting, and versioned model persistence.

During development, the numerical core was cross-checked against an earlier
internal MATLAB prototype for increasing, decreasing, and custom-knot
configurations. Coefficients and predictions agreed to approximately `5e-14`
in the tested cases. This comparison is retained as cross-implementation
verification; MATLAB is not required to install, use, test, or reproduce the
Python package. See `docs/CROSS_IMPLEMENTATION_VALIDATION.md`.

Project links:

- Repository: <https://github.com/Lily11898/MonotoneLSPIA>
- Version 1.0.1 release: <https://github.com/Lily11898/MonotoneLSPIA/releases/tag/v1.0.1>
- Version 1.0.1 archive: <https://doi.org/10.5281/zenodo.21916433>
- Documentation: <https://github.com/Lily11898/MonotoneLSPIA/tree/main/docs>
- Issue tracker: <https://github.com/Lily11898/MonotoneLSPIA/issues>

## Installation

Python 3.10 or later is required.

```bash
python -m pip install .
```

Install plotting, data-example, and test dependencies when needed:

```bash
python -m pip install ".[plot,data,test]"
```

## Quick start

```python
import numpy as np
from monotone_lspia import fit, plot

rng = np.random.default_rng(7)
x = np.linspace(10, 80, 100)
y = np.log(x) + 0.08 * rng.standard_normal(x.size)

model = fit(x, y, direction="increasing", num_control_points=12)
y_query = model.predict(np.linspace(x.min(), x.max(), 500))
print(model.rmse, model.diagnostics.converged)
plot(model)
```

The public API is:

- `fit` / `monotone_lspia_fit`
- `predict` / `monotone_lspia_predict`
- `save` / `load`
- `basis_matrix` / `monotone_lspia_basis_matrix`
- `plot` / `monotone_lspia_plot`
- `version` / `monotone_lspia_version`

Fitted models can be stored without pickle and restored for later prediction:

```python
from monotone_lspia import load, save

save(model, "monotone_model.npz")
restored = load("monotone_model.npz")
assert np.allclose(restored.predict(x), model.predict(x))
```

## Testing and reproduction

```bash
python -m pytest
python examples/getting_started.py
python validation/run_comparative_validation.py --num-trials 2
python validation/run_publication_workflow.py --smoke --download-nasa \
  --output-root results/publication_smoke
```

Use `--num-trials 100` for the final five-method paired SoftwareX accuracy
experiment. Generated validation files are written to `results/`. The
publication comparison uses LSPIA, scikit-learn `IsotonicRegression`, a
cross-validated penalized monotone B-spline, pyGAM, and MonotoneLSPIA.

## Numerical method

For a B-spline collocation matrix `A`, observations `y`, and coefficients
`p`, one iteration is

```text
p_tilde = p_k + mu A.T (y - A p_k)
p_(k+1) = P_monotone(p_tilde),
```

where `P_monotone` is the Euclidean monotone-cone projection computed by
the pool-adjacent-violators algorithm. See `docs/ALGORITHM.md` for details.

## License and data

The software is released under the BSD 3-Clause License. The bundled UCI
Combined Cycle Power Plant data retain their separate CC BY 4.0 attribution;
see `data/CCPP/README.md`.
