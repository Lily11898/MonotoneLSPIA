# Changelog

## 1.0.0 — 2026-08-12

- Ported the MATLAB 1.0.0 numerical kernel to Python.
- Added a NumPy/SciPy public API, sparse B-spline matrix construction, model
  diagnostics, plotting, tests, examples, benchmarks, and validation scripts.
- Replaced row-wise Python B-spline assembly with SciPy's vectorized sparse
  design matrix and reused the training collocation matrix across fitting and
  diagnostics. This preserves numerical results while substantially reducing
  runtime for large observation sets.
- Added a reproducible value-proposition benchmark against a cross-validated
  penalized monotone B-spline and pyGAM, including scaling and projection-path
  audits.
- Added a separate fair-workflow timing benchmark with identical boundaries,
  warm-ups, randomized execution order, 20 repetitions, convergence tracking,
  median/IQR summaries, and environment metadata.
- Added versioned, pickle-free NPZ model persistence through `save`, `load`,
  and `model.save`, with round-trip prediction tests.
- Corrected `fix_endpoints=True` to use a fixed-boundary monotone-cone
  projection and reject infeasible endpoint order.
- Replaced dense SVD of the full collocation matrix with conservative rank and
  condition diagnostics based on the small Gram matrix.
- Preserved MATLAB-style function aliases for reference comparison.
