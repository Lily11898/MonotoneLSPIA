# Changelog

## 1.0.2 — 2026-08-17

- Fixed `fix_endpoints=True` for repeated predictor values at either boundary
  by fixing each endpoint coefficient to the mean response at that boundary.
- Made endpoint-fixed fits invariant to the input order of repeated boundary
  observations, with regression tests for increasing and decreasing fits.
- Updated the API documentation to describe boundary averaging explicitly.

## 1.0.1 — 2026-08-13

- Clarified that MonotoneLSPIA is released and maintained as a native Python
  research software package.
- Reframed the earlier MATLAB code as a development-time prototype used for
  cross-implementation numerical verification; MATLAB is not a runtime,
  testing, or reproduction dependency.
- Replaced migration-oriented documentation with a focused record of the
  cross-implementation validation.

## 1.0.0 — 2026-08-12

- Implemented the projected-LSPIA numerical kernel in Python and cross-checked
  it against an earlier internal MATLAB prototype.
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
- Added descriptive functional aliases for alternative calling styles and
  automated numerical comparisons.
