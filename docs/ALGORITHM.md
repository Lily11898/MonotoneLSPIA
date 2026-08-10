# Algorithm and numerical conventions

The implementation sorts observations by `x` and maps the finite predictor
interval to `[0, 1]`. For B-spline collocation matrix `A`, response vector `y`,
and coefficient vector `p`, Projected LSPIA performs

```text
p_tilde = p_k + mu A.T (y - A p_k)
p_(k+1) = projection_to_monotone_cone(p_tilde).
```

The pool-adjacent-violators algorithm computes the Euclidean projection.
For a degree-`d` B-spline, its derivative is a nonnegative combination of
scaled adjacent coefficient differences:

```text
s'(t) = d * sum_i ((p_(i+1) - p_i) / (u_(i+d+1) - u_(i+1)))
                   * N_(i+1,d-1)(t).
```

Terms with a zero knot denominator are omitted under the standard B-spline
convention. Consequently, nondecreasing coefficients are a sufficient
condition for a nondecreasing curve; the decreasing case follows by sign
reversal. The coefficient condition is sufficient but not necessary for
every possible monotone spline.

The practical step is

```text
mu = safety_factor * 2 / max(row_sum(A.T A)),
```

with default safety factor `0.99`. The theoretical option uses the largest
and smallest positive eigenvalues of `A.T A`. The stopping rule is the
relative Euclidean norm of the projected coefficient update.

The collocation matrix is assembled with SciPy's vectorized B-spline design
matrix and converted directly to CSR sparse form. The training matrix is
constructed once and reused throughout the projected iteration, fitted-value
calculation, and rank diagnostics. Separate sparse design matrices are
constructed for prediction or monotonicity evaluation on new input grids. It has at most
`degree + 1` stored entries per row. Default knots are open-uniform; validated
nonuniform knot vectors can be supplied.

When `fix_endpoints=True`, the two boundary coefficients are held exactly at
the boundary observations and PAVA projects only the interior coefficients
subject to those fixed bounds. Incompatible endpoint order is rejected.

Rank and condition diagnostics are computed from the small Gram matrix
`A.T @ A`; the full sparse training matrix is not converted to a dense array.
The Gram-based numerical rank is intentionally conservative for severely
ill-conditioned designs.
