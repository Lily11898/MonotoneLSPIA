import numpy as np
from scipy.interpolate import BSpline

from monotone_lspia import basis_matrix
from monotone_lspia._algorithm import fixed_endpoint_projection, pava, projected_lspia
from monotone_lspia._basis import open_uniform_knots


def test_pava_known_projection():
    result = pava(np.array([3.0, 1.0, 2.0, 5.0, 4.0]))
    np.testing.assert_allclose(result, [2.0, 2.0, 2.0, 4.5, 4.5])


def test_decreasing_pava():
    result = pava(np.array([1.0, 4.0, 3.0, 2.0]), "decreasing")
    assert np.max(np.diff(result)) <= 0


def test_fixed_endpoint_projection_preserves_boundaries():
    values = np.array([-2.0, -4.0, 3.0, 1.0, 5.0])
    projected = fixed_endpoint_projection(values, -2.0, 5.0)
    assert projected[0] == -2.0
    assert projected[-1] == 5.0
    assert np.min(np.diff(projected)) >= 0.0

    decreasing = fixed_endpoint_projection(-values, 2.0, -5.0, "decreasing")
    assert decreasing[0] == 2.0
    assert decreasing[-1] == -5.0
    assert np.max(np.diff(decreasing)) <= 0.0


def test_basis_matches_scipy_design_matrix():
    t = np.linspace(0, 1, 97)
    knots = open_uniform_knots(10, 3)
    actual = basis_matrix(t, 10, degree=3, knots=knots).toarray()
    expected = BSpline.design_matrix(t, knots, 3).toarray()
    np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_degree_zero_basis():
    t = np.array([0, 0.2, 0.6, 1.0])
    matrix = basis_matrix(t, 3, degree=0).toarray()
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    assert matrix[0, 0] == 1 and matrix[-1, -1] == 1


def test_precomputed_collocation_matches_internal_construction():
    t = np.linspace(0, 1, 71)
    y = np.log1p(4 * t)
    knots = open_uniform_knots(9, 3)
    matrix = basis_matrix(t, 9, degree=3, knots=knots)
    options = {
        "direction": "increasing",
        "max_iterations": 200,
        "tolerance": 1e-8,
        "weight": "practical",
        "step_size": None,
        "safety_factor": 0.99,
        "fix_endpoints": False,
        "store_history": False,
    }
    implicit = projected_lspia(y, t, 9, 3, knots, **options)
    explicit = projected_lspia(y, t, 9, 3, knots, matrix=matrix, **options)
    np.testing.assert_array_equal(explicit[0], implicit[0])
    np.testing.assert_array_equal(
        explicit[2].error_history,
        implicit[2].error_history,
    )
