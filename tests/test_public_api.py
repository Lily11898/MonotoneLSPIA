from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse

import monotone_lspia as ml


def test_increasing_fit_on_arbitrary_scale():
    rng = np.random.default_rng(11)
    x = np.linspace(20, 120, 90)
    y = np.sqrt(x) + 0.12 * rng.standard_normal(x.size)
    model = ml.fit(x, y, num_control_points=10)
    prediction = model.predict(np.linspace(20, 120, 1001))
    assert np.min(np.diff(prediction)) >= -1e-11
    assert model.x_minimum == 20
    assert model.x_maximum == 120
    assert model.knots.size == model.num_control_points + model.degree + 1


def test_decreasing_fit():
    rng = np.random.default_rng(12)
    x = np.linspace(-3, 5, 75)
    y = np.exp(-0.2 * x) + 0.03 * rng.standard_normal(x.size)
    model = ml.fit(x, y, direction="decreasing", num_control_points=9)
    prediction = ml.predict(model, np.linspace(-3, 5, 1001))
    assert np.max(np.diff(prediction)) <= 1e-11
    assert model.direction == "decreasing"


def test_repeated_predictors():
    x = [0, 0, 0.2, 0.4, 0.4, 0.7, 1, 1]
    y = [0, 0.1, 0.18, 0.37, 0.42, 0.69, 0.95, 1.05]
    model = ml.fit(x, y, degree=2, num_control_points=5)
    assert model.fitted_values.shape == (8,)
    assert np.all(np.isfinite(model.coefficients))


def test_custom_knots_are_stored():
    x = np.linspace(0, 1, 30)
    knots = np.array([0, 0, 0, 0.35, 0.7, 1, 1, 1], dtype=float)
    model = ml.fit(x, x**2, degree=2, num_control_points=5, knots=knots)
    np.testing.assert_array_equal(model.knots, knots)
    assert model.diagnostics.sampled_curve_violation <= 1e-20


def test_extrapolation_policy():
    x = np.arange(6, dtype=float)
    model = ml.fit(x, x, num_control_points=4)
    with pytest.raises(ValueError, match="outside"):
        ml.predict(model, -1)
    assert ml.predict(model, -1, extrapolation="clip") == pytest.approx(
        ml.predict(model, 0), abs=1e-12
    )


def test_constant_predictor_rejected():
    with pytest.raises(ValueError, match="distinct"):
        ml.fit(np.ones(6), np.arange(6))


def test_rank_deficiency_is_reported():
    x = np.linspace(0, 1, 5)
    model = ml.fit(x, x**2, degree=3, num_control_points=8, max_iterations=25)
    assert not model.diagnostics.full_column_rank
    assert np.isinf(model.diagnostics.condition_number)


def test_history_storage_is_optional():
    x = np.linspace(0, 1, 20)
    compact = ml.fit(x, x, num_control_points=6)
    history = ml.fit(x, x, num_control_points=6, store_history=True)
    assert compact.coefficient_sequence is None
    assert history.coefficient_sequence is not None
    assert len(history.coefficient_sequence) == history.diagnostics.iterations + 1


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_fixed_endpoints_remain_exact(direction):
    rng = np.random.default_rng(41)
    x = np.linspace(-2, 3, 40)
    trend = np.linspace(-1, 2, x.size)
    y = trend + 0.8 * rng.standard_normal(x.size)
    y[0], y[-1] = -1.0, 2.0
    if direction == "decreasing":
        y = -y
    model = ml.fit(
        x,
        y,
        direction=direction,
        num_control_points=9,
        fix_endpoints=True,
    )
    assert model.coefficients[0] == y[0]
    assert model.coefficients[-1] == y[-1]
    signed_difference = np.diff(model.coefficients)
    if direction == "decreasing":
        signed_difference = -signed_difference
    assert np.min(signed_difference) >= 0.0


def test_incompatible_fixed_endpoints_are_rejected():
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    y[0], y[-1] = 2.0, -1.0
    with pytest.raises(ValueError, match="incompatible"):
        ml.fit(x, y, num_control_points=6, fix_endpoints=True)


@pytest.mark.parametrize(
    ("direction", "y", "expected_endpoints"),
    [
        ("increasing", np.array([0.0, 2.0, 2.5, 3.0, 5.0]), (1.0, 4.0)),
        ("decreasing", np.array([5.0, 3.0, 2.5, 2.0, 0.0]), (4.0, 1.0)),
    ],
)
def test_fixed_endpoints_average_boundary_replicates_and_are_permutation_invariant(
    direction, y, expected_endpoints
):
    x = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    first = ml.fit(
        x,
        y,
        direction=direction,
        degree=1,
        num_control_points=3,
        fix_endpoints=True,
    )
    permutation = np.array([1, 0, 2, 4, 3])
    second = ml.fit(
        x[permutation],
        y[permutation],
        direction=direction,
        degree=1,
        num_control_points=3,
        fix_endpoints=True,
    )
    assert first.coefficients[0] == pytest.approx(expected_endpoints[0])
    assert first.coefficients[-1] == pytest.approx(expected_endpoints[1])
    np.testing.assert_allclose(second.coefficients, first.coefficients, atol=1e-13)


def test_basis_matrix_properties():
    parameters = np.linspace(0, 1, 301)
    basis = ml.basis_matrix(parameters, 11, degree=3)
    assert issparse(basis)
    assert basis.min() >= -1e-14
    np.testing.assert_allclose(np.asarray(basis.sum(axis=1)).ravel(), 1, atol=1e-13)
    assert basis[0, 0] == 1
    assert basis[-1, -1] == 1


def test_version_metadata_is_consistent():
    root = Path(__file__).parents[1]
    model = ml.fit(np.linspace(0, 1, 10), np.linspace(0, 1, 10), num_control_points=5)
    assert model.version == ml.__version__ == ml.version() == "1.0.2"
    assert 'version: "1.0.2"' in (root / "CITATION.cff").read_text()
    assert '"version": "1.0.2"' in (root / "codemeta.json").read_text()


def test_model_save_load_roundtrip(tmp_path):
    x = np.linspace(5, 25, 60)
    y = np.log(x)
    model = ml.fit(x, y, num_control_points=8, store_history=True)
    query = np.linspace(5, 25, 101)
    expected = model.predict(query)

    archive = tmp_path / "fitted-model.npz"
    assert ml.save(model, archive) == archive
    restored = ml.load(archive)
    np.testing.assert_array_equal(restored.predict(query), expected)
    np.testing.assert_array_equal(restored.coefficients, model.coefficients)
    np.testing.assert_array_equal(
        restored.diagnostics.error_history,
        model.diagnostics.error_history,
    )
    assert restored.coefficient_sequence is not None
    assert model.coefficient_sequence is not None
    for actual, original in zip(restored.coefficient_sequence, model.coefficient_sequence):
        np.testing.assert_array_equal(actual, original)

    second_archive = tmp_path / "model-method.npz"
    assert model.save(second_archive) == second_archive
    np.testing.assert_array_equal(ml.load(second_archive).predict(query), expected)


def test_plot_public_api_smoke():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 30)
    model = ml.fit(x, np.sqrt(x), num_control_points=7)
    figure, axes = ml.plot(model)
    assert len(axes) == 2
    assert axes[0].get_xlabel() == "x"
    assert axes[1].get_xlabel() == "Iteration"
    plt.close(figure)


def test_functional_aliases_match_concise_api():
    x = np.linspace(0, 1, 20)
    model = ml.monotone_lspia_fit(x, x**2, num_control_points=6)
    np.testing.assert_allclose(ml.monotone_lspia_predict(model, x), ml.predict(model, x))
    assert ml.monotone_lspia_version() == ml.version()
