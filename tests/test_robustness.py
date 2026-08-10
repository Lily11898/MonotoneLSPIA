from pathlib import Path

import numpy as np
import pytest

from monotone_lspia import fit, predict


def test_unsorted_input_matches_sorted_input():
    rng = np.random.default_rng(21)
    x = np.linspace(-5, 7, 60)
    y = np.arctan(x) + 0.02 * rng.standard_normal(x.size)
    permutation = rng.permutation(x.size)
    sorted_model = fit(x, y, num_control_points=9)
    unsorted_model = fit(x[permutation], y[permutation], num_control_points=9)
    np.testing.assert_allclose(unsorted_model.training_x, x)
    np.testing.assert_allclose(unsorted_model.coefficients, sorted_model.coefficients, atol=1e-13)


def test_repeated_run_is_deterministic():
    x = np.linspace(0, 2, 45)
    y = 1 - np.exp(-x)
    first = fit(x, y, num_control_points=8, weight="theoretical")
    second = fit(x, y, num_control_points=8, weight="theoretical")
    np.testing.assert_array_equal(second.coefficients, first.coefficients)
    np.testing.assert_array_equal(second.diagnostics.error_history, first.diagnostics.error_history)


def test_constant_response_is_preserved():
    x = np.linspace(4, 9, 35)
    model = fit(x, np.full(x.shape, 2.75), num_control_points=7)
    prediction = predict(model, np.linspace(4, 9, 300))
    np.testing.assert_allclose(prediction, 2.75, atol=1e-12)
    assert model.rmse == pytest.approx(0, abs=1e-12)


def test_prediction_preserves_query_shape():
    x = np.linspace(0, 1, 20)
    model = fit(x, x**2, num_control_points=6)
    assert predict(model, np.array([[0, 0.25], [0.75, 1]])).shape == (2, 2)
    assert np.isscalar(predict(model, 0.5))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"degree": 3, "num_control_points": 3}, "greater than degree"),
        (
            {"degree": 2, "num_control_points": 4, "knots": [0, 0, 0, 0.8, 0.2, 1, 1]},
            "nondecreasing",
        ),
        (
            {"degree": 2, "num_control_points": 4, "knots": [0, 0, 0, 0.5, 1, 1]},
            "contain",
        ),
    ],
)
def test_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit(np.arange(6), np.arange(6), **kwargs)


def test_size_mismatch_rejected():
    with pytest.raises(ValueError, match="same number"):
        fit([0, 1], [0])


def test_invalid_model_is_rejected():
    with pytest.raises(TypeError, match="returned"):
        predict({}, 0.5)


def test_bundled_puromycin_data():
    data_file = Path(__file__).parents[1] / "data" / "Puromycin.csv"
    data = np.genfromtxt(data_file, delimiter=",", names=True, dtype=None, encoding="utf-8")
    for state in ("treated", "untreated"):
        selected = data["state"] == state
        x, y = data["conc"][selected], data["rate"][selected]
        model = fit(
            x,
            y,
            num_control_points=6,
            max_iterations=2000,
            tolerance=1e-10,
        )
        prediction = predict(model, np.linspace(x.min(), x.max(), 500))
        assert np.min(np.diff(prediction)) >= -1e-10
        assert model.diagnostics.sampled_curve_violation <= 1e-18
        assert np.all(np.isfinite(model.coefficients))
