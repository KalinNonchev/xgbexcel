"""Shared fixtures and the formula evaluator used to verify generated output.

The generated Excel formula is close enough to Python that it can be evaluated
directly: ``IF(a,b,c)`` is a call, ``EXP`` is ``math.exp``, and feature names are
plain identifiers. That lets every test compare the formula against what the model
itself predicts, instead of asserting on hard-coded strings that go stale whenever
XGBoost changes a leaf value in the last decimal place.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xgboost as xgb

XGBOOST_VERSION = tuple(int(part) for part in xgb.__version__.split(".")[:2])

# XGBoost 2.0 added multi_strategy, several objectives, and the estimated intercept
# that replaced the fixed base_score of 0.5.
requires_xgboost_2 = pytest.mark.skipif(
    XGBOOST_VERSION < (2, 0), reason="requires XGBoost 2.0 or newer"
)

NUMPY_VERSION = tuple(int(part) for part in np.__version__.split(".")[:2])

# XGBoost below 2.1 reaches for np.NaN when handling categorical features, and NumPy
# 2.0 removed it. The combination cannot build the fixture; it says nothing about us.
categorical_fixture_works = pytest.mark.skipif(
    XGBOOST_VERSION < (2, 1) and NUMPY_VERSION >= (2, 0),
    reason="XGBoost <2.1 uses np.NaN, removed in NumPy 2.0",
)


def _excel_if(condition, if_true, if_false):
    return if_true if condition else if_false


def _excel_exp(value):
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def evaluate(expression: str, row, feature_names) -> float:
    """Evaluate one generated formula for one row of features."""
    namespace = {name: float(value) for name, value in zip(feature_names, row)}
    namespace["IF"] = _excel_if
    namespace["EXP"] = _excel_exp
    return eval(expression, {"__builtins__": {}}, namespace)


def predict_from_formula(converter, X) -> np.ndarray:
    """Evaluate every output formula over every row, shaped (n_rows, n_outputs)."""
    return np.array(
        [
            [evaluate(expr, row, converter.feature_names) for expr in converter.expressions]
            for row in np.asarray(X)
        ]
    )


def reference_prediction(model, X, link: str) -> np.ndarray:
    """What the formula is supposed to reproduce, taken from XGBoost itself."""
    n = len(X)
    if link == "logistic":
        return model.predict_proba(X)[:, 1:2]
    if link == "softmax":
        return model.predict_proba(X)
    if link == "identity":
        return np.asarray(model.predict(X, output_margin=True)).reshape(n, -1)
    return np.asarray(model.predict(X)).reshape(n, -1)


def assert_formula_matches_model(converter, model, X, rtol=1e-5, atol=1e-6) -> None:
    """The core assertion: the spreadsheet returns what the model returns."""
    expected = reference_prediction(model, X, converter.link)
    actual = predict_from_formula(converter, X)
    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def integer_data():
    """Integer features, so many rows land exactly on a split threshold.

    That is the case that distinguishes a strict ``<`` split from ``<=``, and the
    original converter got it wrong.
    """
    generator = np.random.default_rng(0)
    X = generator.integers(0, 12, (200, 4)).astype(float)
    y = X[:, 0] * 0.5 + generator.normal(0, 0.4, 200)
    return X, y


@pytest.fixture(scope="session")
def continuous_data():
    generator = np.random.default_rng(1)
    X = generator.normal(0, 1, (150, 3))
    y = X[:, 0] * 2.0 + generator.normal(0, 0.3, 150)
    return X, y
