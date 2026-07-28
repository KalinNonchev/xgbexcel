"""The generated formula must reproduce the model's own predictions."""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from tests.conftest import assert_formula_matches_model, predict_from_formula
from xgbexcel import XGBtoExcel

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.mark.parametrize(
    ("objective", "n_estimators", "max_depth"),
    [
        ("reg:squarederror", 5, 3),
        ("reg:squarederror", 1, 1),
        ("reg:absoluteerror", 4, 2),
        ("reg:pseudohubererror", 4, 2),
        ("count:poisson", 4, 3),
        ("reg:gamma", 4, 2),
        ("reg:tweedie", 4, 2),
    ],
)
def test_regressor_objectives(integer_data, objective, n_estimators, max_depth):
    X, y = integer_data
    target = np.abs(y) + 1.0 if objective in {"count:poisson", "reg:gamma", "reg:tweedie"} else y
    model = xgb.XGBRegressor(
        objective=objective, n_estimators=n_estimators, max_depth=max_depth
    ).fit(X, target)

    assert_formula_matches_model(XGBtoExcel(model), model, X)


@pytest.mark.parametrize("max_depth", [1, 3, 6])
def test_binary_classifier(integer_data, max_depth):
    X, y = integer_data
    labels = (y > np.median(y)).astype(int)
    model = xgb.XGBClassifier(n_estimators=6, max_depth=max_depth).fit(X, labels)

    converter = XGBtoExcel(model)
    assert converter.link == "logistic"
    assert converter.n_outputs == 1
    assert_formula_matches_model(converter, model, X)


@pytest.mark.parametrize("n_classes", [3, 4])
def test_multiclass_classifier(integer_data, n_classes):
    X, y = integer_data
    labels = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_classes + 1)[1:-1]))
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2).fit(X, labels)

    converter = XGBtoExcel(model)
    assert converter.link == "softmax"
    assert converter.n_outputs == n_classes
    assert len(converter.expressions) == n_classes
    assert_formula_matches_model(converter, model, X)

    probabilities = predict_from_formula(converter, X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)


def test_logitraw_stays_on_the_margin_scale(integer_data):
    X, y = integer_data
    labels = (y > np.median(y)).astype(int)
    model = xgb.XGBClassifier(objective="binary:logitraw", n_estimators=4, max_depth=2).fit(
        X, labels
    )

    converter = XGBtoExcel(model)
    assert converter.link == "identity"
    assert_formula_matches_model(converter, model, X)


def test_multi_output_regression_one_tree_per_output(integer_data):
    X, y = integer_data
    targets = np.column_stack([y, y * 2.0 + 1.0])
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2).fit(X, targets)

    converter = XGBtoExcel(model)
    assert converter.n_outputs == 2
    assert_formula_matches_model(converter, model, X)


def test_vector_leaves(integer_data):
    """multi_output_tree emits a list per leaf. This is what the old README flagged."""
    X, y = integer_data
    targets = np.column_stack([y, y * 2.0 + 1.0])
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2, multi_strategy="multi_output_tree").fit(
        X, targets
    )

    converter = XGBtoExcel(model)
    assert converter.n_outputs == 2
    assert converter._vector_leaves is True
    assert_formula_matches_model(converter, model, X)


def test_raw_booster_is_accepted(integer_data):
    X, y = integer_data
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2).fit(X, y)

    from_estimator = XGBtoExcel(model)
    from_booster = XGBtoExcel(model.get_booster())
    assert from_booster.expressions == from_estimator.expressions


def test_continuous_features(continuous_data):
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=8, max_depth=4).fit(X, y)
    assert_formula_matches_model(XGBtoExcel(model), model, X)


def test_deep_tree_does_not_hit_the_recursion_limit(continuous_data):
    """The old converter called sys.setrecursionlimit(10000) to survive this."""
    X, y = continuous_data
    model = xgb.XGBRegressor(
        n_estimators=2, max_depth=0, grow_policy="lossguide", max_leaves=64
    ).fit(X, y)

    assert_formula_matches_model(XGBtoExcel(model), model, X)
