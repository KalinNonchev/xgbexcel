"""Public surface: construction options, renaming, saving, and reprs."""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from tests.conftest import assert_formula_matches_model
from xgbexcel import XGBtoExcel, __version__
from xgbexcel.convert import UnsupportedModelError

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def model(integer_data):
    X, y = integer_data
    return xgb.XGBRegressor(n_estimators=3, max_depth=2).fit(X, y)


def test_version_is_exposed():
    assert __version__.count(".") == 2


def test_default_feature_names_are_one_indexed(model):
    converter = XGBtoExcel(model)
    assert converter.feature_names == ["x1", "x2", "x3", "x4"]


def test_feature_names_can_be_given_up_front(integer_data, model):
    X, _ = integer_data
    names = ["age", "income", "score", "tenure"]
    converter = XGBtoExcel(model, feature_names=names)

    assert converter.feature_names == names
    assert "x1" not in converter.expression
    assert_formula_matches_model(converter, model, X)


def test_wrong_number_of_feature_names_is_rejected(model):
    with pytest.raises(ValueError, match="4 features but 2 names"):
        XGBtoExcel(model, feature_names=["a", "b"])


def test_rename_returns_self_for_chaining(integer_data, model):
    X, _ = integer_data
    converter = XGBtoExcel(model).rename_features({"x1": "age"}).rename_features({"x2": "income"})

    assert converter.feature_names == ["age", "income", "x3", "x4"]
    assert_formula_matches_model(converter, model, X)


def test_rename_with_empty_mapping_is_a_no_op(model):
    converter = XGBtoExcel(model)
    before = converter.expression
    assert converter.rename_features({}).expression == before


def test_save_expr_writes_one_line_per_output(tmp_path, integer_data):
    X, y = integer_data
    labels = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    classifier = xgb.XGBClassifier(n_estimators=2, max_depth=2).fit(X, labels)
    converter = XGBtoExcel(classifier)

    destination = tmp_path / "formula.txt"
    converter.save_expr(str(destination))

    lines = destination.read_text(encoding="utf-8").split("\n")
    assert lines == converter.expressions
    assert len(lines) == 3


def test_str_returns_the_expression(model):
    converter = XGBtoExcel(model)
    assert str(converter) == converter.expression


def test_repr_summarises_the_model(model):
    text = repr(XGBtoExcel(model))
    assert "reg:squarederror" in text
    assert "n_features=4" in text


def test_single_output_expression_is_not_joined(model):
    converter = XGBtoExcel(model)
    assert converter.expression == converter.expressions[0]
    assert " , " not in converter.expression


def test_multiclass_expression_joins_outputs(integer_data):
    X, y = integer_data
    labels = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    classifier = xgb.XGBClassifier(n_estimators=2, max_depth=1).fit(X, labels)
    converter = XGBtoExcel(classifier)

    assert converter.expression == " , ".join(converter.expressions)


def test_size_warning_can_be_silenced(continuous_data):
    X, y = continuous_data
    big = xgb.XGBRegressor(n_estimators=60, max_depth=5).fit(X, y)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        XGBtoExcel(big, warn_on_size=False)


def test_categorical_splits_are_rejected(integer_data):
    pd = pytest.importorskip("pandas")

    X, y = integer_data
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["a"] = frame["a"].astype(int).astype("category")
    model = xgb.XGBRegressor(n_estimators=4, max_depth=3, enable_categorical=True).fit(frame, y)

    with pytest.raises(UnsupportedModelError, match="categorical"):
        XGBtoExcel(model)


def test_feature_names_come_from_a_dataframe(integer_data):
    pd = pytest.importorskip("pandas")

    X, y = integer_data
    names = ["age", "income", "score", "tenure"]
    frame = pd.DataFrame(X, columns=names)
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2).fit(frame, y)

    converter = XGBtoExcel(model)
    assert converter.feature_names == names
    assert_formula_matches_model(converter, model, X)


def test_explicit_feature_names_override_the_dataframe(integer_data):
    pd = pytest.importorskip("pandas")

    X, y = integer_data
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2).fit(frame, y)

    converter = XGBtoExcel(model, feature_names=["w", "x", "y", "z"])
    assert converter.feature_names == ["w", "x", "y", "z"]


def test_gblinear_is_rejected(integer_data):
    X, y = integer_data
    model = xgb.XGBRegressor(booster="gblinear", n_estimators=3).fit(X, y)

    with pytest.raises(UnsupportedModelError, match="gbtree"):
        XGBtoExcel(model)


def test_dart_without_drops_matches(integer_data):
    """With rate_drop=0, dart serialises as a plain gbtree."""
    X, y = integer_data
    model = xgb.XGBRegressor(booster="dart", n_estimators=4, max_depth=2).fit(X, y)

    converter = XGBtoExcel(model)
    assert converter._tree_weights == [1.0] * 4
    assert_formula_matches_model(converter, model, X)


def test_dart_with_drops_scales_each_tree(integer_data):
    """dart drops trees and reweights the survivors, so a plain sum would be wrong."""
    X, y = integer_data
    model = xgb.XGBRegressor(
        booster="dart", n_estimators=8, max_depth=2, rate_drop=0.3, skip_drop=0.0
    ).fit(X, y)

    converter = XGBtoExcel(model)
    assert any(weight != 1.0 for weight in converter._tree_weights), "fixture needs drops"
    assert_formula_matches_model(converter, model, X)


def test_conversion_is_deterministic(model):
    assert XGBtoExcel(model).expressions == XGBtoExcel(model).expressions


def test_expression_trees_has_one_entry_per_tree(integer_data):
    X, y = integer_data
    trained = xgb.XGBRegressor(n_estimators=7, max_depth=2).fit(X, y)
    converter = XGBtoExcel(trained)

    assert len(converter.expression_trees) == 7
    assert converter.n_trees_used == 7
    assert all(part.startswith("IF(") for part in converter.expression_trees)
