"""One test per defect found in the pre-1.0 converter, so none of them come back.

Each test names the wrong behaviour it guards against.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import xgboost as xgb

from tests.conftest import (
    XGBOOST_VERSION,
    assert_formula_matches_model,
    predict_from_formula,
    requires_xgboost_2,
)
from xgbexcel import XGBtoExcel
from xgbexcel.convert import UnsupportedModelError

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_splits_use_strict_less_than(integer_data):
    """Was ``x <= threshold``. XGBoost branches left on ``value < threshold``.

    With integer features, many rows sit exactly on a threshold, and the wrong
    operator sends them down the wrong branch.
    """
    X, y = integer_data
    model = xgb.XGBRegressor(n_estimators=6, max_depth=3).fit(X, y)
    converter = XGBtoExcel(model)

    assert "<=" not in converter.expression
    assert_formula_matches_model(converter, model, X)

    # Prove the data actually exercises the boundary, or the test proves nothing.
    thresholds = {
        node["split_condition"]
        for tree in model.get_booster().get_dump(dump_format="json")
        for node in _walk(json.loads(tree))
        if "split_condition" in node
    }
    assert any(np.isclose(X, t).any() for t in thresholds), "no row lands on a threshold"


def test_base_score_read_from_booster_config(integer_data):
    """Was read from the estimator attribute, which is None on xgboost >= 2.0."""
    X, y = integer_data
    y_shifted = y + 100.0  # a base_score far from the old 0.5 fallback
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2).fit(X, y_shifted)

    assert model.base_score is None, "fixture assumption: attribute is unset"
    converter = XGBtoExcel(model)

    config = json.loads(model.get_booster().save_config())
    stored = config["learner"]["learner_model_param"]["base_score"]
    expected = json.loads(stored)[0] if stored.strip().startswith("[") else float(stored)

    assert converter.base_scores[0] == pytest.approx(expected)
    if XGBOOST_VERSION >= (2, 0):
        # 2.0+ estimates the intercept from the data; 1.x always used 0.5.
        assert converter.base_scores[0] != 0.5
    assert_formula_matches_model(converter, model, X)


def test_explicit_zero_base_score_is_not_replaced(integer_data):
    """Was ``if model.base_score`` so an explicit 0.0 silently became 0.5."""
    X, y = integer_data
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, base_score=0.0).fit(X, y)

    converter = XGBtoExcel(model)
    assert converter.base_scores == [0.0]
    assert_formula_matches_model(converter, model, X)


def test_multiclass_uses_per_class_base_scores(integer_data):
    """Was a single 0.5 added to every class. xgboost 3.x stores a vector."""
    X, y = integer_data
    labels = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2).fit(X, labels)

    converter = XGBtoExcel(model)
    assert len(converter.base_scores) == 3
    if XGBOOST_VERSION >= (2, 0):
        # 3.x stores one intercept per class; 1.x stored a single 0.5 for all of them.
        assert converter.base_scores != [0.5, 0.5, 0.5]
    assert_formula_matches_model(converter, model, X)


def test_binary_classification_uses_a_sigmoid_not_softmax(integer_data):
    """Was treated as two interleaved tree groups fed through a softmax.

    binary:logistic trains one tree per round and applies a logit link, so the old
    output was not a probability at all.
    """
    X, y = integer_data
    labels = (y > np.median(y)).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(X, labels)
    converter = XGBtoExcel(model)

    assert converter.n_outputs == 1
    assert len(converter.expressions) == 1
    assert len(converter.expression_trees) == 5  # one per round, not two groups

    # The intercept is the logit of the stored probability, not the probability.
    config = json.loads(model.get_booster().save_config())
    stored = float(json.loads(config["learner"]["learner_model_param"]["base_score"])[0])
    assert converter.base_scores[0] == pytest.approx(math.log(stored / (1 - stored)))

    probabilities = predict_from_formula(converter, X)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert_formula_matches_model(converter, model, X)


@requires_xgboost_2
def test_vector_leaves_are_supported(integer_data):
    """Was unsupported, and is what the old README pointed at."""
    X, y = integer_data
    targets = np.column_stack([y, y * -3.0])
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, multi_strategy="multi_output_tree").fit(
        X, targets
    )

    dump = json.loads(model.get_booster().get_dump(dump_format="json")[0])
    assert isinstance(next(iter(_leaves(dump)))["leaf"], list), "fixture needs vector leaves"

    assert_formula_matches_model(XGBtoExcel(model), model, X)


def test_rename_does_not_match_a_longer_feature_name(continuous_data):
    """``str.replace('x1', ...)`` also rewrote the prefix of ``x10``."""
    generator = np.random.default_rng(3)
    X = generator.normal(0, 1, (120, 12))
    y = X[:, 0] + X[:, 9] + X[:, 10]
    model = xgb.XGBRegressor(n_estimators=6, max_depth=3).fit(X, y)

    converter = XGBtoExcel(model)
    assert "x10" in converter.expression, "fixture needs x10 to be used"

    converter.rename_features({"x1": "Age"})
    assert "x10" in converter.expression
    assert "Age0" not in converter.expression
    assert "Age" in converter.expression


def test_rename_does_not_cascade(continuous_data):
    """Sequential replacement let a new name be renamed again by a later entry."""
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=4, max_depth=3).fit(X, y)
    converter = XGBtoExcel(model)

    converter.rename_features({"x1": "x2", "x2": "x3"})
    assert converter.feature_names[:2] == ["x2", "x3"]
    # The x1 -> x2 result must not then be swept up by x2 -> x3.
    assert converter.expression.count("x3") <= converter.expression.count("x2") + len(
        converter.expression
    )


def test_rename_rejects_unknown_features(continuous_data):
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2).fit(X, y)
    with pytest.raises(KeyError, match="unknown feature"):
        XGBtoExcel(model).rename_features({"not_a_feature": "z"})


def test_no_global_recursion_limit_change(continuous_data):
    """The old constructor called sys.setrecursionlimit(10000) as a side effect."""
    import sys

    before = sys.getrecursionlimit()
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=4, max_depth=6).fit(X, y)
    XGBtoExcel(model)
    assert sys.getrecursionlimit() == before


def test_separator_reaches_nested_nodes(continuous_data):
    """``sep`` was accepted but never passed to child nodes."""
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=3, max_depth=3).fit(X, y)

    converter = XGBtoExcel(model, sep=";")
    assert "," not in converter.expression
    assert converter.expression.count(";") >= 2 * 3  # at least two per depth-3 tree


def test_expression_does_not_nest_one_level_per_tree(continuous_data):
    """Was built with ``expr = f'({expr}+{tree})'`` in a loop.

    That is quadratic to assemble and adds a bracket level for every tree.
    """
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=40, max_depth=1).fit(X, y)
    expression = XGBtoExcel(model).expression

    depth = maximum = 0
    for character in expression:
        if character == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif character == ")":
            depth -= 1
    assert maximum < 40, f"bracket depth {maximum} grows with the tree count"


def test_oversized_formula_warns(continuous_data):
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=60, max_depth=5).fit(X, y)

    with pytest.warns(UserWarning, match="over Excel's"):
        XGBtoExcel(model)


def test_unsupported_objective_fails_loudly(integer_data):
    X, y = integer_data
    model = xgb.XGBRanker(n_estimators=2, max_depth=2)
    model.fit(X, (y > y.mean()).astype(int), qid=np.zeros(len(X), dtype=int))

    with pytest.raises(UnsupportedModelError, match="not supported"):
        XGBtoExcel(model)


def test_non_model_input_is_rejected():
    with pytest.raises(TypeError, match="Booster"):
        XGBtoExcel("not a model")


def test_early_stopping_uses_only_the_trees_predict_uses(continuous_data):
    """predict() stops at best_iteration. The model dump still holds every tree.

    Converting all of them adds boosting rounds the model itself discarded.
    """
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, early_stopping_rounds=5)
    model.fit(X[:100], y[:100], eval_set=[(X[100:], y[100:])], verbose=False)

    total_trees = len(model.get_booster().get_dump())
    assert model.best_iteration + 1 < total_trees, "fixture must actually stop early"

    converter = XGBtoExcel(model, warn_on_size=False)
    assert converter.n_trees_used == model.best_iteration + 1
    assert converter.n_trees_used < total_trees
    assert_formula_matches_model(converter, model, X)


def test_num_parallel_tree_sums_every_tree(continuous_data):
    """Forest mode puts several trees in one boosting round. All of them count."""
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=3, max_depth=3, num_parallel_tree=4).fit(X, y)

    converter = XGBtoExcel(model, warn_on_size=False)
    assert converter.n_trees_used == 12
    assert_formula_matches_model(converter, model, X)


def test_tree_groups_come_from_the_model_not_from_position(integer_data):
    """Output grouping reads tree_info rather than assuming index % n_outputs."""
    X, y = integer_data
    labels = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    model = xgb.XGBClassifier(n_estimators=4, max_depth=2).fit(X, labels)

    raw = json.loads(bytes(model.get_booster().save_raw(raw_format="json")))
    tree_info = raw["learner"]["gradient_booster"]["model"]["tree_info"]
    assert sorted(tree_info) == sorted([0, 1, 2] * 4)

    assert_formula_matches_model(XGBtoExcel(model), model, X)


def test_text_dump_thresholds_would_give_wrong_branches(continuous_data):
    """Why the converter reads the model JSON instead of the text dump.

    ``get_dump()`` prints split conditions at reduced precision. Comparing against
    those values in float64 sends rows down the wrong branch. This test shows the old
    approach failing and the current one succeeding on the same model, so the reason
    for the extra machinery stays visible.
    """
    X, y = continuous_data
    model = xgb.XGBRegressor(n_estimators=10, max_depth=4).fit(X, y)

    trees = [json.loads(tree) for tree in model.get_booster().get_dump(dump_format="json")]

    def walk_with_dump_thresholds(node, row):
        while "leaf" not in node:
            children = {child["nodeid"]: child for child in node["children"]}
            feature = int(node["split"].removeprefix("f"))
            go_left = row[feature] < node["split_condition"]
            node = children[node["yes"] if go_left else node["no"]]
        return node["leaf"]

    converter = XGBtoExcel(model, warn_on_size=False)
    naive = (
        np.array([sum(walk_with_dump_thresholds(t, row) for t in trees) for row in X])
        + converter.base_scores[0]
    )
    expected = model.predict(X, output_margin=True)

    naive_wrong = int((~np.isclose(naive, expected, rtol=1e-5, atol=1e-6)).sum())
    assert naive_wrong > 0, "fixture no longer reproduces the precision problem"

    assert_formula_matches_model(converter, model, X)


# --------------------------------------------------------------------- helpers


def _walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.get("children", []))


def _leaves(node):
    return (n for n in _walk(node) if "leaf" in n)
