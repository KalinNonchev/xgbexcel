"""Branch decisions must survive the float32 to float64 gap.

XGBoost evaluates ``float32(x) < float32(threshold)``. Excel has no float32 and
evaluates in float64. Those two disagree exactly when ``x`` rounds to the threshold,
which is the common case rather than a corner case: XGBoost's histogram method picks
thresholds that *are* feature values from the training data, so every row sitting on a
split is at risk. The converter compensates by emitting the midpoint between the
float32 threshold and the float32 below it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xgboost as xgb

from tests.conftest import assert_formula_matches_model, predict_from_formula
from xgbexcel import XGBtoExcel
from xgbexcel.convert import _excel_threshold

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_threshold_midpoint_reproduces_the_float32_comparison():
    """The compensation, checked directly against float32 semantics."""
    generator = np.random.default_rng(0)
    thresholds = generator.normal(0, 5, 400).astype(np.float32)

    for threshold in thresholds:
        shifted = _excel_threshold(float(threshold))
        for value in (
            float(threshold),
            float(np.nextafter(threshold, np.float32("inf"))),
            float(np.nextafter(threshold, np.float32("-inf"))),
            float(threshold) * (1 + 1e-9),
            float(threshold) * (1 - 1e-9),
        ):
            xgboost_goes_left = bool(np.float32(value) < threshold)
            excel_goes_left = value < shifted
            assert excel_goes_left == xgboost_goes_left, (
                f"threshold={threshold!r} value={value!r}: "
                f"xgboost={xgboost_goes_left} excel={excel_goes_left}"
            )


def test_threshold_shift_is_tiny():
    """The shift must be small enough to be invisible, and never zero."""
    for value in (-1234.5, -0.9031178, 0.0, 1e-8, 0.5, 3.25, 98765.4):
        shifted = _excel_threshold(value)
        exact = float(np.float32(value))
        assert shifted < exact
        assert abs(shifted - exact) <= abs(exact) * 1e-6 + 1e-37


def test_threshold_shift_fits_excel_precision():
    """Excel keeps 15 significant decimal digits. The shift must survive that."""
    for value in (-0.9031178, 0.03952289, 1234.5678):
        shifted = _excel_threshold(value)
        rounded = float(f"{shifted:.15g}")
        exact = float(np.float32(value))
        assert rounded < exact, "the shift was lost when rounded to Excel's precision"


def test_rows_sitting_exactly_on_a_split_are_routed_correctly():
    """The failure this fix exists for, reproduced end to end."""
    generator = np.random.default_rng(1)
    X = generator.normal(0, 1, (250, 3))
    y = X[:, 0] * 2.0 + generator.normal(0, 0.3, 250)

    model = xgb.XGBRegressor(n_estimators=10, max_depth=4, tree_method="hist").fit(X, y)
    converter = XGBtoExcel(model, warn_on_size=False)

    # The thresholds really are training values, which is what makes this bite.
    raw = json.loads(bytes(model.get_booster().save_raw(raw_format="json")))
    thresholds = {
        float(np.float32(condition))
        for tree in raw["learner"]["gradient_booster"]["model"]["trees"]
        for condition, left in zip(tree["split_conditions"], tree["left_children"])
        if left != -1
    }
    on_a_split = sum(1 for t in thresholds if (X.astype(np.float32) == np.float32(t)).any())
    assert on_a_split > 0, "fixture no longer exercises the boundary"

    assert_formula_matches_model(converter, model, X)


def test_every_row_matches_not_merely_the_average():
    """Guards against a fix that reduces mean error while leaving branches wrong."""
    generator = np.random.default_rng(7)
    X = generator.normal(0, 1, (300, 4))
    y = np.sin(X[:, 0]) * 3 + X[:, 2] + generator.normal(0, 0.2, 300)

    model = xgb.XGBRegressor(n_estimators=12, max_depth=5, tree_method="hist").fit(X, y)
    converter = XGBtoExcel(model, warn_on_size=False)

    actual = predict_from_formula(converter, X)
    expected = model.predict(X, output_margin=True).reshape(-1, 1)
    wrong = int((~np.isclose(actual, expected, rtol=1e-5, atol=1e-6)).sum())
    assert wrong == 0, f"{wrong} of {len(X)} rows take a different branch"


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_feature_scales(scale):
    """Thresholds span many magnitudes; the shift has to work at all of them."""
    generator = np.random.default_rng(4)
    X = generator.normal(0, 1, (150, 3)) * scale
    y = X[:, 0] / scale + generator.normal(0, 0.2, 150)

    model = xgb.XGBRegressor(n_estimators=6, max_depth=3).fit(X, y)
    assert_formula_matches_model(XGBtoExcel(model, warn_on_size=False), model, X)


def test_leaf_values_keep_full_precision():
    """Leaf values are float32; writing fewer digits would drift the sum."""
    generator = np.random.default_rng(5)
    X = generator.normal(0, 1, (120, 2))
    y = generator.normal(0, 1, 120) * 1e-7  # tiny leaves, where truncation would show

    model = xgb.XGBRegressor(n_estimators=5, max_depth=3).fit(X, y)
    assert_formula_matches_model(XGBtoExcel(model), model, X, rtol=1e-6, atol=1e-12)
