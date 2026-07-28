"""Execute the README's Python examples.

The examples in the README build on each other, so a reader runs them in order in one
session. This test does the same. It catches missing imports, undefined names, and
snippets that look like they compose but raise when you actually try, which is how the
feature-naming example was wrong before.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

CODE_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def python_blocks() -> list[str]:
    return CODE_BLOCK.findall(README.read_text(encoding="utf-8"))


def test_readme_has_python_examples():
    assert len(python_blocks()) >= 4, "README lost its examples"


def test_readme_examples_run_in_order(tmp_path, monkeypatch):
    """Run every block in one namespace, the way a reader would."""
    monkeypatch.chdir(tmp_path)  # save_expr writes a file
    namespace: dict[str, object] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index, block in enumerate(python_blocks()):
            try:
                exec(compile(block, f"README.md[block {index}]", "exec"), namespace)
            except Exception as error:  # pragma: no cover - only on a broken README
                pytest.fail(
                    f"README block {index} failed with {type(error).__name__}: {error}\n\n{block}"
                )


def test_naming_the_two_documented_ways_agree(tmp_path):
    """The README claims both ways of naming features give the same formula."""
    import numpy as np
    import xgboost as xgb

    from xgbexcel import XGBtoExcel

    generator = np.random.default_rng(0)
    X = generator.integers(0, 100, (80, 2)).astype(float)
    y = X[:, 0] * 0.3 + generator.normal(0, 0.5, 80)
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2).fit(X, y)

    up_front = XGBtoExcel(model, feature_names=["age", "income"])
    renamed = XGBtoExcel(model).rename_features({"x1": "age", "x2": "income"})

    assert up_front.expression == renamed.expression
    assert up_front.feature_names == renamed.feature_names == ["age", "income"]


def test_renaming_after_naming_raises_as_documented(tmp_path):
    """The README warns that doing both raises rather than silently doing nothing."""
    import numpy as np
    import xgboost as xgb

    from xgbexcel import XGBtoExcel

    generator = np.random.default_rng(0)
    X = generator.integers(0, 100, (80, 2)).astype(float)
    y = X[:, 0] * 0.3
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1).fit(X, y)

    converter = XGBtoExcel(model, feature_names=["age", "income"])
    with pytest.raises(KeyError, match="unknown feature"):
        converter.rename_features({"x1": "age", "x2": "income"})


def test_documented_objectives_match_the_code():
    """The objective table in the README must not drift from what is implemented."""
    from xgbexcel.convert import _OBJECTIVES

    text = README.read_text(encoding="utf-8")
    for objective in _OBJECTIVES:
        if objective == "reg:linear":
            continue  # a pre-1.0 alias, deliberately not advertised
        assert f"`{objective}`" in text, f"{objective} is supported but undocumented"


def test_documented_attributes_exist():
    """Every attribute named in the API table must actually be there."""
    import numpy as np
    import xgboost as xgb

    from xgbexcel import XGBtoExcel

    generator = np.random.default_rng(0)
    X = generator.integers(0, 100, (60, 2)).astype(float)
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1).fit(X, X[:, 0] * 0.5)
    converter = XGBtoExcel(model)

    for name in (
        "expression",
        "expressions",
        "expression_trees",
        "feature_names",
        "base_scores",
        "objective",
        "link",
        "n_features",
        "n_outputs",
        "n_trees_used",
        "rename_features",
        "save_expr",
    ):
        assert hasattr(converter, name), f"README documents {name}, which does not exist"
