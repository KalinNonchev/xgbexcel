"""Convert trained XGBoost models into Excel formula expressions."""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["EXCEL_MAX_FORMULA_CHARS", "UnsupportedModelError", "XGBtoExcel"]

# Excel refuses a formula longer than this, and nests functions no deeper than 64.
EXCEL_MAX_FORMULA_CHARS = 8192


class UnsupportedModelError(ValueError):
    """Raised when a model cannot be represented exactly as an Excel formula."""


_IDENTITY = "identity"
_LOGISTIC = "logistic"
_SOFTMAX = "softmax"
_EXP = "exp"

# XGBoost keeps base_score in the model's output space and converts it to the margin
# space internally. We repeat that conversion, or the intercept lands on the wrong
# scale: a logistic model needs log(p / (1 - p)), not p.
_OBJECTIVES: dict[str, str] = {
    "reg:squarederror": _IDENTITY,
    "reg:linear": _IDENTITY,  # pre-1.0 alias
    "reg:absoluteerror": _IDENTITY,
    "reg:pseudohubererror": _IDENTITY,
    "reg:quantileerror": _IDENTITY,
    "binary:logitraw": _IDENTITY,
    "binary:logistic": _LOGISTIC,
    "reg:logistic": _LOGISTIC,
    "multi:softprob": _SOFTMAX,
    "multi:softmax": _SOFTMAX,
    "count:poisson": _EXP,
    "reg:gamma": _EXP,
    "reg:tweedie": _EXP,
}


def _excel_threshold(value: float) -> float:
    """Shift a split threshold so a float64 comparison reproduces XGBoost's float32 one.

    XGBoost evaluates ``float32(x) < float32(t)``. Excel has no float32 and evaluates
    ``x < t`` in float64. Those disagree whenever ``x`` rounds to exactly ``t`` in
    float32, which happens constantly because XGBoost picks thresholds that *are*
    feature values from the training data.

    ``float32(x) >= float32(t)`` exactly when ``x`` reaches the midpoint between
    ``float32(t)`` and the float32 below it, so emitting that midpoint makes the two
    comparisons agree for every input.
    """
    exact = np.float32(value)
    with np.errstate(over="ignore"):
        # Stepping below the smallest float32 overflows to -inf, which is handled next.
        below = np.nextafter(exact, np.float32("-inf"))
    if not np.isfinite(below):
        return float(exact)
    return (float(exact) + float(below)) / 2.0


def _prob_to_margin(value: float, link: str) -> float:
    if link == _LOGISTIC:
        if not 0.0 < value < 1.0:
            raise UnsupportedModelError(
                f"base_score {value} is outside (0, 1) for a logistic objective"
            )
        return math.log(value / (1.0 - value))
    if link == _EXP:
        if value <= 0.0:
            raise UnsupportedModelError(
                f"base_score {value} must be positive for a log-link objective"
            )
        return math.log(value)
    return value


def _format_number(value: Any) -> str:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise UnsupportedModelError(
            f"model contains a non-finite value ({number}) that Excel cannot represent"
        )
    return repr(number)


def _join_sum(terms: Sequence[str]) -> str:
    """Join with '+' without emitting 'a+-b', and without nesting once per term."""
    if not terms:
        return "0.0"
    parts = [terms[0]]
    for term in terms[1:]:
        parts.append(f"-{term[1:]}" if term.startswith("-") else f"+{term}")
    return "".join(parts)


def _add_constant(expression: str, constant: float) -> str:
    if constant == 0.0:
        return f"({expression})"
    sign = "-" if constant < 0 else "+"
    return f"({expression}{sign}{_format_number(abs(constant))})"


class XGBtoExcel:
    """Convert a trained XGBoost model into Excel formula expressions.

    Accepts any fitted XGBoost estimator (``XGBRegressor``, ``XGBClassifier``, ...) or a
    raw :class:`xgboost.Booster`.

    Features are named ``x1``, ``x2``, ... in training column order unless the booster
    carries names or ``feature_names`` is given. Multi-output models produce one formula
    per output in :attr:`expressions`.

    Args:
        xgb_model: A fitted XGBoost estimator or Booster.
        feature_names: Names for the input features, in training column order.
        sep: Argument separator for Excel functions. Use ``";"`` in locales where the
            comma is the decimal separator.
        warn_on_size: Warn when a formula exceeds Excel's length limit.

    Raises:
        UnsupportedModelError: For objectives, boosters or splits that cannot be
            represented exactly, rather than emitting a formula that is quietly wrong.
    """

    def __init__(
        self,
        xgb_model: Any,
        feature_names: Sequence[str] | None = None,
        sep: str = ",",
        warn_on_size: bool = True,
    ) -> None:
        booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model
        if not hasattr(booster, "save_raw"):
            raise TypeError(
                "expected a fitted XGBoost estimator or xgboost.Booster, "
                f"got {type(xgb_model).__name__}"
            )

        raw = json.loads(bytes(booster.save_raw(raw_format="json")))
        learner = raw["learner"]
        gradient_booster = learner["gradient_booster"]

        booster_name = gradient_booster.get("name", "gbtree")
        # A dart booster nests its trees under "gbtree" on XGBoost 2.0 through 3.0, and
        # flattens into a plain gbtree on newer builds. Either way the trees are the
        # same; the per-tree weights in "weight_drop" carry the dropout.
        tree_container = gradient_booster
        if booster_name == "dart":
            tree_container = gradient_booster.get("gbtree", gradient_booster)
        elif booster_name != "gbtree":
            raise UnsupportedModelError(
                f"booster {booster_name!r} is not supported; only tree boosters can be "
                "written as an exact Excel formula"
            )
        if "model" not in tree_container:
            raise UnsupportedModelError(
                f"could not locate the tree model in a {booster_name!r} booster"
            )

        self.objective: str = learner["objective"]["name"]
        if self.objective not in _OBJECTIVES:
            raise UnsupportedModelError(
                f"objective {self.objective!r} is not supported. Supported: "
                + ", ".join(sorted(_OBJECTIVES))
            )
        self.link: str = _OBJECTIVES[self.objective]

        model = tree_container["model"]
        model_param = learner["learner_model_param"]

        self.sep = sep
        self.n_features: int = int(model_param["num_feature"])
        num_class = int(model_param.get("num_class", 0) or 0)
        num_target = int(model_param.get("num_target", 1) or 1)
        self.n_outputs: int = max(num_class, num_target, 1)
        self.feature_names: list[str] = self._resolve_feature_names(booster, feature_names)

        trees = model["trees"]
        self._reject_categorical_splits(trees)
        self._leaf_size = int(trees[0]["tree_param"].get("size_leaf_vector", 1) or 1)
        self._vector_leaves = self._leaf_size > 1

        self.n_trees_used: int = self._trees_in_use(
            xgb_model, booster, model, len(trees), self.n_outputs
        )
        used = trees[: self.n_trees_used]
        tree_info = model["tree_info"][: self.n_trees_used]

        # A dart model serialises as a gbtree carrying a per-tree weight. Trees that
        # were dropped during training contribute a scaled amount, not a full one.
        weights = gradient_booster.get("weight_drop") or []
        self._tree_weights: list[float] = [
            float(weights[i]) if i < len(weights) else 1.0 for i in range(self.n_trees_used)
        ]

        base_scores = self._parse_base_score(model_param["base_score"])
        if len(base_scores) == 1 and self.n_outputs > 1:
            base_scores = base_scores * self.n_outputs
        if len(base_scores) != self.n_outputs:
            raise UnsupportedModelError(
                f"model reports {self.n_outputs} outputs but {len(base_scores)} base scores"
            )
        self.base_scores: list[float] = [_prob_to_margin(b, self.link) for b in base_scores]

        self.expression_trees: list[str] = [self._render_tree(tree) for tree in used]
        self.expressions: list[str] = self._build_expressions(used, tree_info)

        if warn_on_size:
            self._warn_if_oversized()

    # ------------------------------------------------------------------ public API

    @property
    def expression(self) -> str:
        """The Excel formula.

        A single-output model gives one formula. A multi-output model gives the per
        output formulas joined by ``" , "``; read :attr:`expressions` instead there.
        """
        if len(self.expressions) == 1:
            return self.expressions[0]
        return " , ".join(self.expressions)

    def rename_features(self, feature_names: Mapping[str, str]) -> XGBtoExcel:
        """Rename features in every generated formula.

        Replacements happen simultaneously and match whole names only, so renaming
        ``x1`` leaves ``x10`` untouched and a freshly written name is never rewritten
        again by a later entry. Returns self, so calls chain.
        """
        if not feature_names:
            return self
        unknown = set(feature_names) - set(self.feature_names)
        if unknown:
            raise KeyError(f"unknown feature(s): {sorted(unknown)}. Known: {self.feature_names}")

        pattern = re.compile(
            r"(?<![A-Za-z0-9_.])(?:"
            + "|".join(re.escape(name) for name in sorted(feature_names, key=len, reverse=True))
            + r")(?![A-Za-z0-9_])"
        )
        mapping = dict(feature_names)
        self.expressions = [
            pattern.sub(lambda m: mapping[m.group(0)], expr) for expr in self.expressions
        ]
        self.expression_trees = [
            pattern.sub(lambda m: mapping[m.group(0)], expr) for expr in self.expression_trees
        ]
        self.feature_names = [mapping.get(name, name) for name in self.feature_names]
        return self

    def save_expr(self, out_file: str) -> None:
        """Write the formula to a text file, one line per output."""
        with open(out_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.expressions))

    def __str__(self) -> str:
        return self.expression

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(objective={self.objective!r}, "
            f"n_features={self.n_features}, n_outputs={self.n_outputs}, "
            f"n_trees={self.n_trees_used})"
        )

    # ------------------------------------------------------------------- internals

    def _resolve_feature_names(
        self, booster: Any, feature_names: Sequence[str] | None
    ) -> list[str]:
        if feature_names is not None:
            names = list(feature_names)
            if len(names) != self.n_features:
                raise ValueError(
                    f"model has {self.n_features} features but {len(names)} names were given"
                )
            return names
        stored = getattr(booster, "feature_names", None)
        if stored and len(stored) == self.n_features:
            return list(stored)
        return [f"x{i + 1}" for i in range(self.n_features)]

    @staticmethod
    def _reject_categorical_splits(trees: Sequence[dict]) -> None:
        for tree in trees:
            if any(kind != 0 for kind in tree.get("split_type", ())):
                raise UnsupportedModelError(
                    "categorical splits are not supported; retrain with "
                    "enable_categorical=False or one-hot encode the categories"
                )

    @staticmethod
    def _trees_in_use(
        estimator: Any, booster: Any, model: dict, n_trees: int, n_outputs: int
    ) -> int:
        """Honour early stopping: predict() stops at best_iteration, the dump does not."""
        best = getattr(estimator, "best_iteration", None)
        if best is None:
            best = getattr(booster, "best_iteration", None)
        if best is None:
            return n_trees

        indptr = model.get("iteration_indptr")
        if indptr and best + 1 < len(indptr):
            return int(indptr[best + 1])

        # XGBoost 1.x does not record iteration_indptr, so derive the boundary from
        # how many trees each boosting round adds.
        per_round = n_outputs * int(model["gbtree_model_param"].get("num_parallel_tree", 1) or 1)
        return min(n_trees, (best + 1) * per_round)

    @staticmethod
    def _parse_base_score(raw: Any) -> list[float]:
        """base_score is a scalar string on xgboost 1.x/2.x and a JSON list on 3.x."""
        if isinstance(raw, (int, float)):
            return [float(raw)]
        text = str(raw).strip()
        if text.startswith("["):
            return [float(value) for value in json.loads(text)]
        return [float(text)]

    def _leaf_value(self, tree: dict, node: int, output: int) -> float:
        if self._vector_leaves:
            return float(tree["base_weights"][node * self._leaf_size + output])
        return float(np.float32(tree["split_conditions"][node]))

    def _render_tree(self, tree: dict, output: int = 0) -> str:
        """Render one tree as a nested IF.

        Iterative post-order, so tree depth never touches Python's recursion limit and
        every node is rendered exactly once.
        """
        left = tree["left_children"]
        right = tree["right_children"]
        conditions = tree["split_conditions"]
        features = tree["split_indices"]

        rendered: dict[int, str] = {}
        stack: list[int] = [0]

        while stack:
            node = stack[-1]
            if node in rendered:
                stack.pop()
                continue

            if left[node] == -1:
                rendered[node] = _format_number(self._leaf_value(tree, node, output))
                stack.pop()
                continue

            child_left, child_right = left[node], right[node]
            if child_left not in rendered or child_right not in rendered:
                if child_right not in rendered:
                    stack.append(child_right)
                if child_left not in rendered:
                    stack.append(child_left)
                continue

            # XGBoost sends a row left when the value is strictly below the threshold.
            threshold = _format_number(_excel_threshold(conditions[node]))
            rendered[node] = (
                f"IF(({self.feature_names[features[node]]}<{threshold})"
                f"{self.sep}{rendered[child_left]}{self.sep}{rendered[child_right]})"
            )
            stack.pop()

        return rendered[0]

    def _weighted(self, expression: str, index: int) -> str:
        weight = self._tree_weights[index]
        if weight == 1.0:
            return expression
        return f"({expression})*{_format_number(weight)}"

    def _margins(self, trees: Sequence[dict], tree_info: Sequence[int]) -> list[str]:
        margins: list[str] = []
        for output in range(self.n_outputs):
            if self._vector_leaves:
                # Every tree feeds every output; the leaf holds a vector.
                terms = [
                    self._weighted(self._render_tree(tree, output), index)
                    for index, tree in enumerate(trees)
                ]
            else:
                terms = [
                    self._weighted(self.expression_trees[index], index)
                    for index, group in enumerate(tree_info)
                    if group == output
                ]
            margins.append(_add_constant(_join_sum(terms), self.base_scores[output]))
        return margins

    def _build_expressions(self, trees: Sequence[dict], tree_info: Sequence[int]) -> list[str]:
        margins = self._margins(trees, tree_info)

        if self.link == _IDENTITY:
            return margins
        if self.link == _EXP:
            return [f"EXP({margin})" for margin in margins]
        if self.link == _LOGISTIC:
            return [f"(1/(1+EXP(-{margin})))" for margin in margins]

        # Softmax inlines every class margin into every denominator, so the formula
        # grows with the square of the class count.
        denominator = _join_sum([f"EXP({margin})" for margin in margins])
        return [f"(EXP({margin})/({denominator}))" for margin in margins]

    def _warn_if_oversized(self) -> None:
        longest = max(len(expression) for expression in self.expressions)
        if longest > EXCEL_MAX_FORMULA_CHARS:
            warnings.warn(
                f"generated formula is {longest:,} characters, over Excel's "
                f"{EXCEL_MAX_FORMULA_CHARS:,} character limit, so Excel will reject it. "
                "Train a smaller model (fewer trees, lower max_depth), or place one "
                "tree per helper cell and sum those cells.",
                UserWarning,
                stacklevel=3,
            )
