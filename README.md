[![CI](https://github.com/KalinNonchev/xgbexcel/actions/workflows/ci.yml/badge.svg)](https://github.com/KalinNonchev/xgbexcel/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/xgbexcel)](https://pypi.org/project/xgbexcel/)
[![Downloads](https://static.pepy.tech/badge/xgbexcel)](https://pepy.tech/project/xgbexcel)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# xgbexcel

Convert a trained XGBoost model into an Excel formula, so it can be scored in a
spreadsheet with no Python and no add-ins.

Works with XGBoost 1.7 through 3.x.

## Install

```bash
pip install xgbexcel
```

## Use

```python
import numpy as np
from xgboost import XGBRegressor
from xgbexcel import XGBtoExcel

X = np.random.randint(0, 1000, (100, 2))
y = np.random.randint(0, 10, 100)
model = XGBRegressor(n_estimators=2, max_depth=1).fit(X, y)

formula = XGBtoExcel(model)
print(formula.expression)
```

Features are called `x1`, `x2`, ... in training column order. Give them real names
either up front or afterwards:

```python
XGBtoExcel(model, feature_names=["age", "income"])

formula.rename_features({"x1": "age", "x2": "income"})
```

Renaming matches whole names, so renaming `x1` leaves `x10` alone.

Write the formula out, then paste it into a cell:

```python
formula.save_expr("model.txt")
```

Multi-output models (multiclass classification, multi-output regression) produce one
formula per output:

```python
classifier = XGBClassifier(n_estimators=3).fit(X, y)
for label, expr in zip(classifier.classes_, XGBtoExcel(classifier).expressions):
    print(label, expr)
```

## What is supported

| | |
|---|---|
| Estimators | `XGBRegressor`, `XGBClassifier`, and raw `xgboost.Booster` |
| Objectives | `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`, `reg:quantileerror`, `reg:logistic`, `binary:logistic`, `binary:logitraw`, `multi:softmax`, `multi:softprob`, `count:poisson`, `reg:gamma`, `reg:tweedie` |
| Model shapes | Multiclass, multi-output regression, vector leaves (`multi_strategy="multi_output_tree"`), forest mode (`num_parallel_tree`), dart, early stopping |

Anything outside that list raises `UnsupportedModelError` rather than returning a
formula that quietly computes the wrong number. That includes categorical splits,
`gblinear`, and ranking or survival objectives.

## Limitations

Excel caps a formula at 8192 characters and 64 levels of nested functions. A tree
ensemble outgrows that quickly: XGBoost's default of 100 trees at depth 6 produces
something in the order of 150,000 characters. `XGBtoExcel` warns when a formula is too
long. To stay inside the limit, either train a smaller model or put one tree per helper
cell and sum those cells. In practice the approach suits small models: a handful of
trees at shallow depth.

Blank cells are not missing values. XGBoost routes NaN down a branch the model chose
during training, while Excel treats a blank cell as 0 and compares it as such. Fill
every input cell, or the row takes whichever branch 0 leads to.

Predictions are reproduced to float32 accuracy, which is the precision XGBoost itself
computes in. Expect agreement to roughly 1e-6 relative, not to the last bit.

## Accuracy

XGBoost compares `float32(value) < float32(threshold)`, and it picks thresholds that
are feature values from the training data, so rows sitting exactly on a split are
common rather than rare. Excel has no float32 and compares in float64, which sends
those rows the other way.

`xgbexcel` compensates by emitting the midpoint between the float32 threshold and the
float32 immediately below it, which makes a float64 comparison agree with XGBoost's
float32 one for every input. The test suite asserts row-by-row equality with
`model.predict` rather than checking an average, because an average hides exactly this
class of error.

This is why thresholds in the output look slightly off. A split that XGBoost reports as
`981` is written as `980.9999694824219`. That is deliberate, and the two behave
identically on every input the model can see.

## Contributing

```bash
git clone https://github.com/KalinNonchev/xgbexcel
cd xgbexcel
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

Issues and pull requests are welcome at
[github.com/KalinNonchev/xgbexcel](https://github.com/KalinNonchev/xgbexcel/issues).

## License

MIT. See [LICENSE](LICENSE).
