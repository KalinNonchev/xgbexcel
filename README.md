[![Downloads](https://static.pepy.tech/badge/xgbexcel)](https://pepy.tech/project/xgbexcel) ![Python package](https://github.com/KalinNonchev/xgbexcel/actions/workflows/python-package.yml/badge.svg) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/KalinNonchev/xgbexcel/issues)
# xgbexcel

Python package that converts an XGBRegressor model to an Excel formula expression.

## NB: The package works with xgboost==1.7.1. In [xgboost==2.0](https://github.com/dmlc/xgboost/releases/tag/v2.0.0), vector leaves were introduced. While I may have limited time to address this myself, contributions and improvements are always welcome. Start from [here](https://github.com/dmlc/xgboost/releases/tag/v2.0.0)

## How to start

First, you have to install the package.

```bash
pip install xgbexcel
```

## How to convert XGBRegressor model to an Excel formula (same for XGBClassifier)

1. Load packages
```
from xgbexcel import XGBtoExcel
import numpy as np
from xgboost import XGBRegressor
```

2. Create dummy dataset and fit XGBRegressor model
```
X_train, y_train = np.random.randint(0, 1000, (100, 2)), np.random.randint(0, 10, 100)
model = XGBRegressor(n_estimators=2, max_depth=1)
model.fit(X_train, y_train)
```

3. Convert XGBRegressor model to an Excel formula
```
xgb_excel_expr = XGBtoExcel(model)
xgb_excel_expr.expression
```

4. The features in the Excel formula are represented using the `x1`, `x2`, `x3`, etc. notation, where the numbers correspond to the enumeration of the features in the `XGBRegressor` model and `X_train`. You can manually rename the features in the Excel formula to match the desired column names in the Excel sheet. Once you have renamed the features, you can copy the expression in the Excel sheet as a formula.

```
feature_map = {'x1': 'feature1', 'x2': 'feature2'}
xgb_excel_expr.rename_features(feature_map)
xgb_excel_expr.expression
```

5. Save Excel expression to a file
```
xgb_excel_expr.save_expr('dummy.txt')
```

6. **Enjoy**

Try it yourself in the example notebook: `howXGBtoExcel.ipynb` 
