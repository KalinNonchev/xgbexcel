# xgbexcel

Python package that converts an XGBRegressor model to an Excel formula expression.

## How to start

First, you have to install the package.

```bash
pip install xgbexcel
```

## How to convert XGBRegressor model to an Excel formula

1. Load packages
```
from xgbexcel import XGBtoExcel
import numpy as np
from xgboost import XGBRegressor
```

2. Create dummy dataset and fit XGBRegressor model
```
X, y = np.random.randint(0, 1000, (100, 2)), np.random.randint(0, 10, 100)
model = XGBRegressor(n_estimators=2, max_depth=1)
model.fit(X, y)
```

3. Convert XGBRegressor model to an Excel formula
```
xgb_excel_expr = XGBtoExcel(model)
xgb_excel_expr.expression
```

4. Save Excel expression to a file
```
xgb_excel_expr.save_expr('dummy.txt')
```
5. You can copy the expression in the Excel sheet as a formula

