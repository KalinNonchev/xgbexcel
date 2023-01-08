import os
import pytest
import xgboost as xgb
from xgbexcel import XGBtoExcel

# Generate the data to fit the model
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
              [11, 1], [13, 3], [1, 16], [17, 3], [19, 20]]
y = [100, 1, 2, 2, 1, 1, 2, 2, 1, 2]

# Fit the XGBoost model
model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
model.fit(X, y)

def test_xgb_to_excel_expr():
    # Initialize the XGBtoExcel object
    xgb_to_excel = XGBtoExcel(model)
    print(xgb_to_excel.expression)
    # Test that the expression attribute is a string
    assert isinstance(xgb_to_excel.expression, str)
    
    # Test that the expression attribute is not empty
    assert xgb_to_excel.expression == "((IF((x1<=2),10.1000004,0.266666681)+IF((x2<=2.5),8.96333408,-0.0988889039))+0.5)"

def test_rename_features():
    # Initialize the XGBtoExcel object
    xgb_to_excel = XGBtoExcel(model)
    # Test that the rename_features method works
    feature_names = {"x1": "feature_1", 
                     "x2": "feature_2"}
    
    xgb_to_excel.rename_features(feature_names)
    
    for old_name, new_name in feature_names.items():
        assert old_name not in xgb_to_excel.expression, old_name
        assert new_name in xgb_to_excel.expression, new_name

def test_save_expr():
    # Initialize the XGBtoExcel object
    xgb_to_excel = XGBtoExcel(model)
    # Test that the save_expr method works
    xgb_to_excel.save_expr("test.txt")
    with open("test.txt", "r") as f:
        saved_expr = f.read()
    assert saved_expr == xgb_to_excel.expression
    
    os.remove("test.txt")
