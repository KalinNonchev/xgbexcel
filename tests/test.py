import os
import pytest
import numpy as np
import xgboost as xgb
from xgbexcel import XGBtoExcel

# Set the seed for the random number generator
np.random.seed(13)

# Generate some random data to fit the model
X = np.random.rand(100, 2)
y = np.random.rand(100)

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
    assert xgb_to_excel.expression == "((IF((x1<=0.255339563),-0.0447236858,0.00825651549)+IF((x2<=0.717021346),0.00409270683,-0.038642168))+0.5)"

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
