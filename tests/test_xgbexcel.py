import os
import pytest
import xgboost as xgb
from xgbexcel import XGBtoExcel

# Generate the data to fit the model
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
              [11, 1], [13, 3], [1, 16], [17, 3], [19, 20]]
y = [0, 1, 2, 2, 1, 1, 2, 2, 1, 2]


def test_xgb_regressor_to_excel_expr():
    # Fit the XGBoost Regressor model
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
    model.fit(X, y)
    # Initialize the XGBtoExcel object
    xgb_to_excel = XGBtoExcel(model)
    print(xgb_to_excel.expression)
    # Test that the expression attribute is a string
    assert isinstance(xgb_to_excel.expression, str)
    
    # Test that the expression attribute is not empty
    assert xgb_to_excel.expression == "((IF((x2<=2.5),0,0.300000012)+IF((x2<=2.5),0,0.220000014))+0.5)", xgb_to_excel.expression

def test_xgb_clf_to_excel_expr():
    # Fit the XGBoost Classifier model
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1)
    model.fit(X, y)
    # Initialize the XGBtoExcel object
    xgb_to_excel = XGBtoExcel(model)
    print(xgb_to_excel.expression)
    # Test that the expression attribute is a string
    assert isinstance(xgb_to_excel.expression, str)
    
    # Test that the expression attribute is not empty
    assert xgb_to_excel.expression == "(EXP(((IF((x1<=4),-7.66345476e-09,-0.170270279)+IF((x1<=4),-0.00112346245,-0.153451324))+0.5))/(EXP(((IF((x1<=4),-7.66345476e-09,-0.170270279)+IF((x1<=4),-0.00112346245,-0.153451324))+0.5))+EXP(((IF((x1<=8),-0.0620689802,0.124137931)+IF((x2<=5),0.107327893,-0.0584098995))+0.5))+EXP(((IF((x2<=5),-0.0620689802,0.217241377)+IF((x2<=5),-0.0582885109,0.178286016))+0.5)))) , (EXP(((IF((x1<=8),-0.0620689802,0.124137931)+IF((x2<=5),0.107327893,-0.0584098995))+0.5))/(EXP(((IF((x1<=4),-7.66345476e-09,-0.170270279)+IF((x1<=4),-0.00112346245,-0.153451324))+0.5))+EXP(((IF((x1<=8),-0.0620689802,0.124137931)+IF((x2<=5),0.107327893,-0.0584098995))+0.5))+EXP(((IF((x2<=5),-0.0620689802,0.217241377)+IF((x2<=5),-0.0582885109,0.178286016))+0.5)))) , (EXP(((IF((x2<=5),-0.0620689802,0.217241377)+IF((x2<=5),-0.0582885109,0.178286016))+0.5))/(EXP(((IF((x1<=4),-7.66345476e-09,-0.170270279)+IF((x1<=4),-0.00112346245,-0.153451324))+0.5))+EXP(((IF((x1<=8),-0.0620689802,0.124137931)+IF((x2<=5),0.107327893,-0.0584098995))+0.5))+EXP(((IF((x2<=5),-0.0620689802,0.217241377)+IF((x2<=5),-0.0582885109,0.178286016))+0.5))))", xgb_to_excel.expression


def test_rename_features():
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
    model.fit(X, y)
    # Initialize the XGBtoExcel object
    xgb_to_excel = XGBtoExcel(model)
    # Test that the rename_features method works
    feature_names = {#"x1": "feature_1", 
                     "x2": "feature_2"}
    
    xgb_to_excel.rename_features(feature_names)
    
    for old_name, new_name in feature_names.items():
        assert old_name not in xgb_to_excel.expression, old_name
        assert new_name in xgb_to_excel.expression, new_name

def test_save_expr():
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1)
    model.fit(X, y)
    # Initialize the XGBtoExcel object
    xgb_to_excel = XGBtoExcel(model)
    # Test that the save_expr method works
    xgb_to_excel.save_expr("test.txt")
    with open("test.txt", "r") as f:
        saved_expr = f.read()
    assert saved_expr == xgb_to_excel.expression, xgb_to_excel.expression
    
    os.remove("test.txt")
