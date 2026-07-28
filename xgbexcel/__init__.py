"""Convert trained XGBoost models into Excel formula expressions."""

from xgbexcel import convert
from xgbexcel.convert import EXCEL_MAX_FORMULA_CHARS, UnsupportedModelError, XGBtoExcel

__version__ = "1.0.0"

__all__ = [
    "EXCEL_MAX_FORMULA_CHARS",
    "UnsupportedModelError",
    "XGBtoExcel",
    "__version__",
    "convert",
]
