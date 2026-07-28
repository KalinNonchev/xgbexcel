"""Unit tests for the helper functions, including the paths models rarely reach."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xgbexcel.convert import (
    UnsupportedModelError,
    XGBtoExcel,
    _add_constant,
    _excel_threshold,
    _format_number,
    _join_sum,
    _prob_to_margin,
)


class TestFormatNumber:
    def test_round_trips(self):
        for value in (0.0, -1.5, 3.14159265358979, 1e-30, -2.5e18):
            assert float(_format_number(value)) == value

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite(self, value):
        with pytest.raises(UnsupportedModelError, match="non-finite"):
            _format_number(value)


class TestJoinSum:
    def test_empty(self):
        assert _join_sum([]) == "0.0"

    def test_single(self):
        assert _join_sum(["A"]) == "A"

    def test_positive_terms(self):
        assert _join_sum(["A", "B", "C"]) == "A+B+C"

    def test_negative_term_does_not_produce_double_operator(self):
        assert _join_sum(["A", "-1.5"]) == "A-1.5"
        assert "+-" not in _join_sum(["A", "-1.5", "-2.0"])

    def test_leading_negative_is_preserved(self):
        assert _join_sum(["-1.0", "B"]) == "-1.0+B"


class TestAddConstant:
    def test_zero_only_parenthesises(self):
        assert _add_constant("A", 0.0) == "(A)"

    def test_positive(self):
        assert _add_constant("A", 2.5) == "(A+2.5)"

    def test_negative_uses_a_minus(self):
        assert _add_constant("A", -2.5) == "(A-2.5)"


class TestProbToMargin:
    def test_identity_passes_through(self):
        assert _prob_to_margin(0.7, "identity") == 0.7

    def test_softmax_passes_through(self):
        assert _prob_to_margin(-0.05, "softmax") == -0.05

    def test_logistic_is_the_logit(self):
        assert _prob_to_margin(0.5, "logistic") == pytest.approx(0.0)
        assert _prob_to_margin(0.75, "logistic") == pytest.approx(math.log(3.0))

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
    def test_logistic_rejects_values_outside_the_unit_interval(self, value):
        with pytest.raises(UnsupportedModelError, match="outside"):
            _prob_to_margin(value, "logistic")

    def test_exp_is_the_log(self):
        assert _prob_to_margin(math.e, "exp") == pytest.approx(1.0)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_exp_rejects_non_positive(self, value):
        with pytest.raises(UnsupportedModelError, match="positive"):
            _prob_to_margin(value, "exp")


class TestExcelThreshold:
    def test_shifts_below_the_float32_value(self):
        for value in (-3.5, -1e-9, 0.0, 1e-9, 42.0):
            assert _excel_threshold(value) < float(np.float32(value)) or value == 0.0

    def test_handles_the_float32_minimum_without_producing_infinity(self):
        smallest = float(np.finfo(np.float32).min)
        result = _excel_threshold(smallest)
        assert math.isfinite(result)

    def test_zero_shifts_to_a_negative_subnormal(self):
        assert _excel_threshold(0.0) < 0.0

    def test_is_deterministic(self):
        assert _excel_threshold(1.2345) == _excel_threshold(1.2345)


class TestParseBaseScore:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("[5.0078073E0]", [5.0078073]),
            ("5E-1", [0.5]),
            ("[-5.0e-2,8.3e-4,4.9e-2]", [-0.05, 0.00083, 0.049]),
            ("0.5", [0.5]),
            (0.25, [0.25]),
            (3, [3.0]),
        ],
    )
    def test_accepts_every_format_xgboost_has_used(self, raw, expected):
        assert XGBtoExcel._parse_base_score(raw) == pytest.approx(expected)
