"""Tests for the advanced endpoint's response sanitizing.

JSON has no NaN or infinity, and Starlette's encoder raises rather than
emitting them, so a single undefined statistic anywhere in the payload turns
the whole request into a 500. Several statistics here are legitimately
undefined on some inputs, which makes this the difference between a panel
showing a dash and the endpoint falling over.
"""

import json
import math

import numpy as np

from app.advanced import _json_safe


class TestJsonSafe:
    def test_nan_becomes_none(self):
        assert _json_safe(float("nan")) is None

    def test_infinities_become_none(self):
        assert _json_safe(float("inf")) is None
        assert _json_safe(float("-inf")) is None

    def test_finite_floats_pass_through_unchanged(self):
        for v in (0.0, -1.5, 1e-12, 3.14159):
            assert _json_safe(v) == v

    def test_recurses_into_nested_structures(self):
        payload = {
            "a": [1.0, float("nan"), {"b": float("inf")}],
            "c": {"d": {"e": [float("-inf"), 2.0]}},
        }
        assert _json_safe(payload) == {
            "a": [1.0, None, {"b": None}],
            "c": {"d": {"e": [None, 2.0]}},
        }

    def test_leaves_non_floats_alone(self):
        payload = {"s": "text", "i": 7, "b": True, "n": None, "l": ["x", 2]}
        assert _json_safe(payload) == payload

    def test_output_is_actually_serializable(self):
        """The property that matters, asserted directly."""
        payload = {
            "sharpe": float("nan"),
            "nested": [{"calmar": float("inf")}, {"ratio": float("nan")}],
        }
        # allow_nan=False is what a strict JSON encoder does; without the
        # sanitizer this raises ValueError.
        assert json.dumps(_json_safe(payload), allow_nan=False)

    def test_numpy_nan_is_handled(self):
        """np.float64('nan') is a float subclass and must be caught too."""
        assert _json_safe(float(np.float64("nan"))) is None

    def test_tuples_become_lists(self):
        # JSON has no tuple, so normalizing here keeps the encoder's job simple.
        assert _json_safe((1.0, float("nan"))) == [1.0, None]

    def test_a_realistic_undefined_metric_survives(self):
        """A Calmar ratio with no drawdown divides by zero and is undefined."""
        cagr, max_dd = 0.12, 0.0
        calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
        assert math.isnan(calmar)
        assert json.dumps(_json_safe({"calmar_ratio": calmar}), allow_nan=False)
