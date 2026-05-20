import pytest

from tools import calculator, get_current_year


def test_calculator_normal():
    res = calculator("2 + 3 * 4")
    assert res["error"] is None
    assert res["result"] == "14"


def test_calculator_malformed():
    res = calculator("2 +")
    assert res["result"] == ""
    assert res["error"] is not None
    assert "Parse error" in res["error"] or "Disallowed node" in res["error"] or "Evaluation error" in res["error"]


def test_calculator_security_rejects_names_and_calls():
    # Attempt to use __import__ or attribute access should be rejected by AST whitelist
    malicious = "__import__('os').system('echo hello')"
    res = calculator(malicious)
    assert res["result"] == ""
    assert res["error"] is not None
    assert "Disallowed node" in res["error"] or "Unsupported" in res["error"]


def test_get_current_year():
    res = get_current_year()
    assert res["error"] is None
    assert res["result"] == "2026"
