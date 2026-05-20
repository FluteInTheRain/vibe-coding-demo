import pytest

from tools import calculator, get_current_year, web_search


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


def test_web_search_missing_key(monkeypatch):
    monkeypatch.setenv('TAVILY_API_KEY', '')
    res = web_search('test')
    assert res['result'] == ''
    assert res['error'] is not None
    assert 'TAVILY_API_KEY' in res['error']


def test_web_search_network_error(monkeypatch):
    class FakeExc(Exception):
        pass

    def fake_post(*args, **kwargs):
        raise Exception('network fail')

    monkeypatch.setenv('TAVILY_API_KEY', 'key')
    monkeypatch.setattr('httpx.post', lambda *a, **k: (_ for _ in ()).throw(Exception('network fail')))

    res = web_search('test')
    assert res['result'] == ''
    assert res['error'] is not None


def test_web_search_success(monkeypatch):
    class FakeResp:
        def __init__(self, data):
            self.status_code = 200
            self._data = data
        def json(self):
            return self._data

    data = {'results': [
        {'title': 'T1', 'snippet': 'S1'},
        {'title': 'T2', 'snippet': 'S2'},
        {'title': 'T3', 'snippet': 'S3'},
    ]}

    monkeypatch.setenv('TAVILY_API_KEY', 'key')
    monkeypatch.setattr('httpx.post', lambda *a, **k: FakeResp(data))

    res = web_search('query')
    assert res['error'] is None
    assert 'T1' in res['result']
    assert 'T2' in res['result']
