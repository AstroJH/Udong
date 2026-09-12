"""Tests for the ESO ObsCore catalog client (`udong.catalogs.obscore`).

Hermetic: the network layer is monkeypatched; no real ESO requests.
"""

import pytest
from astropy import units as u
from astropy.table import Table

from udong.catalogs.obscore import (
    EsoQueryError,
    pixel_scale_mode,
    query_obscore,
    search_muse_products,
)

COLS = [
    "dp_id", "instrument_name", "target_name", "s_ra", "s_dec", "s_pixel_scale",
    "dataproduct_type", "em_min", "em_max", "t_exptime", "access_url",
    "access_estsize",
]
SAMPLE = {
    "metadata": [{"name": c} for c in COLS],
    "data": [
        ["ADP.2024-04-30T18:20:44.624", "MUSE", "Ton S 180", 14.334779,
         -22.382177, 0.2, "cube", 4.70041e-07, 9.35166e-07, 3909.62,
         "http://archive.eso.org/datalink/links?ID=x", 3105921],
        ["ADP.2026-08-07T13:19:46.251", "MUSE", "Ton S 180", 14.334782,
         -22.382176, 0.0253, "cube", 4.74957e-07, 9.34957e-07, 3793.1,
         "http://archive.eso.org/datalink/links?ID=y", 3285233],
    ],
}


def _install_fake(monkeypatch, payload: dict | None = None, exc: Exception | None = None):
    import json

    import udong.catalogs.obscore as mod

    captured: dict = {}

    def fake_get_text(url, timeout=120.0):
        captured["url"] = url
        if exc is not None:
            raise exc
        from urllib.parse import parse_qs, unquote, urlparse

        captured["adql"] = unquote(parse_qs(urlparse(url).query)["QUERY"][0])
        return json.dumps(payload if payload is not None else SAMPLE)

    monkeypatch.setattr(mod, "_get_text", fake_get_text)
    return captured


def test_pixel_scale_mode():
    assert pixel_scale_mode(0.2) == "WFM"
    assert pixel_scale_mode(0.0253) == "NFM"
    assert pixel_scale_mode(1.0) == "unknown"
    assert pixel_scale_mode(None) == "unknown"


def test_query_parses_units_and_rows(monkeypatch):
    _install_fake(monkeypatch)
    tab = query_obscore(instrument="MUSE")
    assert isinstance(tab, Table)
    assert len(tab) == 2
    assert tab.colnames == COLS
    assert tab["dp_id"][0] == "ADP.2024-04-30T18:20:44.624"
    assert tab["s_ra"].unit == u.deg
    assert tab["s_pixel_scale"].unit == u.arcsec
    assert tab["em_min"].unit == u.m
    assert tab["access_estsize"].unit == u.kbyte


def test_search_muse_defaults_and_coords(monkeypatch):
    cap = _install_fake(monkeypatch)
    search_muse_products(ra=14.3343, dec=-22.3822, radius=0.5)
    adql = cap["adql"]
    assert "instrument_name LIKE '%MUSE%'" in adql
    assert "dataproduct_type = 'cube'" in adql
    assert "CIRCLE('ICRS', 14.33430000, -22.38220000, 0.50000000)" in adql
    assert "TOP" not in adql


def test_top_and_target_filter(monkeypatch):
    cap = _install_fake(monkeypatch)
    query_obscore(target="Ton S 180", instrument="MUSE", top=5)
    adql = cap["adql"]
    assert "SELECT TOP 5 " in adql
    assert "target_name LIKE '%Ton S 180%'" in adql


def test_bad_arguments():
    with pytest.raises(ValueError):
        query_obscore(ra=1.0, dec=2.0)  # radius missing
    with pytest.raises(ValueError):
        query_obscore(columns=["bad column name"])
    with pytest.raises(ValueError):
        query_obscore(columns=["s_ra; DROP TABLE"])


def _broken_client(exc):
    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_text(self, url):
            raise exc

        def close(self):
            pass

    return _Client()


def test_error_body_is_included(monkeypatch):
    import udong.data.net as net

    exc = RuntimeError("boom")
    exc.status = 400
    exc.body_snippet = "Encountered UPPER()"
    monkeypatch.setattr(net, "http_client", lambda **kw: _broken_client(exc))
    with pytest.raises(EsoQueryError, match="Encountered UPPER"):
        search_muse_products()


def test_httperror_status_and_snippet(monkeypatch):
    import udong.data.net as net

    exc = RuntimeError("400 Client Error")
    exc.status = 400
    exc.body_snippet = "VOTable error detail"
    monkeypatch.setattr(net, "http_client", lambda **kw: _broken_client(exc))
    with pytest.raises(EsoQueryError, match="HTTP 400.*VOTable error detail"):
        search_muse_products()


def test_http_error_raises_eso_query_error(monkeypatch):
    import udong.data.net as net

    exc = RuntimeError("server says no")
    exc.response = type("_R", (), {"status_code": 500, "text": "VOTable error detail"})()
    monkeypatch.setattr(net, "http_client", lambda **kw: _broken_client(exc))
    with pytest.raises(EsoQueryError):
        search_muse_products()


def test_network_error_raises_eso_query_error(monkeypatch):
    import udong.data.net as net

    monkeypatch.setattr(net, "http_client", lambda **kw: _broken_client(ConnectionError("no route")))
    with pytest.raises(EsoQueryError):
        search_muse_products()
