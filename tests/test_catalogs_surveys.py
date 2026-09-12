"""Tests for the external-survey registry and easycat Archive adapters."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from udong.catalogs.surveys import (
    SURVEYS,
    SurveyNotImplemented,
    SurveyProduct,
    available_surveys,
    cone_search,
    fetch_cone,
    fetch_one,
    read_product_table,
    register,
    resolve,
)


def test_registry_contents():
    names = available_surveys()
    for expected in ("sdss", "manga", "wise", "ztf", "galex", "gaia", "first", "nvss", "desi"):
        assert expected in names
    for spec in SURVEYS.values():
        assert spec.description.strip()
        assert spec.access.strip()


def test_resolve_aliases():
    assert resolve("WISE").name == "wise"
    assert resolve("allwise").name == "wise"
    assert resolve("legacy-surveys").name == "desi"
    assert resolve("drpall").name == "manga"
    with pytest.raises(KeyError):
        resolve("no-such-survey")


def test_unimplemented_surveys_still_raise():
    for name in ("gaia", "sdss", "first"):
        with pytest.raises(SurveyNotImplemented) as ei:
            cone_search(name, ra=0.0, dec=0.0, radius=1.0 * u.arcmin)
        assert name in str(ei.value)
        assert "planned access" in str(ei.value)


class _FakeItem:
    def __init__(self, obj_id="obj1", data=None, success=True, error="", meta=None):
        self.obj_id = obj_id
        self.data = data
        self.success = success
        self.error = error
        self.meta = meta or {}


class _FakeArchive:
    """Records fetch_one calls and returns a canned item."""

    def __init__(self, item):
        self.item = item
        self.calls = []

    def fetch_one(self, target, **kwargs):
        self.calls.append((target, kwargs))
        dest = kwargs.get("dest")
        if dest is None and kwargs.get("store_dir") is not None:
            dest = kwargs["store_dir"] / "obj1.fits"
        if self.item.data is not None and dest is not None:
            import shutil
            from pathlib import Path

            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.item.data, dest)
            self.item.data = Path(dest)
        return self.item


def _write_fits_table(path):
    t = Table({"ra": [1.0], "dec": [2.0], "w1mpro": [12.3]})
    t.write(path, overwrite=True)
    return path


def test_wise_fetch_one_parses_table(tmp_path, monkeypatch):
    src = _write_fits_table(tmp_path / "wise_src.fits")
    fake = _FakeArchive(_FakeItem(data=src, meta={"url": "https://wise.example/x.fits"}))
    monkeypatch.setattr("udong.catalogs.wise.wise_archive", lambda **kw: fake)

    product = fetch_one("wise", "obj1", dest=tmp_path / "out.fits")
    assert isinstance(product, SurveyProduct)
    assert product.survey == "wise"
    assert product.table is not None and product.table["w1mpro"][0] == 12.3
    assert product.meta["url"].startswith("https://wise.example")
    assert fake.calls[0][1]["dest"] == tmp_path / "out.fits"


def test_ztf_cone_search_returns_table(tmp_path, monkeypatch):
    csv = tmp_path / "ztf.csv"
    csv.write_text("mjd,band,mag\n60000.1,g,19.5\n60001.2,r,19.8\n")

    class _FakeZtf(_FakeArchive):
        def fetch_one(self, target, **kwargs):
            self.calls.append((target, kwargs))
            return _FakeItem(data=csv, meta={"url": "https://ztf.example/lc.csv"})

    fake = _FakeZtf(None)
    factory_kwargs = {}

    def factory(**kw):
        factory_kwargs.update(kw)
        return fake

    monkeypatch.setattr("udong.catalogs.ztf.ztf_archive", factory)

    tab = cone_search("ztf", ra=10.0, dec=-5.0, radius=6 * u.arcsec)
    assert len(tab) == 2 and "mag" in tab.colnames
    assert tab.meta["url"].endswith("lc.csv")
    target, kwargs = fake.calls[0]
    assert target["raj2000"] == 10.0 and target["dej2000"] == -5.0
    # the cone radius configures the Archive (easycat ZTFArchive.radius_arcsec)
    assert factory_kwargs["radius_arcsec"] == pytest.approx(6.0)


def test_desi_file_modes_require_dest_and_are_not_tabular(tmp_path, monkeypatch):
    fits = tmp_path / "desi_spec.fits"
    Table({"wave": [4000.0], "flux": [1.0]}).write(fits, overwrite=True)
    fake = _FakeArchive(_FakeItem(data=fits))
    monkeypatch.setattr("udong.catalogs.desi.desi_archive", lambda **kw: fake)

    # spectra/image return a file, not a Table -> must give dest/store_dir
    with pytest.raises(ValueError, match="pass dest="):
        fetch_one("desi", "obj1", mode="spectra")
    with pytest.raises(ValueError, match="pass dest="):
        fetch_one("desi", "obj1", mode="image")

    product = fetch_one("desi", "obj1", mode="spectra", dest=tmp_path / "spec.fits")
    assert product.table is None and product.path is not None
    with pytest.raises(TypeError, match="tabular"):
        cone_search("desi", ra=1.0, dec=2.0, radius=1.0 * u.arcmin,
                    mode="spectra", dest=tmp_path / "spec2.fits")


def test_read_product_table_rejects_unknown(tmp_path):
    p = tmp_path / "image.jpg"
    p.write_bytes(b"not a table")
    with pytest.raises(TypeError):
        read_product_table(p)


def test_register_custom_product(monkeypatch):
    @register("gaia")
    def _fake_gaia(target, **kwargs):
        return SurveyProduct(survey="gaia", obj_id=str(target), table=Table({"a": [1]}))

    try:
        product = fetch_one("gaia", "x")
        assert product.table is not None
        tab = cone_search("gaia", ra=0.0, dec=0.0, radius=1.0 * u.arcmin)
        assert len(tab) == 1
    finally:
        import udong.catalogs.surveys as mod

        mod._IMPLS.pop("gaia", None)


def test_desi_image_mode_and_cutout_helper(tmp_path, monkeypatch):
    from astropy.io import fits

    from udong.catalogs.desi import fetch_cutout

    src = tmp_path / "src_cutout.fits"
    fits.PrimaryHDU(np.zeros((4, 8, 8), dtype=np.float32)).writeto(src)
    fake = _FakeArchive(_FakeItem(data=src, meta={"url": "https://legacysurvey.org/x"}))
    seen = {}

    def factory(**kw):
        seen.update(kw)
        return fake

    monkeypatch.setattr("udong.catalogs.desi.desi_archive", factory)

    product = fetch_cutout(14.333953, -22.382503, dest=tmp_path / "out.fits",
                           size=64, pixscale=0.25, layer="ls-dr10", obj_id="ton_s180")
    assert product.path == tmp_path / "out.fits"
    assert product.table is None          # image is a file, not a table
    assert seen["mode"] == "image" and seen["image_size"] == 64
    assert seen["image_pixscale"] == 0.25 and seen["image_layer"] == "ls-dr10"
    assert seen["image_format"] == "fits"

    target, call = fake.calls[0]
    assert target["obj_id"] == "ton_s180"
    assert target["raj2000"] == 14.333953 and target["dej2000"] == -22.382503
    assert call["dest"] == tmp_path / "out.fits"
    assert call.get("validate") == "fits"   # FITS validation for file modes
