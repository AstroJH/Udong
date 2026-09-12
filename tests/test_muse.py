"""Tests for the MUSE (ESO Phase 3) data layer (`udong.data.muse`)."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from udong.data.base import CatalogDataset, IFUDataset
from udong.data.manga.dataset import MangaDataset
from udong.data.muse import (
    MuseDataset,
    MusePath,
    dp_id_from_filename,
    muse_unit,
    read_cube,
    read_exposure_map,
    read_whitelight,
)
from udong.data.muse.dataset import default_root

DPID = "ADP.2024-04-30T18:20:44.624"


def test_muse_unit():
    assert muse_unit("10**(-20)erg.s**(-1).cm**(-2).angstrom**(-1)") == u.Unit(
        "1e-20 erg/(s cm2 AA)"
    )
    assert muse_unit("10**(-20)erg.s**(-1).cm**(-2).angstrom**(-1)").is_equivalent(
        u.erg / (u.s * u.cm**2 * u.AA)
    )
    assert muse_unit("10**(-40)erg**2.s**(-2).cm**(-4).angstrom**(-2)") == (
        u.Unit("1e-20 erg/(s cm2 AA)") ** 2
    )
    assert muse_unit("s") == u.s
    with pytest.raises(ValueError):
        muse_unit("bogus")


def test_dp_id_from_filename():
    assert dp_id_from_filename("MU_SCBD_..._WFM-AO-N_OBJ.fits") is None
    assert dp_id_from_filename(f"{DPID}.fits") == DPID
    assert dp_id_from_filename(f"some/dir/{DPID}.fits") == DPID


def test_locator(tmp_path):
    mp = MusePath(tmp_path)
    assert mp.local(DPID) == tmp_path / f"{DPID}.fits"
    assert mp.url(DPID) == f"https://dataportal.eso.org/dataPortal/file/{DPID}"
    with pytest.raises(ValueError):
        mp.local("ADP../evil/name")


def test_read_cube(fake_muse_file):
    cube = read_cube(fake_muse_file, dp_id=DPID)
    assert cube.shape == (40, 10, 12)
    assert cube.nwave == 40 and cube.ny == 10 and cube.nx == 12
    assert cube.flux.unit == u.Unit("1e-20 erg/(s cm2 AA)")
    # explicit wavelength vector: linear, air, 1.25 A
    w = cube.wavelength
    assert w is not None and len(w) == 40
    assert np.isclose(w[0].value, 4700.40576171875)
    assert np.isclose((w[1] - w[0]).value, 1.25)
    assert cube.wavelength_frame == "air"
    # variance -> IVAR; NaN pixels -> zero IVAR + masked
    assert np.isclose(cube.ivar[2, 3, 4], 1.0 / 25.0)
    assert cube.ivar[0, 0, 0] == 0.0
    assert cube.mask[0, 0, 0]
    assert not cube.mask[2, 3, 4]
    assert cube.mask[-1, 5, 5]
    # metadata
    assert cube.dp_id == DPID
    assert cube.target_name == "Ton S 180"
    assert cube.mode == "WFM-AO-N"
    assert cube.ncombine == 18
    assert cube.exptime == u.Quantity(3909.62, u.s)
    assert cube.sky_res == u.Quantity(1.1325, u.arcsec)
    assert cube.pipeline_version == "muse/2.8.6"
    # provenance
    assert cube.provenance is not None
    assert cube.provenance.product == "muse-cube"
    assert cube.provenance.survey == "MUSE"
    assert cube.provenance.steps[0].name == "read_muse_cube"
    # WCS
    assert cube.wcs is not None and cube.wcs.pixel_n_dim == 3
    assert cube.wcs.has_celestial and cube.wcs.has_spectral
    # spectrum extraction via core (bad pixel at wavelength 0 of column x=0,y=0)
    spec = cube.spectrum(x=0, y=0)
    assert len(spec) == 40
    assert not spec.bad[10]
    assert spec.bad[0]


def test_protocols(tmp_path):
    muse = MuseDataset(root=tmp_path, allow_download=False)
    assert isinstance(muse, IFUDataset)
    assert not isinstance(muse, CatalogDataset)
    # MaNGA provides both the cube interface and a summary catalog
    assert isinstance(MangaDataset(root=tmp_path, allow_download=False), IFUDataset)
    assert isinstance(MangaDataset(root=tmp_path, allow_download=False), CatalogDataset)


def test_find_target(fake_muse_file, tmp_path):
    # fake_muse_file already lives at tmp_path/<dp_id>.fits -> default layout
    muse = MuseDataset(root=tmp_path, allow_download=False)
    hits = muse.find_target("ton s 180")
    assert hits == [(DPID, fake_muse_file)]
    assert muse.find_target("no such galaxy") == []
    # alias in the header (OBJECT) is also matched
    assert muse.local_cubes()[0]["target"] == "Ton S 180"


def test_muse_path_url():
    assert MusePath("/tmp/muse").url(DPID) == (
        f"https://dataportal.eso.org/dataPortal/file/{DPID}"
    )
    from udong.data.muse.locator import valid_dp_id

    assert valid_dp_id(DPID)
    assert not valid_dp_id("ADP../x")


def test_discover_forwards_to_obscore(tmp_path, monkeypatch):
    import udong.catalogs.obscore as obscore
    from astropy.table import Table

    seen = {}
    def fake_search(**kw):
        seen.update(kw)
        return Table()

    monkeypatch.setattr(obscore, "search_muse_products", fake_search)
    muse = MuseDataset(root=tmp_path, allow_download=False)
    muse.discover(target="Ton S 180", ra=14.3343, dec=-22.3822, radius=0.5)
    assert seen["target"] == "Ton S 180"
    assert seen["ra"] == 14.3343
    assert seen["dataproduct_type"] == "cube"
    assert "instrument" not in seen  # search_muse_products hard-codes MUSE


def _make_ancillary(path, shape=(20, 24), bunit="10**(-20)erg.s**(-1).cm**(-2).angstrom**(-1)",
                    procatg="IMAGE_FOV_DEEP"):
    """Write a tiny MUSE-like ancillary image (PRIMARY + DATA 2-D)."""
    from astropy.io import fits

    import udong.data.muse.reader as reader_mod

    img = np.full(shape, 5.0, dtype=np.float32)
    img[0, 0] = np.nan
    prim = fits.PrimaryHDU()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prim.header["ESO OBS TARG NAME"] = "Ton S 180"
        prim.header["ESO INS MODE"] = "WFM-AO-N"
        prim.header["ESO PRO CATG"] = procatg
        prim.header["EXPTIME"] = 240.0
    hdu = fits.ImageHDU(img, name="DATA")
    hdu.header["BUNIT"] = bunit
    hdu.header["CRPIX1"] = shape[1] / 2
    hdu.header["CRPIX2"] = shape[0] / 2
    hdu.header["CRVAL1"] = 14.334183
    hdu.header["CRVAL2"] = -22.38238
    hdu.header["CD1_1"] = -5.55555555555556e-05
    hdu.header["CD2_2"] = 5.55555555555556e-05
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    fits.HDUList([prim, hdu]).writeto(path, overwrite=True)


def test_whitelight_and_expmap_readers(tmp_path):
    from astropy import units as u

    wl = tmp_path / "whitelight.fits"
    _make_ancillary(wl, shape=(20, 24), bunit="10**(-20)erg.s**(-1).cm**(-2).angstrom**(-1)")
    m = read_whitelight(wl)
    assert m.shape == (20, 24)
    assert m.value.unit == u.Unit("1e-20 erg/(s cm2 AA)")
    assert m.mask[0, 0] and not m.mask[5, 5]
    assert m.meta["kind"] == "whitelight"
    assert m.meta["TARGNAME"] == "Ton S 180"
    assert m.provenance.product == "muse-whitelight"
    assert m.wcs is not None and m.wcs.pixel_n_dim == 2

    ex = tmp_path / "expmap.fits"
    _make_ancillary(ex, shape=(19, 23), bunit="s", procatg="EXPOSURE_MAP")
    me = read_exposure_map(ex)
    assert me.shape == (19, 23)
    assert me.value.unit == u.s
    assert me.meta["kind"] == "exposure-map"
    assert me.provenance.product == "muse-exposure-map"


def test_dataset_ancillary_methods(fake_muse_file, tmp_path):
    wl = tmp_path / "ADP.2024-04-30T18:20:44.624.whitelight.fits"
    _make_ancillary(wl)
    muse = MuseDataset(root=tmp_path, allow_download=False)
    # explicit path
    m = muse.load_whitelight(DPID, path=wl)
    assert m.shape == (20, 24)
    # default sibling convention (file already at root/<dp_id>.whitelight.fits)
    m2 = muse.load_whitelight(DPID)
    assert m2.shape == (20, 24)
    # missing file with no path override
    with pytest.raises(FileNotFoundError):
        muse.load_exposure_map("ADP.2099-01-01T00:00:00.000")


def test_muse_dataset(fake_muse_file, tmp_path, monkeypatch):
    # explicit path, no download
    muse = MuseDataset(root=tmp_path, allow_download=False)
    cube = muse.load_cube(DPID, path=fake_muse_file, download=False)
    assert cube.dp_id == DPID
    assert cube.shape == (40, 10, 12)
    # default layout lookup fails when file absent and download disabled
    with pytest.raises(FileNotFoundError):
        MuseDataset(root=tmp_path, allow_download=False).load_cube(
            "ADP.2023-07-14T08:54:39.317", download=False
        )
    # default_root honouring the environment variable
    monkeypatch.setenv("UDONG_MUSE_ROOT", str(tmp_path / "muse_root"))
    assert default_root() == tmp_path / "muse_root"


class _FakeMuseArchive:
    """Fake easycat MUSEArchive recording fetch_one calls."""

    def __init__(self, *, write=None, data_none=False):
        self.write = write
        self.data_none = data_none
        self.calls = []

    def fetch_one(self, target, *, dest=None, download=False, **kwargs):
        from easycat.download import ItemResult

        self.calls.append({"target": target, "dest": dest, "download": download,
                           **kwargs})
        if not download or self.data_none:
            return ItemResult(obj_id=str(target), success=True, data=None,
                              meta={"url": f"https://dataportal.eso.org/dataPortal/file/{target}"})
        if self.write is not None:
            self.write(dest)
        return ItemResult(obj_id=str(target), success=True, data=dest,
                          meta={"url": "https://dataportal.eso.org/dataPortal/file/x",
                                "dest": str(dest)})


def test_load_cube_download_delegates_to_muse_archive(tmp_path, fake_muse_file, monkeypatch):
    import shutil

    root = tmp_path / "muse"
    fake = _FakeMuseArchive(write=lambda dest: (
        Path(dest).parent.mkdir(parents=True, exist_ok=True),
        shutil.copy(fake_muse_file, dest),
    ))
    seen = {}

    def factory(*, mode="cube", **kw):
        seen["mode"] = mode
        return fake

    monkeypatch.setattr("udong.data.muse.dataset.muse_archive", factory)
    cube = MuseDataset(root=root, allow_download=True).load_cube(DPID, download=True)
    assert cube.shape == (40, 10, 12)
    call = fake.calls[0]
    assert call["target"] == DPID and call["download"] is True
    assert call["validate"] == "fits"
    assert Path(call["dest"]) == root / f"{DPID}.fits"
    assert seen["mode"] == "cube"


def test_load_cube_missing_without_download_raises_url(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataPortal/file"):
        MuseDataset(root=tmp_path / "empty", allow_download=False).load_cube(
            DPID, download=False
        )


def test_load_whitelight_downloads_its_own_dp_id(tmp_path, monkeypatch):
    from astropy import units as u

    wl_dp_id = "ADP.2024-04-30T18:20:44.625"
    fake = _FakeMuseArchive(write=lambda dest: _make_ancillary(dest))
    seen = {}

    def factory(*, mode="cube", **kw):
        seen["mode"] = mode
        return fake

    monkeypatch.setattr("udong.data.muse.dataset.muse_archive", factory)
    m = MuseDataset(root=tmp_path / "muse", allow_download=True).load_whitelight(
        wl_dp_id, download=True
    )
    assert m.shape == (20, 24) and m.value.unit == u.Unit("1e-20 erg/(s cm2 AA)")
    assert seen["mode"] == "whitelight"
    assert Path(fake.calls[0]["dest"]) == tmp_path / "muse" / f"{wl_dp_id}.whitelight.fits"


def test_load_missing_product_raises_archive_url(tmp_path, monkeypatch):
    fake = _FakeMuseArchive(data_none=True)
    monkeypatch.setattr("udong.data.muse.dataset.muse_archive",
                        lambda **kw: fake)
    with pytest.raises(FileNotFoundError, match="dataPortal/file"):
        MuseDataset(root=tmp_path / "muse", allow_download=True).load_cube(DPID)
