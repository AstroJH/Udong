"""Shared fixtures: synthetic MaNGA-like FITS files and helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

os.environ.setdefault("MPLBACKEND", "Agg")


def wcs_cards(header, nx=8, ny=8, naxis=2, crval1=232.5447, crval2=48.690201,
              crval3=4000.0, cd3=1.0):
    """Write standard MaNGA-style TAN WCS keywords into a header."""
    header["WCSAXES"] = naxis
    header["CRPIX1"] = (nx + 1) / 2
    header["CRPIX2"] = (ny + 1) / 2
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    header["CD1_1"] = -0.000138889  # ~0.5 arcsec/pix
    header["CD2_2"] = 0.000138889
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["RADESYS"] = "FK5"
    header["EQUINOX"] = 2000.0
    if naxis >= 3:
        header["CRPIX3"] = 1.0
        header["CRVAL3"] = crval3
        header["CD3_3"] = cd3
        header["CTYPE3"] = "WAVE-LOG"
        header["CUNIT3"] = "Angstrom"
    return header


def make_cube(path, nwave=50, ny=8, nx=8):
    """Write a small MaNGA-like LOGCUBE file."""
    rng = np.random.default_rng(42)
    wave = 10 ** np.linspace(np.log10(4000.0), np.log10(4400.0), nwave)
    flux = np.abs(rng.normal(loc=10.0, scale=1.0, size=(nwave, ny, nx))).astype(np.float32)
    ivar = np.full((nwave, ny, nx), 100.0, dtype=np.float32)
    mask = np.zeros((nwave, ny, nx), dtype=np.int64)
    mask[0, 0, 0] = 1 << 10  # DONOTUSE (DRP3PIXMASK)
    prim = fits.PrimaryHDU()
    wcs_cards(prim.header, nx=nx, ny=ny, naxis=3)
    prim.header["BUNIT"] = "1E-17 erg/s/cm^2/ang/spaxel"
    prim.header["MASKNAME"] = "MANGA_DRP3PIXMASK"
    prim.header["PLATEIFU"] = "8485-1901"
    prim.header["MANGAID"] = "1-209232"
    prim.header["VERSDRP3"] = "v3_1_1"
    prim.header["VERSDAP"] = "3.1.0"
    hdul = fits.HDUList(
        [
            prim,
            fits.ImageHDU(flux, name="FLUX"),
            fits.ImageHDU(ivar, name="IVAR"),
            fits.ImageHDU(mask, name="MASK"),
            fits.ImageHDU(flux * 0.1 + 1.0, name="LSFPRE"),
            fits.ImageHDU(flux * 0.1 + 1.0, name="LSFPOST"),
            fits.ImageHDU(wave, name="WAVE"),
        ]
    )
    hdul.writeto(path, overwrite=True)


def make_maps(path, ny=8, nx=8, lines=("Hb-4862", "OIII-5008", "Ha-6564")):
    """Write a small MaNGA-like DAP MAPS file."""
    rng = np.random.default_rng(1)
    prim = fits.PrimaryHDU()
    h = prim.header
    h["DAPTYPE"] = "HYB10-MILESHC-MASTARSSP"
    h["PLATEIFU"] = "8485-1901"
    h["MANGAID"] = "1-209232"
    h["VERSDRP3"] = "v3_1_1"
    h["VERSDAP"] = "3.1.0"
    h["DRP3QUAL"] = 0
    h["DAPQUAL"] = 0
    h["MASKNAME"] = "MANGA_DAPPIXMASK"
    hdus = [prim]

    vel = rng.normal(size=(ny, nx)) * 50.0
    vel_mask = np.zeros((ny, nx), dtype=np.int32)
    vel_mask[0, 0] = 1 << 30  # DONOTUSE

    def triple(base, data, bunit, mask=None, nchan=None):
        hdu = fits.ImageHDU(data, name=base)
        wcs_cards(hdu.header, nx=nx, ny=ny, naxis=2 if nchan is None else 3)
        hdu.header["BUNIT"] = bunit
        ih = fits.ImageHDU(np.full_like(data, 0.01), name=base + "_IVAR")
        wcs_cards(ih.header, nx=nx, ny=ny, naxis=2 if nchan is None else 3)
        ih.header["BUNIT"] = f"({bunit})^-2"
        mh = fits.ImageHDU(mask if mask is not None else np.zeros_like(data, dtype=np.int32),
                           name=base + "_MASK")
        wcs_cards(mh.header, nx=nx, ny=ny, naxis=2 if nchan is None else 3)
        return hdu, ih, mh

    for hdu in triple("STELLAR_VEL", vel, "km/s", mask=vel_mask):
        hdus.append(hdu)

    # 3-channel emission-line flux extension
    elflux = np.abs(rng.normal(size=(len(lines), ny, nx))) * 10.0
    elmask = np.zeros((len(lines), ny, nx), dtype=np.int32)
    elmask[1, 2, 3] = 1 << 30
    eh = fits.ImageHDU(elflux, name="EMLINE_GFLUX")
    wcs_cards(eh.header, nx=nx, ny=ny, naxis=3)
    eh.header["BUNIT"] = "1E-17 erg/s/cm^2/spaxel"
    for i, name in enumerate(lines, start=1):
        eh.header[f"C{i:02d}"] = name
    eih = fits.ImageHDU(np.full_like(elflux, 0.01), name="EMLINE_GFLUX_IVAR")
    wcs_cards(eih.header, nx=nx, ny=ny, naxis=3)
    eih.header["BUNIT"] = "(1E-17 erg/s/cm^2/spaxel)^-2"
    emh = fits.ImageHDU(elmask, name="EMLINE_GFLUX_MASK")
    wcs_cards(emh.header, nx=nx, ny=ny, naxis=3)
    hdus += [eh, eih, emh]

    # SPX_ELLCOO (4 channels)
    ell = np.zeros((4, ny, nx))
    ell[0] = np.hypot(*np.indices((ny, nx)) - (nx - 1) / 2) * 0.5
    ell[1] = ell[0] / 5.0
    ell[3] = 30.0
    e0 = fits.ImageHDU(ell, name="SPX_ELLCOO")
    wcs_cards(e0.header, nx=nx, ny=ny, naxis=3)
    e0.header["BUNIT"] = "arcsec"
    for i, (nm, un) in enumerate(
        [("Elliptical radius", "arcsec"), ("R/Re", ""), ("R h/kpc", "kpc/h"), ("Elliptical azimuth", "degrees")],
        start=1,
    ):
        e0.header[f"C{i}"] = nm
        e0.header[f"U{i}"] = un
    hdus.append(e0)

    fits.HDUList(hdus).writeto(path, overwrite=True)


def make_drpall(path):
    """Write a small DRPall-like catalog (MaNGA cubes in HDU 1)."""
    t = Table(
        {
            "MANGAID": ["1-209232", "1-114262"],
            "PLATEIFU": ["8485-1901", "7443-12703"],
            "Z": [0.04, 0.03],
            "DRP3QUAL": [0, 0],
        }
    )
    fits.HDUList([fits.PrimaryHDU(), fits.table_to_hdu(t)]).writeto(path, overwrite=True)


@pytest.fixture
def fake_cube_file(tmp_path):
    p = tmp_path / "manga-8485-1901-LOGCUBE.fits"
    make_cube(p)
    return p


@pytest.fixture
def fake_maps_file(tmp_path):
    p = tmp_path / "manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits"
    make_maps(p)
    return p


@pytest.fixture
def fake_drpall_file(tmp_path):
    p = tmp_path / "drpall-v3_1_1.fits"
    make_drpall(p)
    return p


def pytest_addoption(parser):
    parser.addoption(
        "--run-realdata",
        action="store_true",
        default=False,
        help="run tests that require real downloaded MaNGA files",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-realdata"):
        skip = pytest.mark.skip(reason="real MaNGA data not requested (use --run-realdata)")
        for item in items:
            if "realdata" in item.keywords:
                item.add_marker(skip)
