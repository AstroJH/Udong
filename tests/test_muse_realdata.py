"""Optional golden-file checks against the real Ton S 180 WFM cube.

Skipped when the file is not present locally.  Light-weight on purpose:
headers + a single-pixel spectrum slice (memmap), never a full-array load.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

GOLDEN = Path.home() / "Repository/muse/ADP.2024-04-30T18:20:44.624.fits"

pytestmark = pytest.mark.skipif(
    not GOLDEN.exists(), reason=f"golden MUSE cube not present: {GOLDEN}"
)


def test_golden_structure_and_headers():
    with fits.open(GOLDEN, memmap=True) as hdul:
        names = [h.name for h in hdul]
        assert names == ["PRIMARY", "DATA", "STAT"]
        d, s = hdul["DATA"], hdul["STAT"]
        # ESO FITS files are big-endian float32 (dtype '>f4')
        assert np.issubdtype(d.data.dtype, np.float32)
        assert np.issubdtype(s.data.dtype, np.float32)
        assert d.data.shape == s.data.shape == (3722, 318, 328)
        assert d.header["BUNIT"].startswith("10**(-20)")
        assert s.header["BUNIT"].startswith("10**(-40)")
        assert d.header["CTYPE3"] == "AWAV"
        assert np.isclose(d.header["CD3_3"], 1.25)
        assert np.isclose(abs(d.header["CD1_1"]) * 3600, 0.2)
        assert d.header["CRVAL3"] > 4000
        assert "SKY_RES" in hdul[0].header
        assert hdul[0].header["ESO INS MODE"].startswith("WFM")
        assert hdul[0].header["NCOMBINE"] == 18


def test_golden_spectrum_slice():
    with fits.open(GOLDEN, memmap=True) as hdul:
        flux = hdul["DATA"].data[:, 159, 164]  # central-ish spaxel
        var = hdul["STAT"].data[:, 159, 164]
        ok = np.isfinite(flux) & np.isfinite(var) & (var > 0)
        assert ok.sum() > 3000
        assert np.all(var[ok] > 0)

WL = Path.home() / "Repository/muse/ADP.2024-04-30T18:20:44.625.whitelight.fits"
EX = Path.home() / "Repository/muse/ADP.2024-04-30T18:20:44.628.expmap.fits"

pytestmark_real = pytest.mark.skipif(
    not (WL.exists() and EX.exists()),
    reason="golden MUSE ancillary files not present",
)


@pytest.mark.skipif(not WL.exists(), reason="whitelight file not present")
def test_golden_whitelight():
    from udong.data.muse import read_whitelight

    wl = read_whitelight(WL)
    assert wl.shape == (318, 328)
    assert wl.meta["kind"] == "whitelight"
    assert wl.provenance.product == "muse-whitelight"
    # same WCS as the cube (verified in golden-file notes)
    with fits.open(GOLDEN, memmap=True) as hdul:
        cube_hdr = hdul["DATA"].header
    assert abs(wl.wcs.wcs.crval[0] - cube_hdr["CRVAL1"]) < 1e-9
    assert abs(wl.wcs.wcs.crval[1] - cube_hdr["CRVAL2"]) < 1e-9
    assert wl.value.unit == u.Unit("1e-20 erg/(s cm2 AA)")


@pytest.mark.skipif(not EX.exists(), reason="expmap file not present")
def test_golden_exposure_map():
    from udong.data.muse import read_exposure_map

    ex = read_exposure_map(EX)
    assert ex.shape == (317, 327)  # one pixel smaller than the cube grid
    assert str(ex.value.unit) == "s"
    assert np.nanmax(ex.value.to_value("s")) > 4000  # ~18 x 240 s
