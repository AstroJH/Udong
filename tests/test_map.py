import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from udong.core.map import Map2D


def make_wcs(nx=8, ny=8):
    hdr = {
        "WCSAXES": 2,
        "CRPIX1": (nx + 1) / 2,
        "CRPIX2": (ny + 1) / 2,
        "CRVAL1": 232.5447,
        "CRVAL2": 48.690201,
        "CD1_1": -0.000138889,
        "CD2_2": 0.000138889,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }
    w = WCS(hdr)
    w.wcs.set()
    return w


def test_map2d_basic():
    m = Map2D(value=np.ones((8, 8)) * 10.0 * u.km / u.s,
              uncertainty=np.full((8, 8), 1.0 / 0.01) / (u.km / u.s) ** 2,
              mask=np.zeros((8, 8), dtype=np.int32))
    assert m.value.unit == u.km / u.s
    assert m.shape == (8, 8)
    assert np.allclose(m.sigma.value, 0.1)
    assert not m.bad.any()
    assert np.allclose(m.masked_value(), 10.0)


def test_bad_masking():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[3, 3] = 7
    m = Map2D(value=np.ones((8, 8)) * u.km / u.s, mask=mask)
    assert m.bad[3, 3]
    mv = m.masked_value()
    assert np.isnan(mv[3, 3])


def test_wcs_roundtrip():
    w = make_wcs()
    m = Map2D(value=np.ones((8, 8)) * u.km / u.s, wcs=w)
    coord = m.pixel_to_sky(4.0, 4.0)
    assert isinstance(coord, SkyCoord)
    x, y = m.sky_to_pixel(coord)
    assert np.allclose([x, y], [4.0, 4.0])
