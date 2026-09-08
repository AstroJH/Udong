import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from udong.core.cube import Cube


def make_cube(nwave=10, ny=4, nx=4):
    wave = np.linspace(4000, 4100, nwave) * u.AA
    flux = np.ones((nwave, ny, nx)) * 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    ivar = np.full((nwave, ny, nx), 1e34)
    mask = np.zeros((nwave, ny, nx), dtype=np.int64)
    mask[:, 1, 1] = 1
    return Cube(flux=flux, ivar=ivar, mask=mask, wavelength=wave)


def test_cube_basic():
    c = make_cube()
    assert c.shape == (10, 4, 4)
    assert c.nwave == 10 and c.ny == 4 and c.nx == 4
    assert c.wavelength is not None and len(c.wavelength) == 10


def test_spectrum_extraction():
    c = make_cube()
    s = c.spectrum(2, 2)
    assert len(s) == 10
    assert s.flux.unit == c.flux.unit
    # masked spaxel at (1,1)
    s_bad = c.spectrum(1, 1)
    assert s_bad.mask is not None and s_bad.mask.all()
    with pytest.raises(IndexError):
        c.spectrum(99, 0)


def test_spectrum_at():
    c = make_cube()
    hdr = {
        "CRPIX1": 2.5, "CRPIX2": 2.5, "CRVAL1": 232.5447, "CRVAL2": 48.690201,
        "CD1_1": -0.000138889, "CD2_2": 0.000138889,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
    }
    w = WCS(hdr)
    w.wcs.set()
    c2 = Cube(flux=c.flux, ivar=c.ivar, mask=c.mask, wavelength=c.wavelength, wcs=w)
    coord = SkyCoord(232.5447 * u.deg, 48.690201 * u.deg)
    s = c2.spectrum_at(coord)
    assert len(s) == 10
