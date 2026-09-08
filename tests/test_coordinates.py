import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from udong.core.coordinates import (
    elliptical_radius_map,
    pixel_scale,
    radial_map,
    wcs_center_pixel,
)


def make_wcs():
    hdr = {
        "CRPIX1": 4.0, "CRPIX2": 4.0, "CRVAL1": 0.0, "CRVAL2": 0.0,
        "CD1_1": -0.000138889, "CD2_2": 0.000138889,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
    }
    w = WCS(hdr)
    w.wcs.set()
    return w


def test_pixel_scale():
    scale = pixel_scale(make_wcs())
    assert np.isclose(scale.to_value(u.arcsec / u.pixel), 0.5, atol=1e-3)


def test_wcs_center_pixel():
    cx, cy = wcs_center_pixel(make_wcs())
    assert (cx, cy) == (4.0, 4.0)


def test_radial_map():
    r = radial_map((8, 8), center=(3.5, 3.5), scale=0.5 * u.arcsec / u.pixel)
    assert r.shape == (8, 8)
    assert r.unit == u.arcsec
    assert np.isclose(r[3, 3].value, 0.5 * np.hypot(0.5, 0.5))
    assert np.isclose(r[0, 0].value, 0.5 * np.hypot(3.5, 3.5))


def test_elliptical_radius_map():
    r = elliptical_radius_map((8, 8), center=(3.5, 3.5), scale=0.5 * u.arcsec / u.pixel,
                              ellipticity=0.0)
    assert r.shape == (8, 8)
    # with flattening along the east-west axis, north (row 0) sits at larger radius
    re = elliptical_radius_map((8, 8), center=(3.5, 3.5), scale=0.5 * u.arcsec / u.pixel,
                               ellipticity=0.5, position_angle=0 * u.deg)
    assert re[0, 3].value > re[3, 7].value
    # circular case must match radial_map
    rc = elliptical_radius_map((8, 8), center=(3.5, 3.5), scale=0.5 * u.arcsec / u.pixel,
                               ellipticity=0.0)
    assert np.allclose(rc.value, radial_map((8, 8), center=(3.5, 3.5),
                                            scale=0.5 * u.arcsec / u.pixel).value)
