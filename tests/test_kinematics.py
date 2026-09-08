import numpy as np
import pytest
from astropy import units as u
from astropy.wcs import WCS

from udong.core.map import Map2D
from udong.science.kinematics import (
    fit_velocity_plane,
    global_dispersion,
    global_mean_velocity,
    lsf_correct_sigma,
    radial_dispersion_profile,
    radial_velocity_profile,
)


def make_vel_map(ny=16, nx=16, center=(7.5, 7.5)):
    """Velocity field with a linear gradient along x: v = v0 + g*(x-cx)."""
    yy, xx = np.indices((ny, nx), dtype=float)
    cx, cy = center
    v = 50.0 + 15.0 * (xx - cx)  # km/s
    ivar = np.full((ny, nx), 1.0) / (u.km / u.s) ** 2
    m = Map2D(value=v * u.km / u.s, uncertainty=ivar, mask=np.zeros((ny, nx), dtype=bool))
    return m


def make_wcs():
    hdr = {
        "CRPIX1": 7.5, "CRPIX2": 7.5, "CRVAL1": 0.0, "CRVAL2": 0.0,
        "CD1_1": -0.000138889, "CD2_2": 0.000138889,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
    }
    w = WCS(hdr)
    w.wcs.set()
    return w


def test_lsf_correct_sigma():
    obs = Map2D(value=np.full((4, 4), 100.0) * u.km / u.s,
                uncertainty=np.full((4, 4), 1.0) / (u.km / u.s) ** 2)
    corr = lsf_correct_sigma(obs, 60.0 * u.km / u.s)
    assert np.isclose(corr.value.value[0, 0], np.sqrt(100**2 - 60**2))
    # where obs < lsf -> flagged bad
    obs2 = Map2D(value=np.full((4, 4), 50.0) * u.km / u.s,
                 uncertainty=np.full((4, 4), 1.0) / (u.km / u.s) ** 2)
    corr2 = lsf_correct_sigma(obs2, 60.0 * u.km / u.s)
    assert corr2.bad.all()


def test_radial_profiles():
    v = make_vel_map()
    prof = radial_velocity_profile(v, bins=4, statistic="median")
    assert len(prof.radius) == 4
    s = Map2D(value=np.full((16, 16), 80.0) * u.km / u.s,
              uncertainty=np.full((16, 16), 1.0) / (u.km / u.s) ** 2,
              mask=np.zeros((16, 16), dtype=bool))
    sprof = radial_dispersion_profile(s, bins=2, statistic="median")
    assert np.allclose(sprof.value.value, 80.0)


def test_global_mean_velocity():
    v = make_vel_map()
    mean, err = global_mean_velocity(v)
    assert np.isclose(mean.value, 50.0, atol=0.5)
    assert np.isclose(err.value, 1.0 / 16.0)  # mean ivar = 256 -> sigma = 1/16


def test_global_dispersion_ivar():
    s = Map2D(value=np.full((16, 16), 80.0) * u.km / u.s,
              uncertainty=np.full((16, 16), 100.0) / (u.km / u.s) ** 2,
              mask=np.zeros((16, 16), dtype=bool))
    mean, err = global_dispersion(s)
    assert np.isclose(mean.value, 80.0)
    assert np.isclose(err.value, 1.0 / np.sqrt(100.0 * 256.0))


def test_fit_velocity_plane():
    v = make_vel_map()
    v = Map2D(value=v.value, uncertainty=v.uncertainty, mask=v.mask, wcs=make_wcs())
    fit = fit_velocity_plane(v)
    assert np.isclose(fit.v0.value, 50.0, atol=1e-6)
    assert np.isclose(fit.a.value, 15.0, atol=1e-6)
    assert np.isclose(fit.b.value, 0.0, atol=1e-6)
    assert np.isclose(fit.residual_rms.value, 0.0, atol=1e-6)
    assert fit.gradient_magnitude.value == pytest.approx(15.0)
    # gradient along +x -> PA of ~90 deg (east is x) for this simple WCS
    assert np.isclose(fit.gradient_pa.to_value(u.deg) % 180, 90.0, atol=5.0)
