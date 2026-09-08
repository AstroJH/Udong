import numpy as np
from astropy import units as u
from astropy.units import Quantity

from udong.core.cube import Cube
from udong.science.spectroscopy.moments import (
    line_moments,
    moment_maps,
    subtract_linear_continuum,
    velocity_axis,
)

HA = 6564.6 * u.AA


def gaussian_line(wave, area, v, sigma, rest=HA):
    """Flux-density Gaussian with total area `area` (a Quantity); returns
    a Quantity in erg/s/cm2/AA."""
    lam0 = rest.to_value(u.AA) * (1.0 + v / 2.998e5)
    sig_lam = sigma / 2.998e5 * rest.to_value(u.AA)
    norm = area.value / (sig_lam * np.sqrt(2 * np.pi))
    unit = area.unit / u.AA
    return Quantity(norm * np.exp(-0.5 * ((wave.to_value(u.AA) - lam0) / sig_lam) ** 2), unit)


def test_velocity_axis():
    wave = HA * (1 + np.array([0.0, 100.0, -100.0]) / 2.998e5)
    v = velocity_axis(wave, HA)
    assert np.allclose(v.value, [0.0, 100.0, -100.0], atol=0.1)


def test_line_moments_gaussian():
    wave = np.arange(6500, 6630, 0.5) * u.AA
    area = 3.0 * 1e-17 * u.erg / (u.s * u.cm**2)
    v_true, sig_true = 80.0, 120.0
    flux = gaussian_line(wave, area, v_true, sig_true)
    m0, v, s = line_moments(wave, flux, HA)
    assert np.isclose(m0.to_value(u.erg / (u.s * u.cm**2)), 3e-17, rtol=2e-3)
    assert np.isclose(v.value, v_true, atol=1.0)
    assert np.isclose(s.value, sig_true, atol=3.0)


def test_subtract_linear_continuum():
    wave = np.arange(6500, 6630, 0.5) * u.AA
    cont = (1.0 + 0.05 * (wave.to_value(u.AA) - 6560) / 100.0) * 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    area = 2.0 * 1e-17 * u.erg / (u.s * u.cm**2)
    flux = cont + gaussian_line(wave, area, 0.0, 150.0)
    sub = subtract_linear_continuum(wave, flux, (6520 * u.AA, 6540 * u.AA), (6590 * u.AA, 6610 * u.AA),
                                    wlo=6550 * u.AA, whi=6580 * u.AA)
    # residual integrated over dlambda recovers the injected line area
    dlam = 0.5  # AA
    assert np.isclose(np.sum(sub) * dlam, area.to_value(u.erg / (u.s * u.cm**2)), rtol=1e-2)
    # far wings of the window are continuum-free (near zero)
    wsub = wave[(wave >= 6550 * u.AA) & (wave <= 6580 * u.AA)]
    m = wsub.to_value(u.AA) <= 6554
    assert np.abs(sub[m]).max() < 5e-19


def _synthetic_cube(ny=6, nx=6):
    wave = np.arange(6450, 6680, 0.5) * u.AA
    nw = len(wave)
    flux = np.zeros((nw, ny, nx))
    ivar = np.ones((nw, ny, nx)) * 1e6
    for y in range(ny):
        for x in range(nx):
            v = 60.0 * (x - (nx - 1) / 2)
            sig = 100.0 + 20.0 * y
            area = Quantity(5.0, 1e-17 * u.erg / (u.s * u.cm**2))
            flux[:, y, x] = gaussian_line(wave, area, v, sig).value
    flux *= u.erg / (u.s * u.cm**2 * u.AA)
    return Cube(flux=flux, ivar=ivar, wavelength=wave)


def test_moment_maps_synthetic():
    cube = _synthetic_cube()
    m0, v, s = moment_maps(cube, 6550 * u.AA, 6580 * u.AA, HA)
    assert m0.shape == (6, 6)
    # M0 ~ area
    assert np.allclose(m0.value.value, 5.0, rtol=5e-3)
    # velocity gradient along x
    for x in range(6):
        expected = 60.0 * (x - 2.5)
        assert np.isclose(v.value.value[0, x], expected, atol=8.0)
    # sigma increases with y
    assert s.value.value[0, 0] < s.value.value[-1, 0]
    assert np.isclose(s.value.value[0, 0], 100.0, atol=8.0)
    assert np.isclose(s.value.value[-1, 0], 200.0, atol=12.0)
