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


def _anchored_cube(nwave=120, ny=3, nx=4):
    """Small cube with a continuum + Gaussian line and line-free anchors."""
    from astropy import units as u

    from udong.core.cube import Cube

    wave = np.linspace(6500.0, 6620.0, nwave) * u.AA
    unit = u.Unit("1e-17 erg/(s cm2 AA)")
    flux = np.zeros((nwave, ny, nx))
    for y in range(ny):
        for x in range(nx):
            line = gaussian_line(wave, 5.0 * 1e-17 * u.erg / (u.s * u.cm**2),
                                 v=60.0 * (x - 1.5), sigma=90.0 + 20.0 * y)
            flux[:, y, x] = line.to_value(unit) + 2.0  # unit = 1e-17 erg/(s cm2 AA)
    return Cube(
        flux=flux * unit,
        ivar=np.full((nwave, ny, nx), 100.0),
        wavelength=wave,
    )


def test_moment_maps_matches_per_spaxel_reference():
    """The vectorised implementation must match the old per-spaxel maths."""
    from astropy import units as u

    cube = _anchored_cube()
    wave = cube.wavelength
    wlo, whi = 6540 * u.AA, 6590 * u.AA
    blue = (6500 * u.AA, 6525 * u.AA)
    red = (6600 * u.AA, 6620 * u.AA)

    m0, m1, m2 = moment_maps(cube, wlo, whi, HA, blue=blue, red=red)

    win = (wave >= wlo) & (wave <= whi)
    ref0 = np.full((cube.ny, cube.nx), np.nan)
    ref1 = np.full_like(ref0, np.nan)
    ref2 = np.full_like(ref0, np.nan)
    for y in range(cube.ny):
        for x in range(cube.nx):
            fl = Quantity(cube.flux.value[:, y, x], cube.flux.unit)
            sub = Quantity(
                subtract_linear_continuum(wave, fl, blue, red, wlo, whi),
                cube.flux.unit,
            )
            a, v, s = line_moments(wave[win], sub, HA, valid=np.ones(win.sum(), bool))
            ref0[y, x], ref1[y, x], ref2[y, x] = a.value, v.value, s.value

    assert np.allclose(m0.value.value, ref0, rtol=1e-6, atol=1e-9)
    assert np.allclose(m1.value.value, ref1, rtol=1e-6, atol=1e-6)
    assert np.allclose(m2.value.value, ref2, rtol=1e-6, atol=1e-6)


def test_moment_maps_marks_invalid_spaxels():
    """NaN / ivar<=0 / masked spaxels inside the window become bad."""
    from astropy import units as u

    cube = _anchored_cube(nwave=60, ny=2, nx=3)
    wave = cube.wavelength
    wlo, whi = 6550 * u.AA, 6580 * u.AA
    win = (wave >= wlo) & (wave <= whi)

    flux = cube.flux.value.copy()
    flux[win, 0, 0] = np.nan                    # not finite
    ivar = cube.ivar.copy()
    ivar[win, 0, 1] = 0.0                       # invalid inverse variance
    mask = np.zeros(flux.shape, dtype=np.int32)
    mask[win, 0, 2] = 1                         # masked bit

    dirty = Cube(flux=flux * cube.flux.unit, ivar=ivar, mask=mask,
                 wavelength=wave)
    _, m1, _ = moment_maps(dirty, wlo, whi, HA)
    assert m1.bad[0, 0] and m1.bad[0, 1] and m1.bad[0, 2]
    assert not m1.bad[1, 0]
