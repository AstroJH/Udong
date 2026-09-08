"""0/1/2 order line-profile moments and moment maps (spectroscopy).

These are *spectral measurements*: M0 (line flux), M1 (intensity-weighted
mean velocity) and M2 (intensity-weighted velocity dispersion) of a line
profile.  M1/M2 are kinematic descriptors that downstream
``udong.science.kinematics`` consumes as 2-D maps (see the package
docstring for the boundary).

Definitions (per spectrum / spaxel, after continuum subtraction):

.. math::

    M_0 &= \\sum_i w_i \\\\
    M_1 &= \\frac{\\sum_i w_i\\, v_i}{M_0} \\\\
    M_2 &= \\sqrt{\\frac{\\sum_i w_i\\,(v_i - M_1)^2}{M_0}}

with channel weights :math:`w_i = F_i\\,\\Delta\\lambda_i` (flux per spectral
pixel) and Doppler velocities :math:`v_i = c\\,(\\lambda_i-\\lambda_0)/\\lambda_0`
relative to the line rest wavelength :math:`\\lambda_0`.

Physical meaning
----------------
* M0 : integrated line flux (the "0th moment" map);
* M1 : intensity-weighted mean line-of-sight velocity (velocity field);
* M2 : intensity-weighted velocity dispersion.

For a pure Gaussian line profile (continuum subtracted) M0 equals the line
flux, M1 the Gaussian centroid and M2 the Gaussian sigma; comparing moment
maps with Gaussian fits (e.g. MaNGA DAP ``EMLINE_G*``) is a useful
cross-check.  Blending, noise and continuum subtraction bias the moments, so
interpretation should be explicit.

The classic higher-order description continues with Gauss-Hermite h3/h4
(skewness/kurtosis); the DAP's pPXF and emission-line fits provide those in
specialised outputs, not as raw moments.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.constants import c as _c
from astropy.units import Quantity

from udong.core.cube import Cube
from udong.core.map import Map2D

__all__ = [
    "velocity_axis",
    "line_moments",
    "subtract_linear_continuum",
    "moment_maps",
]

_C_KMPS = _c.to(u.km / u.s).value


def velocity_axis(wavelength: Quantity, rest_wavelength: Quantity) -> Quantity:
    """Doppler velocities ``c * (lambda - lambda0) / lambda0`` in km/s."""
    wave = Quantity(wavelength).to(u.AA).value
    lam0 = Quantity(rest_wavelength).to(u.AA).value
    return Quantity(_C_KMPS * (wave - lam0) / lam0, u.km / u.s)


def line_moments(
    wavelength: Quantity,
    flux: Quantity,
    rest_wavelength: Quantity,
    valid=None,
) -> tuple[Quantity, Quantity, Quantity]:
    """M0 / M1 / M2 of a single spectrum over the supplied wavelength range.

    Parameters
    ----------
    wavelength : Quantity
        1-D wavelength array.
    flux : Quantity
        1-D flux-density array (same length).
    rest_wavelength : Quantity
        Rest wavelength of the line (defines v = 0).
    valid : bool array, optional
        Good-pixel flag (e.g. from the cube ivar/mask); defaults to finite.

    Returns
    -------
    (M0, M1, M2) with M0 in flux units (flux-density x wavelength), M1 and M2
    in km/s.  NaN when there are too few valid pixels.
    """
    wave = Quantity(wavelength).to(u.AA).value
    f = np.asarray(flux.value if isinstance(flux, Quantity) else flux, dtype=float)
    if len(wave) != len(f):
        raise ValueError("wavelength and flux must have the same length")
    good = np.isfinite(f)
    if valid is not None:
        good &= np.asarray(valid, dtype=bool)
    if good.sum() < 3:
        unit_flux = flux.unit if isinstance(flux, Quantity) else u.dimensionless_unscaled
        return (
            Quantity(np.nan, unit_flux * u.AA),
            Quantity(np.nan, u.km / u.s),
            Quantity(np.nan, u.km / u.s),
        )
    # trapezoid channel widths (weight per pixel, zero at edges via clipping)
    dlam = np.zeros_like(wave)
    dlam[1:-1] = 0.5 * (wave[2:] - wave[:-2])
    dlam[0] = wave[1] - wave[0]
    dlam[-1] = wave[-1] - wave[-2]
    w = f * dlam
    w = np.where(good, w, 0.0)
    v = _C_KMPS * (wave - Quantity(rest_wavelength).to(u.AA).value) / Quantity(rest_wavelength).to(u.AA).value

    m0 = float(np.sum(w))
    if m0 <= 0 or not np.isfinite(m0):
        unit_flux = flux.unit if isinstance(flux, Quantity) else u.dimensionless_unscaled
        return (
            Quantity(m0, unit_flux * u.AA),
            Quantity(np.nan, u.km / u.s),
            Quantity(np.nan, u.km / u.s),
        )
    m1 = float(np.sum(w * v) / m0)
    m2 = float(np.sqrt(max(np.sum(w * (v - m1) ** 2) / m0, 0.0)))
    unit_flux = flux.unit if isinstance(flux, Quantity) else u.dimensionless_unscaled
    return Quantity(m0, unit_flux * u.AA), Quantity(m1, u.km / u.s), Quantity(m2, u.km / u.s)


def subtract_linear_continuum(
    wavelength: Quantity,
    flux: Quantity,
    blue: tuple[Quantity, Quantity],
    red: tuple[Quantity, Quantity],
    wlo: Quantity | None = None,
    whi: Quantity | None = None,
) -> np.ndarray:
    """Subtract a linear continuum anchored on the median of two line-free bands.

    The continuum is a straight line through ``(median lambda, median flux)``
    of the blue and red anchor bands; it is evaluated and subtracted over
    ``[wlo, whi]`` (or the whole array when no window is given).
    """
    wave = Quantity(wavelength).to(u.AA).value
    f = np.asarray(flux.value if isinstance(flux, Quantity) else flux, dtype=float)
    sel_b = (wave >= Quantity(blue[0]).to(u.AA).value) & (wave <= Quantity(blue[1]).to(u.AA).value)
    sel_r = (wave >= Quantity(red[0]).to(u.AA).value) & (wave <= Quantity(red[1]).to(u.AA).value)
    if sel_b.sum() < 2 or sel_r.sum() < 2:
        raise ValueError("continuum anchor bands contain too few pixels")
    xb, xr = float(np.median(wave[sel_b])), float(np.median(wave[sel_r]))
    yb, yr = float(np.median(f[sel_b])), float(np.median(f[sel_r]))
    slope = (yr - yb) / (xr - xb)
    cont = yb + slope * (wave - xb)
    if wlo is not None or whi is not None:
        lo = wave[0] if wlo is None else Quantity(wlo).to(u.AA).value
        hi = wave[-1] if whi is None else Quantity(whi).to(u.AA).value
        m = (wave >= lo) & (wave <= hi)
        return f[m] - cont[m]
    return f - cont


def moment_maps(
    cube: Cube,
    wlo: Quantity,
    whi: Quantity,
    rest_wavelength: Quantity,
    blue: tuple[Quantity, Quantity] | None = None,
    red: tuple[Quantity, Quantity] | None = None,
    min_valid: int = 5,
) -> tuple[Map2D, Map2D, Map2D]:
    """Compute M0/M1/M2 maps from a datacube over ``[wlo, whi]``.

    ``blue``/``red`` are optional line-free anchor bands for linear continuum
    subtraction (recommended).  Returns (M0, M1, M2) as ``Map2D`` with units
    of flux, km/s and km/s; spaxels with too few valid pixels are masked.
    """
    if cube.wavelength is None:
        raise ValueError("cube needs a wavelength vector")
    wave_all = Quantity(cube.wavelength).to(u.AA).value
    ivar = cube.ivar
    cube_mask = cube.mask

    lo = Quantity(wlo).to(u.AA).value
    hi = Quantity(whi).to(u.AA).value
    win = (wave_all >= lo) & (wave_all <= hi)
    idx = np.nonzero(win)[0]
    if len(idx) < min_valid:
        raise ValueError("moment window contains too few spectral pixels")

    def good_pix(x: int, y: int) -> np.ndarray:
        g = np.isfinite(np.asarray(cube.flux.value[:, y, x]))
        if ivar is not None:
            g &= np.asarray(ivar[:, y, x]) > 0
        if cube_mask is not None:
            if cube.mask_defs is not None:
                g &= ~cube.mask_defs.unmask(cube_mask[:, y, x])
            else:
                g &= cube_mask[:, y, x] == 0
        return g

    ny, nx = cube.ny, cube.nx
    m0 = np.full((ny, nx), np.nan)
    m1 = np.full((ny, nx), np.nan)
    m2 = np.full((ny, nx), np.nan)
    bad = np.zeros((ny, nx), dtype=bool)

    wave_win = Quantity(wave_all[win], u.AA)
    rest = Quantity(rest_wavelength)
    for y in range(ny):
        for x in range(nx):
            fl = Quantity(np.asarray(cube.flux.value[:, y, x]), cube.flux.unit)
            good = good_pix(x, y)
            if good.sum() < min_valid or not good[win].any():
                bad[y, x] = True
                continue
            if blue is not None and red is not None:
                try:
                    sub = Quantity(
                        subtract_linear_continuum(
                            Quantity(wave_all, u.AA), fl, blue, red, wlo, whi
                        ),
                        cube.flux.unit,
                    )
                except ValueError:
                    bad[y, x] = True
                    continue
            else:
                sub = fl[win]
            gwin = good[win]
            a, vv, ss = line_moments(wave_win, sub, rest, valid=gwin)
            if np.isfinite(a.value) and np.isfinite(vv.value):
                m0[y, x], m1[y, x], m2[y, x] = a.value, vv.value, ss.value
            else:
                bad[y, x] = True

    flux_unit = cube.flux.unit * u.AA
    wcs = cube.spatial_wcs
    meta = dict(cube.meta)
    m0_map = Map2D(value=Quantity(m0, flux_unit), mask=bad, wcs=wcs, meta=dict(meta, quantity="M0"))
    v_map = Map2D(value=Quantity(m1, u.km / u.s), mask=bad, wcs=wcs, meta=dict(meta, quantity="M1"))
    s_map = Map2D(value=Quantity(m2, u.km / u.s), mask=bad, wcs=wcs, meta=dict(meta, quantity="M2"))
    return m0_map, v_map, s_map
