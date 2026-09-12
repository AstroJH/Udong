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
    block: int = 50_000,
) -> tuple[Map2D, Map2D, Map2D]:
    """Compute M0/M1/M2 maps from a datacube over ``[wlo, whi]``.

    The computation is **vectorised over spaxels in blocks** (``block``
    spaxels per chunk): the window channels, trapezoid weights and velocity
    axis are precomputed once, and the linear continuum (``blue``/``red``
    anchor bands) is evaluated for all spaxels of a block at once.  This
    replaces the previous per-spaxel Python loop, which processed the full
    wavelength axis for every spaxel.

    Parameters
    ----------
    blue, red
        Optional line-free anchor bands for linear continuum subtraction
        (recommended); each is ``(lo, hi)`` in observed wavelength.
    min_valid
        Minimum number of valid (finite flux, ``ivar > 0``, unmasked) pixels
        **inside the moment window** required for a spaxel to be measured.
    block
        Number of spaxels per vectorised chunk (memory control; lower it for
        very large windows/cubes).

    Returns
    -------
    (M0, M1, M2)
        ``Map2D`` with units of flux, km/s and km/s; spaxels with too few
        valid pixels are masked.
    """
    if cube.wavelength is None:
        raise ValueError("cube needs a wavelength vector")
    if block < 1:
        raise ValueError("block must be >= 1")

    wave_all = Quantity(cube.wavelength).to(u.AA).value
    flux2 = np.asarray(cube.flux.value, dtype=float).reshape(wave_all.size, -1)
    ivar = cube.ivar
    cube_mask = cube.mask
    ivar2 = None if ivar is None else np.asarray(ivar, dtype=float).reshape(wave_all.size, -1)
    mask2 = None if cube_mask is None else np.asarray(cube_mask).reshape(wave_all.size, -1)

    lo = Quantity(wlo).to(u.AA).value
    hi = Quantity(whi).to(u.AA).value
    win_idx = np.nonzero((wave_all >= lo) & (wave_all <= hi))[0]
    if win_idx.size < min_valid:
        raise ValueError("moment window contains too few spectral pixels")
    wave_win = wave_all[win_idx]

    # trapezoid channel widths and fixed velocity axis (identical per spaxel)
    dlam = np.zeros_like(wave_win)
    if wave_win.size > 1:
        dlam[1:-1] = 0.5 * (wave_win[2:] - wave_win[:-2])
        dlam[0] = wave_win[1] - wave_win[0]
        dlam[-1] = wave_win[-1] - wave_win[-2]
    rest_aa = float(Quantity(rest_wavelength).to_value(u.AA))
    v_axis = _C_KMPS * (wave_win - rest_aa) / rest_aa

    have_anchors = blue is not None and red is not None
    if have_anchors:
        b_lo, b_hi = (Quantity(b).to_value(u.AA) for b in blue)
        r_lo, r_hi = (Quantity(r).to_value(u.AA) for r in red)
        bidx = np.nonzero((wave_all >= b_lo) & (wave_all <= b_hi))[0]
        ridx = np.nonzero((wave_all >= r_lo) & (wave_all <= r_hi))[0]
        if bidx.size < 2 or ridx.size < 2:
            raise ValueError("continuum anchor bands contain too few pixels")
        xb = float(np.median(wave_all[bidx]))
        xr = float(np.median(wave_all[ridx]))
        if xr == xb:
            raise ValueError("continuum anchor bands overlap")

    def _good(f, v, m):
        g = np.isfinite(f)
        if v is not None:
            g &= v > 0
        if m is not None:
            if cube.mask_defs is not None:
                g &= ~cube.mask_defs.unmask(m)
            else:
                g &= m == 0
        return g

    nspax = flux2.shape[1]
    m0 = np.full(nspax, np.nan)
    m1 = np.full(nspax, np.nan)
    m2 = np.full(nspax, np.nan)

    for start in range(0, nspax, block):
        sl = slice(start, min(start + block, nspax))
        fw = flux2[win_idx, sl]
        gw = _good(fw, None if ivar2 is None else ivar2[win_idx, sl],
                   None if mask2 is None else mask2[win_idx, sl])

        if have_anchors:
            fb, fr = flux2[bidx, sl], flux2[ridx, sl]
            gb = _good(fb, None if ivar2 is None else ivar2[bidx, sl],
                       None if mask2 is None else mask2[bidx, sl])
            gr = _good(fr, None if ivar2 is None else ivar2[ridx, sl],
                       None if mask2 is None else mask2[ridx, sl])
            with np.errstate(all="ignore"):
                yb = np.ma.median(np.ma.masked_where(~gb, fb), axis=0).filled(np.nan)
                yr = np.ma.median(np.ma.masked_where(~gr, fr), axis=0).filled(np.nan)
            slope = (yr - yb) / (xr - xb)
            sub = fw - (yb + slope * (wave_win[:, None] - xb))
        else:
            sub = fw

        weights = np.where(gw, sub * dlam[:, None], 0.0)
        with np.errstate(all="ignore"):
            m0b = np.sum(weights, axis=0)
            m1b = np.sum(weights * v_axis[:, None], axis=0) / m0b
            m2b = np.sqrt(np.maximum(
                np.sum(weights * (v_axis[:, None] - m1b) ** 2, axis=0) / m0b, 0.0
            ))
        nvalid = gw.sum(axis=0)
        ok = (nvalid >= min_valid) & np.isfinite(m0b) & (m0b > 0) & np.isfinite(m1b)
        m0[start:sl.stop] = np.where(ok, m0b, np.nan)
        m1[start:sl.stop] = np.where(ok, m1b, np.nan)
        m2[start:sl.stop] = np.where(ok, m2b, np.nan)

    ny, nx = cube.ny, cube.nx
    m0 = m0.reshape(ny, nx)
    m1 = m1.reshape(ny, nx)
    m2 = m2.reshape(ny, nx)
    bad = ~np.isfinite(m1)

    flux_unit = cube.flux.unit * u.AA
    wcs = cube.spatial_wcs
    meta = dict(cube.meta)
    m0_map = Map2D(value=Quantity(m0, flux_unit), mask=bad, wcs=wcs, meta=dict(meta, quantity="M0"))
    v_map = Map2D(value=Quantity(m1, u.km / u.s), mask=bad, wcs=wcs, meta=dict(meta, quantity="M1"))
    s_map = Map2D(value=Quantity(m2, u.km / u.s), mask=bad, wcs=wcs, meta=dict(meta, quantity="M2"))
    return m0_map, v_map, s_map
