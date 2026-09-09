"""Base visualization functions: spectrum, map, profile."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
from astropy import units as u
from astropy.visualization.wcsaxes import WCSAxes

from udong.core.map import Map2D
from udong.core.spectrum import Spectrum
from udong.science.spatial import RadialProfile

__all__ = ["plot_spectrum", "plot_map", "plot_profile"]


def plot_spectrum(spectrum: Spectrum, ax=None, show_mask: bool = True, **kwargs):
    """Plot flux vs wavelength; masked pixels are overlaid in light gray."""

    if ax is None:
        _, ax = plt.subplots()
    wave = spectrum.spectral_axis.to_value("AA")
    flux = spectrum.flux.value
    ax.plot(wave, flux, lw=0.8, **kwargs)
    if show_mask and spectrum.bad is not None:
        bad = np.asarray(spectrum.bad)
        ax.plot(wave[bad], flux[bad], ".", color="0.7", ms=3, label="masked")
    ax.set_xlabel(f"wavelength [{spectrum.spectral_axis.unit}]")
    ax.set_ylabel(f"flux [{spectrum.flux.unit:latex_inline}]")
    if kwargs.get("label"):
        ax.legend()
    return ax




def _manual_sky_ticks(ax, wcs, ny: int, nx: int) -> None:
    """Label pixel ticks of a plain axes with WCS sky coordinates.

    Used when ``plot_map`` is given a *non-WCS* ``ax`` (e.g. from a plain
    ``plt.subplots`` grid): the pixels still form the data coordinates, but
    the visible tick labels are converted to RA/Dec so the figure reads as a
    sky map instead of raw pixel indices 0..N.
    """
    xt = np.asarray([t for t in ax.get_xticks() if -0.5 <= float(t) <= nx - 0.5])
    yt = np.asarray([t for t in ax.get_yticks() if -0.5 <= float(t) <= ny - 0.5])
    if xt.size:
        coords = wcs.pixel_to_world(xt, np.full(xt.size, ny / 2.0))
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{c.ra.degree:.3f}" for c in coords])
    if yt.size:
        coords = wcs.pixel_to_world(np.full(yt.size, nx / 2.0), yt)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{c.dec.degree:.3f}" for c in coords])


def _finite_values(data):
    """Positive/negative finite unmasked values for norm limits."""
    a = np.ma.asarray(data)
    return a.compressed()


def _log_norm(data, vmin, vmax):
    vals = np.asarray(_finite_values(data), dtype=float)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size == 0:
        raise ValueError(
            "scale='log' needs positive data; pass vmin>0 or use scale='linear'/'symlog'"
        )
    lo = float(vmin) if vmin is not None and vmin > 0 else float(pos.min())
    hi = float(vmax) if vmax is not None else float(pos.max())
    return LogNorm(vmin=lo, vmax=hi)


def _symlog_norm(data, vmin, vmax, linthresh):
    vals = np.asarray(_finite_values(data), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite data for scale='symlog'")
    lo = float(vmin) if vmin is not None else float(vals.min())
    hi = float(vmax) if vmax is not None else float(vals.max())
    if linthresh is None:
        linthresh = max(abs(lo), abs(hi)) / 100.0 or 1.0
    return SymLogNorm(linthresh=float(linthresh), vmin=lo, vmax=hi)


def plot_map(map2d: Map2D, ax=None, colorbar: bool = True, show_mask: bool = True,
             vmin=None, vmax=None, cmap: str = "viridis", scale: str = "linear",
             linthresh=None, **kwargs):
    """Plot a Map2D (WCSAxes when a WCS is available).

    Parameters
    ----------
    scale
        Colour-scale of the colourbar: ``"linear"`` (default), ``"log"``
        (requires positive data unless ``vmin`` is given) or ``"symlog"``
        (for bipolar maps; ``linthresh`` sets the linear threshold).
    linthresh
        Linear threshold for ``scale="symlog"`` (defaults to a hundredth of
        the data range).
    vmin, vmax, cmap
        Passed to the image normalisation (linear scale).
    """
    if scale not in ("linear", "log", "symlog"):
        raise ValueError(f"unknown scale {scale!r}; use 'linear', 'log' or 'symlog'")
    if "norm" in kwargs:
        if scale != "linear":
            raise ValueError("pass either scale= or norm=, not both")
    data = np.asarray(map2d.masked_value())
    if show_mask:
        data = np.ma.masked_where(map2d.bad, data)

    norm = kwargs.pop("norm", None)
    if scale == "log":
        if norm is None:
            norm = _log_norm(data, vmin, vmax)
    elif scale == "symlog":
        if norm is None:
            norm = _symlog_norm(data, vmin, vmax, linthresh)

    if ax is None:
        if map2d.wcs is not None:
            fig = plt.figure()
            ax = fig.add_subplot(projection=map2d.wcs)
        else:
            fig, ax = plt.subplots()

    if map2d.wcs is not None:
        im = ax.imshow(
            data, origin="lower", cmap=cmap, norm=norm,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax, **kwargs)
        if isinstance(ax, WCSAxes):
            # proper WCS axes: astropy draws RA/Dec world ticks itself
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
        else:
            # plain matplotlib axes: convert pixel ticks to sky coordinates
            ny, nx = map2d.shape
            _manual_sky_ticks(ax, map2d.wcs, ny, nx)
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("Dec [deg]")
    else:
        im = ax.imshow(
            data, origin="lower", cmap=cmap, norm=norm,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax, **kwargs)
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax)
        cb.set_label(f"{map2d.unit:latex_inline}")
    return ax


def plot_profile(profile: RadialProfile, ax=None, **kwargs):
    """Plot a radial profile with error bars (if available)."""

    if ax is None:
        _, ax = plt.subplots()
    # radius may be arcsec, kpc or dimensionless (R/Re); keep its own unit
    r = np.asarray(profile.radius.value)
    v = np.asarray(profile.value.value)
    if profile.uncertainty is not None:
        e = np.asarray(profile.uncertainty.to_value(profile.value.unit))
        ax.errorbar(r, v, yerr=e, fmt="o", **kwargs)
    else:
        ax.plot(r, v, "o", **kwargs)
    if profile.radius.unit == u.dimensionless_unscaled:
        ax.set_xlabel("R / Re")
    else:
        ax.set_xlabel(f"radius [{profile.radius.unit:latex_inline}]")
    ax.set_ylabel(f"{profile.value.unit:latex_inline}")
    return ax
