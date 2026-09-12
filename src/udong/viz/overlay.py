"""Overlay one observation on another (different instruments / WCS).

The common astronomy task this solves: draw an IFU quantity (e.g. a
velocity/moment-1 map) -- as **contours** or as a **semi-transparent filled
map** -- on top of an image from a different facility (e.g. an optical cutout
from DESI Legacy Survey, HST, ...).

Two alignment modes:

* **Contours** are WCS-aware without any reprojection: contour vertices are
  computed in the overlay map's pixel grid and mapped to the display through
  the overlay WCS (``ax.get_transform(overlay_wcs)``).  Works for arbitrary
  pixel scales, rotations and offsets between the two images.
* **Filled maps** must be resampled onto the base image grid; this module uses
  :func:`reproject.reproject_interp` when the two WCS differ (a mature
  astropy-affiliated dependency, already required by Udong).  When the overlay
  has no WCS it is assumed to share the base grid.

Example
-------
>>> from udong.viz import plot_map, overlay_contours   # doctest: +SKIP
>>> ax = plot_map(optical_map)                         # base image (DESI cutout)
>>> overlay_contours(velocity_map, ax, symmetric=True, colorbar=True)  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from udong.core.map import Map2D
from udong.viz.scales import resolve_norm

__all__ = ["contour_levels", "overlay_contours", "overlay_map", "plot_overlay", "zoom_to"]


# --------------------------------------------------------------------------- #
def _finite(map2d: Map2D) -> np.ndarray:
    """Finite, unmasked values of a Map2D (1-D, for level statistics)."""
    data = np.ma.asarray(map2d.masked_value()).compressed()
    data = np.asarray(data, dtype=float)
    return data[np.isfinite(data)]


def contour_levels(
    map2d: Map2D,
    nlevels: int = 7,
    symmetric: bool = False,
    percentiles: tuple[float, float] = (2.0, 98.0),
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Choose sensible contour levels for a Map2D.

    Parameters
    ----------
    nlevels
        Number of levels (>= 2).
    symmetric
        Force levels symmetric about zero (typical for velocity maps).
    percentiles
        Percentile range used when ``vmin``/``vmax`` are not given.
    """
    if nlevels < 2:
        raise ValueError("nlevels must be >= 2")
    vals = _finite(map2d)
    if vals.size == 0:
        raise ValueError("map has no finite values to contour")
    if vmin is None:
        vmin = float(np.percentile(vals, percentiles[0]))
    if vmax is None:
        vmax = float(np.percentile(vals, percentiles[1]))
    if symmetric:
        v = max(abs(vmin), abs(vmax))
        vmin, vmax = -v, v
    if not vmax > vmin:
        raise ValueError(f"degenerate contour range: vmin={vmin}, vmax={vmax}")
    return np.linspace(vmin, vmax, nlevels)


def _wcs_transform(map2d: Map2D, ax: Axes):
    """Transform mapping overlay pixel coords -> display (or ``None``)."""
    if map2d.wcs is None or not isinstance(ax, Axes):
        return None
    if getattr(ax, "wcs", None) is None:
        return None
    try:
        return ax.get_transform(map2d.wcs)
    except Exception:
        return None


def overlay_contours(
    map2d: Map2D,
    ax: Axes,
    levels: np.ndarray | list[float] | None = None,
    nlevels: int = 7,
    symmetric: bool = False,
    percentiles: tuple[float, float] = (2.0, 98.0),
    cmap: str = "coolwarm",
    linewidths: float | list[float] = 1.2,
    alpha: float = 0.9,
    labels: bool = False,
    fmt: str = "%g",
    colorbar: bool = False,
    colorbar_label: str | None = None,
    **contour_kwargs: Any,
) -> Axes:
    """Draw WCS-aligned contours of ``map2d`` onto an existing axes ``ax``.

    ``ax`` is typically produced by :func:`udong.viz.plots.plot_map` for the
    base image, so it is a WCSAxes carrying the base image WCS.  When either
    WCS is missing, pixel grids are assumed identical.
    """
    if levels is None:
        levels = contour_levels(map2d, nlevels=nlevels, symmetric=symmetric,
                                percentiles=percentiles)
    levels = np.asarray(levels, dtype=float)

    data = np.asarray(map2d.masked_value(), dtype=float)
    if map2d.bad is not None:
        data = np.where(np.asarray(map2d.bad), np.nan, data)
    ny, nx = data.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    kwargs: dict[str, Any] = dict(linewidths=linewidths, alpha=alpha)
    if "colors" not in contour_kwargs:
        kwargs["cmap"] = cmap
        kwargs["norm"] = Normalize(vmin=float(levels.min()), vmax=float(levels.max()))
    kwargs.update(contour_kwargs)

    transform = _wcs_transform(map2d, ax)

    cs = ax.contour(x, y, data, levels=levels, **kwargs)
    if transform is not None:
        # matplotlib does not propagate ``transform`` from contour() to the
        # resulting collection(s); apply it explicitly so contours from a
        # different WCS land at the correct sky positions.
        colls = getattr(cs, "collections", None)
        if colls:
            for coll in colls:
                coll.set_transform(transform)
        else:
            cs.set_transform(transform)
    if labels:
        ax.clabel(cs, fmt=fmt, inline=True, fontsize=8)
    if colorbar:
        cb = ax.figure.colorbar(cs, ax=ax)
        cb.set_label(colorbar_label or f"{map2d.unit:latex_inline}")
    return ax


def _base_shape(ax: Axes) -> tuple[int, int]:
    if getattr(ax, "images", None):
        return tuple(np.asarray(ax.images[0].get_array()).shape[:2])
    wcs = getattr(ax, "wcs", None)
    if wcs is not None and getattr(wcs, "array_shape", None) is not None:
        return tuple(wcs.array_shape)
    raise ValueError(
        "cannot infer the base image shape; draw the base image first "
        "(e.g. with udong.viz.plot_map) or pass an axes that already has one"
    )


def zoom_to(map2d: Map2D, ax: Axes, margin: float = 0.15) -> Axes:
    """Zoom ``ax`` so that the footprint of ``map2d`` fills the view.

    Works across WCS: the overlay footprint corners are converted to the base
    axes pixel coordinates through the two WCS.  ``margin`` adds that fraction
    of padding around the footprint (e.g. 0.15 = 15 per cent).

    Useful when a small IFU map is drawn on a much larger optical image.
    """
    ny, nx = map2d.shape
    xs = np.array([-0.5, nx - 0.5, -0.5, nx - 0.5], dtype=float)
    ys = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5], dtype=float)

    base_wcs = getattr(ax, "wcs", None)
    if map2d.wcs is not None and base_wcs is not None:
        world = map2d.wcs.pixel_to_world(xs, ys)
        bx, by = base_wcs.world_to_pixel(world)
    else:
        bx, by = xs, ys

    dx = float(np.ptp(bx)) * margin
    dy = float(np.ptp(by)) * margin
    ax.set_xlim(float(np.min(bx)) - dx, float(np.max(bx)) + dx)
    ax.set_ylim(float(np.min(by)) - dy, float(np.max(by)) + dy)
    return ax


def overlay_map(
    map2d: Map2D,
    ax: Axes,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    alpha: float = 0.6,
    scale: str = "linear",
    linthresh: float | None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    zoom: bool = False,
    **kwargs: Any,
) -> Axes:
    """Draw a semi-transparent filled ``map2d`` on an existing axes ``ax``.

    When the overlay WCS differs from the base axes WCS the map is resampled
    onto the base grid with :func:`reproject.reproject_interp`; pixels outside
    the overlay footprint are left transparent.

    Parameters
    ----------
    vmin, vmax
        Colour limits (percentile-based when omitted).
    symmetric
        Force the colour range symmetric about zero (``vmin=-vmax``), the
        standard choice for velocity maps with a diverging colormap.
    scale, linthresh
        Colour scale ``"linear"`` / ``"log"`` / ``"symlog"`` (see
        :func:`udong.viz.scales.resolve_norm`).
    colorbar, colorbar_label
        Draw a colourbar for the filled overlay (label defaults to its unit).
    zoom
        Zoom the view to the overlay footprint after drawing.
    """
    data = np.asarray(map2d.masked_value(), dtype=float)
    if map2d.bad is not None:
        data = np.where(np.asarray(map2d.bad), np.nan, data)

    base_wcs = getattr(ax, "wcs", None)
    if map2d.wcs is not None and base_wcs is not None:
        from reproject import reproject_interp

        shape = _base_shape(ax)
        data, footprint = reproject_interp(
            (data, map2d.wcs), base_wcs, shape_out=shape, return_footprint=True
        )
        data = np.where((footprint > 0) & np.isfinite(data), data, np.nan)
    else:
        shape = _base_shape(ax)
        if data.shape != shape:
            hint = (
                "the base axes has no WCS, so the overlay cannot be reprojected; "
                "create it with projection=wcs (e.g. fig.add_subplot(projection=wcs)) "
                "to enable cross-WCS overlays"
                if map2d.wcs is not None and base_wcs is None
                else "provide a WCS on both images to enable reprojection"
            )
            raise ValueError(
                f"overlay shape {data.shape} does not match base image {shape}; {hint}"
            )

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("map has no finite values to overlay")

    norm = kwargs.pop("norm", None)
    if norm is None:
        norm = resolve_norm(data, scale=scale, vmin=vmin, vmax=vmax,
                            linthresh=linthresh, symmetric=symmetric)

    # transparent colour for pixels outside the (reprojected) footprint
    cmap_obj = plt.get_cmap(cmap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))

    im = ax.imshow(data, origin="lower", cmap=cmap_obj, alpha=alpha,
                   norm=norm, **kwargs)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax)
        cb.set_label(colorbar_label or f"{map2d.unit:latex_inline}")
    if zoom:
        zoom_to(map2d, ax)
    return ax


def plot_overlay(
    base: Map2D,
    overlay: Map2D,
    ax: Axes | None = None,
    filled: bool = False,
    contours: bool = True,
    zoom: bool = False,
    base_kwargs: Mapping[str, Any] | None = None,
    fill_kwargs: Mapping[str, Any] | None = None,
    contour_kwargs: Mapping[str, Any] | None = None,
) -> Axes:
    """Plot ``base`` (e.g. an optical image) with ``overlay`` on top.

    Parameters
    ----------
    filled, contours
        Which overlay representation(s) to draw.
    base_kwargs, fill_kwargs, contour_kwargs
        Forwarded to :func:`udong.viz.plot_map`, :func:`overlay_map` and
        :func:`overlay_contours` respectively.
    """
    from udong.viz.plots import plot_map

    if ax is None:
        ax = plot_map(base, **dict(base_kwargs or {}))
    elif not getattr(ax, "images", None):
        # caller supplied an empty axes: draw the base image first
        plot_map(base, ax=ax, **dict(base_kwargs or {}))
    if filled:
        overlay_map(overlay, ax, **dict(fill_kwargs or {}))
    if contours:
        overlay_contours(overlay, ax, **dict(contour_kwargs or {}))
    if zoom:
        zoom_to(overlay, ax)
    return ax
