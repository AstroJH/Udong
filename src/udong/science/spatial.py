"""Spatial analysis: radial/elliptical profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from udong.core.coordinates import radial_map, wcs_center_pixel
from udong.core.map import Map2D
from udong.core.uncertainty import ivar_to_sigma, propagate_weighted_mean

__all__ = ["RadialProfile", "radial_profile", "binned_profile"]


@dataclass
class RadialProfile:
    """1-D radial profile of a Map2D quantity."""

    radius: Quantity  # bin centers, arcsec
    value: Quantity
    uncertainty: Quantity | None = None
    n: np.ndarray | None = None  # number of good pixels per bin
    statistic: str = "median"
    radius_edges: Quantity | None = None

    @property
    def unit(self):
        return self.value.unit


def _bins_edges(rmax: Quantity, bins: int | Quantity) -> Quantity:
    if isinstance(bins, Quantity):
        return Quantity(bins)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive int or a Quantity of edges")
    
    return Quantity(
        np.linspace(0.0, rmax.to(u.arcsec).value, bins + 1),
        u.arcsec
    )


def binned_profile(
    value_map: Map2D,
    radius_map: Map2D,
    bins: int | Quantity,
    statistic: Literal["median", "mean", "weighted_mean"] = "median",
) -> RadialProfile:
    """Profile of ``value_map`` binned by ``radius_map`` (both same shape)."""
    if value_map.shape != radius_map.shape:
        raise ValueError("value_map and radius_map must have the same shape")
    
    r = np.asarray(radius_map.value.value)
    v = np.asarray(value_map.value.value)

    bad = value_map.bad | radius_map.bad
    rmax = Quantity(
        np.nanmax( np.where(bad, np.nan, r) ),
        radius_map.value.unit
    )

    edges = _bins_edges(rmax, bins).to_value(u.arcsec)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)
    out = np.full(n_bins, np.nan)
    out_unc = np.full(n_bins, np.nan)
    out_n = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        sel = (r >= edges[i]) & (r < edges[i + 1]) & ~bad
        if i == n_bins - 1:
            sel |= (np.isclose(r, edges[i + 1])) & ~bad
        out_n[i] = int(np.count_nonzero(sel))
        if out_n[i] == 0:
            continue
        vals = v[sel]

        if statistic == "median":
            out[i] = np.nanmedian(vals)

        elif statistic == "mean":
            out[i] = np.nanmean(vals)

        elif statistic == "weighted_mean":
            iv = np.asarray(value_map.ivar.value)[sel] if value_map.ivar is not None else None
            if iv is None:
                raise ValueError("weighted_mean requires value_map.uncertainty/ivar")

            mean, mivar = propagate_weighted_mean(vals, iv)
            out[i] = float(mean)
            out_unc[i] = float(ivar_to_sigma(mivar)) if mivar > 0 else np.nan

        else:
            raise ValueError(f"unknown statistic {statistic!r}")
        
    return RadialProfile(
        radius=Quantity(centers, u.arcsec),
        value=Quantity(out, value_map.unit),
        uncertainty=None if statistic != "weighted_mean" else Quantity(out_unc, value_map.unit),
        n=out_n,
        statistic=statistic,
        radius_edges=Quantity(edges, u.arcsec),
    )


def radial_profile(
    map2d: Map2D,
    center: tuple[float, float] | str = "auto",
    bins: int | Quantity = 20,
    statistic: Literal["median", "mean", "weighted_mean"] = "median",
    scale: Quantity = 0.5 * u.arcsec / u.pixel,
) -> RadialProfile:
    """Circular radial profile of ``map2d``.

    ``center`` is pixel ``(x, y)`` or ``"auto"`` (uses the WCS reference
    pixel, or the array center when no WCS is available).
    """
    _center: tuple[float, float]
    if center == "auto":
        if map2d.wcs is not None:
            _center = wcs_center_pixel(map2d.wcs)
        else:
            ny, nx = map2d.shape
            _center = ((nx - 1) / 2, (ny - 1) / 2)
    else:
        if isinstance(center, str):
            raise ValueError(f"invalid center {center!r}")
        _center = center

    radius = Map2D(
        value=radial_map(map2d.shape, center=_center, scale=scale, wcs=map2d.wcs),
        mask=map2d.bad,
        wcs=map2d.wcs,
        meta={"quantity": "radius"},
    )

    return binned_profile(map2d, radius, bins=bins, statistic=statistic)
