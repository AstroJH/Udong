"""Spatial analysis: radial / elliptical (deprojected) profiles.

* ``radial_profile``: 1-D profile in *projected* circular radius.
* ``deprojected_radius_map`` / ``radius_over_re``: thin-disk *deprojected*
  (galactocentric, in-plane) radius ``R_cyl`` and its ``R/R_e`` normalisation.
* ``binned_profile``: bin any ``Map2D`` by an arbitrary radius map
  (arcsec, kpc, or dimensionless ``R/R_e``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from udong.core.coordinates import (
    elliptical_radius_map,
    pixel_scale,
    radial_map,
    wcs_center_pixel,
)
from udong.core.map import Map2D
from udong.core.uncertainty import ivar_to_sigma, propagate_weighted_mean

__all__ = [
    "RadialProfile",
    "binned_profile",
    "deprojected_radius_map",
    "radial_profile",
    "radius_over_re",
]


@dataclass
class RadialProfile:
    """1-D radial profile of a Map2D quantity.

    ``radius`` follows the unit of whatever radius map was used (arcsec,
    kpc, or dimensionless ``R/R_e``).
    """

    radius: Quantity  # bin centres
    value: Quantity
    uncertainty: Quantity | None = None
    n: np.ndarray | None = None  # number of good pixels per bin
    statistic: str = "median"
    radius_edges: Quantity | None = None

    @property
    def unit(self):
        return self.value.unit


def _resolve_center(map2d: Map2D, center: tuple[float, float] | str) -> tuple[float, float]:
    """Pixel centre ``(x, y)`` for a Map2D (``"auto"`` from the WCS)."""
    if center == "auto":
        if map2d.wcs is not None:
            return wcs_center_pixel(map2d.wcs)
        ny, nx = map2d.shape
        return (nx - 1) / 2, (ny - 1) / 2
    if isinstance(center, str):
        raise ValueError(f"invalid center {center!r}")
    return float(center[0]), float(center[1])


def _radius_scale(map2d: Map2D, scale: Quantity | None) -> Quantity:
    """Pixel scale: explicit ``scale``, else the WCS plate scale, else 0.5 arcsec/pix."""
    if scale is not None:
        return Quantity(scale)
    if map2d.wcs is not None:
        return pixel_scale(map2d.wcs)
    return 0.5 * u.arcsec / u.pixel


def _bins_edges(rmax: Quantity, bins: int | Quantity) -> Quantity:
    """Bin edges in the unit of ``rmax``.

    ``bins`` may be a positive int (uniform bins from 0 to ``rmax``), an
    astropy ``Quantity`` of edges, or a plain array of edge values (which are
    interpreted in the unit of ``rmax``).
    """
    if isinstance(bins, Quantity):
        return Quantity(bins).to(rmax.unit)
    if isinstance(bins, bool):
        raise ValueError("bins must be a positive int or an array of edges")
    if isinstance(bins, (int, np.integer)):
        if bins < 1:
            raise ValueError("bins must be a positive int or an array of edges")
        return Quantity(np.linspace(0.0, float(rmax.value), int(bins) + 1), rmax.unit)
    arr = np.asarray(bins, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("bins must be a positive int or an array of edges")
    return Quantity(arr, rmax.unit)


def binned_profile(
    value_map: Map2D,
    radius_map: Map2D,
    bins: int | Quantity,
    statistic: Literal["median", "mean", "weighted_mean"] = "median",
) -> RadialProfile:
    """Profile of ``value_map`` binned by ``radius_map`` (same shape).

    ``radius_map`` may carry any linear unit (arcsec, kpc, ...) or be
    dimensionless (e.g. ``R/R_e``); the returned profile radius uses that
    unit.
    """
    if value_map.shape != radius_map.shape:
        raise ValueError("value_map and radius_map must have the same shape")

    r_unit = radius_map.value.unit
    r = np.asarray(radius_map.value.value)
    v = np.asarray(value_map.value.value)

    bad = value_map.bad | radius_map.bad
    rmax_val = float(np.nanmax(np.where(bad, np.nan, r)))
    edges = _bins_edges(Quantity(rmax_val, r_unit), bins)
    e = np.asarray(edges.value)
    centers = Quantity(0.5 * (e[:-1] + e[1:]), r_unit)
    n_bins = len(centers)
    out = np.full(n_bins, np.nan)
    out_unc = np.full(n_bins, np.nan)
    out_n = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        sel = (r >= e[i]) & (r < e[i + 1]) & ~bad
        if i == n_bins - 1:
            sel |= (np.isclose(r, e[i + 1])) & ~bad
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
        radius=centers,
        value=Quantity(out, value_map.unit),
        uncertainty=None if statistic != "weighted_mean" else Quantity(out_unc, value_map.unit),
        n=out_n,
        statistic=statistic,
        radius_edges=edges,
    )


def radial_profile(
    map2d: Map2D,
    center: tuple[float, float] | str = "auto",
    bins: int | Quantity = 20,
    statistic: Literal["median", "mean", "weighted_mean"] = "median",
    scale: Quantity = 0.5 * u.arcsec / u.pixel,
) -> RadialProfile:
    """Circular (*projected*) radial profile of ``map2d``.

    ``center`` is pixel ``(x, y)`` or ``"auto"`` (WCS reference pixel, or the
    array centre when no WCS is available).  Radius unit is set by ``scale``
    (default arcsec); pass a physical scale, e.g.
    ``0.5 * arcsec * kpc_per_arcsec / pixel``, to work in kpc directly.
    """
    _center = _resolve_center(map2d, center)
    radius = Map2D(
        value=radial_map(map2d.shape, center=_center, scale=scale, wcs=map2d.wcs),
        mask=map2d.bad,
        wcs=map2d.wcs,
        meta={"quantity": "projected radius"},
    )
    return binned_profile(map2d, radius, bins=bins, statistic=statistic)


def deprojected_radius_map(
    map2d: Map2D,
    *,
    center: tuple[float, float] | str = "auto",
    position_angle: Quantity,
    inclination: Quantity | None = None,
    axis_ratio: float | None = None,
    scale: Quantity | None = None,
) -> Map2D:
    """Deprojected, in-plane (galactocentric) radius map of a thin disk.

    Model / assumptions
    -------------------
    The observed galaxy is an *infinitely thin, axisymmetric disk* inclined by
    ``inclination`` about its major axis.  Choosing sky coordinates along the
    major axis (``x'``, PA = ``position_angle``) and minor axis (``y'``), a
    disk point at in-plane radius ``R`` and azimuth ``phi`` projects to

        x' = R cos(phi),     y' = R sin(phi) cos(i),

    so every pixel can be mapped back to its in-plane radius

        R = sqrt( x'^2 + (y' / q)^2 ),   q = b/a = cos(inclination).

    This is *not* the full 3-D radius: it assumes all light lies at ``z = 0``
    (any vertical structure / LOS extent is ignored).

    Parameters
    ----------
    map2d
        Reference map (only ``shape`` / ``wcs`` / ``bad`` are used).
    center
        Galaxy centre in pixels ``(x, y)`` or ``"auto"``.
    position_angle
        Major-axis PA, East of North (Quantity, e.g. ``53 * u.deg``).
    inclination
        Disk inclination ``i`` (0 = face-on).  Exactly one of
        ``inclination`` / ``axis_ratio`` must be given; ``q = cos i``.
    axis_ratio
        Axis ratio ``q = b/a`` (``= cos i`` for a thin disk), e.g. the NSA
        ellipticity-derived ``b/a``.
    scale
        Pixel scale; default is the WCS plate scale (arcsec/pixel) or
        0.5 arcsec/pixel when no WCS.  The returned radius carries the unit
        of ``scale`` (arcsec by default; pass e.g. ``kpc/pixel`` for kpc).

    Returns
    -------
    Map2D
        ``value`` = deprojected galactocentric radius ``R_cyl`` (same unit as
        ``scale``); geometry metadata (PA / inclination / axis ratio /
        assumption) is stored in ``meta`` for provenance.
    """
    if (inclination is None) == (axis_ratio is None):
        raise ValueError("provide exactly one of inclination= or axis_ratio=")

    if inclination is not None:
        i_deg = float(Quantity(inclination).to(u.deg).value)
        if not (0.0 <= i_deg <= 90.0):
            raise ValueError("inclination must be within [0, 90] deg")
        q = float(np.cos(np.radians(i_deg)))
    else:
        q = float(axis_ratio)
        if not (0.0 < q <= 1.0):
            raise ValueError("axis_ratio q = b/a must be in (0, 1]")
        i_deg = float(np.degrees(np.arccos(np.clip(q, 0.0, 1.0))))

    pa = Quantity(position_angle).to(u.deg)
    _center = _resolve_center(map2d, center)
    s = _radius_scale(map2d, scale)

    r_cyl = elliptical_radius_map(
        map2d.shape,
        center=_center,
        scale=s,
        position_angle=pa,
        ellipticity=1.0 - q,  # 1 - b/a ; radius is the deprojected major-axis R
    )
    radius = Map2D(
        value=r_cyl,
        mask=np.zeros(map2d.shape, dtype=bool),
        wcs=map2d.wcs,
        meta={
            "quantity": "deprojected galactocentric radius R_cyl (thin disk)",
            "center": _center,
            "position_angle_deg": float(pa.value),
            "inclination_deg": i_deg,
            "axis_ratio_b_over_a": q,
            "assumption": "thin axisymmetric disk, all emission at z = 0; "
                          "R = sqrt(x'^2 + (y'/q)^2), q = cos i",
        },
        provenance=map2d.provenance,
    )
    return radius


def radius_over_re(radius_map: Map2D, re: Quantity) -> Map2D:
    """Normalise a radius map by the effective radius: ``R / R_e``.

    Parameters
    ----------
    radius_map
        Any radius ``Map2D`` (projected or deprojected; arcsec, kpc, ...).
    re
        Effective radius in the **same unit** as ``radius_map.value``.

    Returns
    -------
    Map2D
        Dimensionless ``R / R_e`` map (masks inherited from ``radius_map``).
    """
    re = Quantity(re).to(radius_map.value.unit)
    val = radius_map.value / re
    meta = dict(radius_map.meta)
    meta["quantity"] = "R / Re"
    meta["re"] = re
    return Map2D(
        value=val,
        mask=radius_map.mask,
        wcs=radius_map.wcs,
        mask_defs=radius_map.mask_defs,
        meta=meta,
        provenance=radius_map.provenance,
    )
