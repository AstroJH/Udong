"""Coordinate helpers shared by all surveys.

Everything here operates on *pixel grids* and *astropy WCS objects*, never on
survey-specific data products.  Conventions used across ``udong.core``:

* pixel coordinates are ``(x, y)`` with ``x`` along RA (column axis) and
``y`` along Dec (row axis); array indexing is ``value[y, x]``;
* angles are astropy ``Quantity`` and follow the astronomical convention
(position angles measured East of North);
* when no WCS is given, code falls back to a plain pixel grid with a caller
supplied pixel ``scale``.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

__all__ = [
    "pixel_scale",
    "wcs_center_pixel",
    "radial_map",
    "elliptical_radius_map",
    "pixel_to_sky",
    "sky_to_pixel",
]


def pixel_scale(wcs: WCS) -> Quantity:
    """Mean pixel scale of a celestial WCS, in arcsec/pixel.

    Parameters
    ----------
    wcs
        2-D (or higher) celestial WCS.

    Returns
    -------
    Quantity
        ``(mean plate scale) * u.arcsec / u.pixel``.  The two in-plane plate
        scales are averaged; exact for square pixels (e.g. MaNGA's 0.5 arcsec
        grid) and approximate for strongly non-square pixels.
    """
    scales = proj_plane_pixel_scales(wcs) # degrees per pixel

    if not isinstance(scales, Quantity): # check: Quantity in new astropy
        scales = Quantity(scales, u.deg)

    return scales.mean().to(u.arcsec) / u.pixel


def wcs_center_pixel(wcs: WCS) -> tuple[float, float]:
    """Return the WCS reference pixel ``(CRPIX)`` as ``(x, y)``.

    Note
    ----
    The value is returned exactly as stored in the FITS WCS, i.e. following the
    FITS *1-based* convention.  Callers that need a 0-based array index
    (``value[y, x]``) should offset by one or pass an explicit ``center``.
    """
    return float(wcs.wcs.crpix[0]), float(wcs.wcs.crpix[1])


def radial_map(
    shape: tuple[int, int],
    center: tuple[float, float] | None = None,
    scale: Quantity = 0.5 * u.arcsec / u.pixel,
    wcs: WCS | None = None,
) -> Quantity:
    """2-D map of the circular radius (arcsec) around ``center``.

    Parameters
    ----------
    shape
        Output grid ``(ny, nx)``.
    center
        Pixel centre ``(x, y)``; defaults to the geometric grid centre
        ``((nx - 1) / 2, (ny - 1) / 2)``.
    scale
        Pixel scale used when no ``wcs`` is given (default 0.5 arcsec/pix, the
        MaNGA grid spacing).
    wcs
        If given, ``scale`` is ignored and taken from the WCS instead.

    Returns
    -------
    Quantity
        ``(ny, nx)`` array of radii in arcsec.
    """
    ny, nx = shape
    cx, cy = center if center is not None else ((nx - 1) / 2, (ny - 1) / 2)
    if wcs is not None:
        scale = pixel_scale(wcs)
    yy, xx = np.indices((ny, nx), dtype=float)
    r_pix = np.hypot(xx - cx, yy - cy)
    return Quantity(r_pix, u.pixel) * scale


def elliptical_radius_map(
    shape: tuple[int, int],
    center: tuple[float, float] | None = None,
    scale: Quantity = 0.5 * u.arcsec / u.pixel,
    position_angle: Quantity = 0.0 * u.deg,
    ellipticity: float = 0.0,
    wcs: WCS | None = None,
) -> Quantity:
    """2-D map of the elliptical *major-axis* radius (arcsec).

    Used e.g. to build ``R/R_e`` radial profiles.  Follows the standard
    definition: an ellipse with position angle ``PA`` (East of North) and axis
    ratio ``q = b/a = 1 - ellipticity``; the radius of each pixel is measured
    along the major axis after deprojecting by the axis ratio.

    Parameters
    ----------
    shape
        Output grid ``(ny, nx)``.
    center
        Pixel centre ``(x, y)``; defaults to the geometric grid centre.
    scale
        Pixel scale used when no ``wcs`` is given (arcsec/pixel).
    position_angle
        Major-axis position angle, East of North.
    ellipticity
        ``1 - b/a``; must lie in ``[0, 1)`` (0 = circle).
    wcs
        If given, ``scale`` is ignored and taken from the WCS instead.

    Returns
    -------
    Quantity
        ``(ny, nx)`` array of major-axis radii in arcsec.

    Notes
    -----
    Internal pixel offsets use ``dy = cy - y`` so that ``+y`` is North.  With
    ``ellipticity = 0`` this reduces exactly to :func:`radial_map`.
    """
    ny, nx = shape
    cx, cy = center if center is not None else ((nx - 1) / 2, (ny - 1) / 2)
    if wcs is not None:
        scale = pixel_scale(wcs)
    if not (0 <= ellipticity < 1):
        raise ValueError("ellipticity must be in [0, 1)")
    yy, xx = np.indices((ny, nx), dtype=float)
    dx, dy = xx - cx, cy - yy  # dy flipped so +y is North
    pa = position_angle.to(u.rad).value

    # rotate to the ellipse frame (major axis along x')
    xp = dx * np.cos(pa) - dy * np.sin(pa)
    yp = dx * np.sin(pa) + dy * np.cos(pa)
    q = 1.0 - ellipticity  # minor/major axis ratio
    r_pix = np.hypot(xp, yp / q)
    return Quantity(r_pix, u.pixel) * scale


def pixel_to_sky(wcs: WCS, x, y) -> SkyCoord:
    """Convert pixel coordinates ``(x, y)`` to a sky coordinate.

    Parameters
    ----------
    wcs
        2-D celestial WCS (e.g. ``Cube.spatial_wcs`` or ``Map2D.wcs``).
    x, y
        Pixel coordinates (floats allowed for sub-pixel positions).

    Returns
    -------
    SkyCoord
    """
    return wcs.pixel_to_world(x, y)


def sky_to_pixel(wcs: WCS, coord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sky coordinate to pixel coordinates ``(x, y)``.

    Parameters
    ----------
    wcs
        2-D celestial WCS.
    coord
        Sky position (scalar or array of positions).

    Returns
    -------
    (x, y)
        Pixel coordinates as float arrays.
    """
    return wcs.world_to_pixel(coord)
