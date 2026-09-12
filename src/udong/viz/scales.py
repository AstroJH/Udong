"""Colour-scale normalisations shared by ``udong.viz`` plotters.

Supports the astronomy-standard choices:

* ``"linear"``  -- default; honours explicit ``vmin``/``vmax``;
* ``"log"``     -- requires positive data (or a positive ``vmin``);
* ``"symlog"``  -- for bipolar data spanning decades (``linthresh``);
* ``symmetric=True`` -- force the range symmetric about zero, *vmin=-vmax*
  with the largest absolute bound taken from the data percentiles or the
  user-supplied limits.  This is the standard choice for velocity maps: it
  puts zero on the neutral colour of a diverging colormap, so blue/red are
  directly comparable.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

__all__ = ["finite_values", "resolve_norm"]

_SCALES = ("linear", "log", "symlog")


def finite_values(data) -> np.ndarray:
    """Finite, unmasked values of a (possibly masked) array, as 1-D floats."""
    a = np.ma.asarray(data).compressed() if np.ma.isMaskedArray(data) else np.asarray(data)
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def resolve_norm(
    data,
    scale: str = "linear",
    vmin=None,
    vmax=None,
    linthresh=None,
    symmetric: bool = False,
):
    """Build a matplotlib ``Normalize`` for the requested scale/limits."""
    if scale not in _SCALES:
        raise ValueError(f"unknown scale {scale!r}; use one of {_SCALES}")
    vals = finite_values(data)
    if vals.size == 0:
        raise ValueError("no finite data to set the colour scale from")
    if symmetric and scale == "log":
        raise ValueError("symmetric=True is not meaningful for a log scale")

    if symmetric:
        lo = abs(float(vmin)) if vmin is not None else abs(float(np.percentile(vals, 2)))
        hi = abs(float(vmax)) if vmax is not None else abs(float(np.percentile(vals, 98)))
        v = max(lo, hi)
        vmin, vmax = -v, v

    if scale == "linear":
        return Normalize(vmin=vmin, vmax=vmax)

    if scale == "log":
        pos = vals[vals > 0]
        if pos.size == 0:
            raise ValueError(
                "scale='log' needs positive data; pass vmin>0 or use "
                "scale='linear'/'symlog'"
            )
        lo = float(vmin) if vmin is not None and vmin > 0 else float(pos.min())
        hi = float(vmax) if vmax is not None else float(pos.max())
        return LogNorm(vmin=lo, vmax=hi)

    lo = float(vmin) if vmin is not None else float(vals.min())
    hi = float(vmax) if vmax is not None else float(vals.max())
    if linthresh is None:
        linthresh = max(abs(lo), abs(hi)) / 100.0 or 1.0
    return SymLogNorm(linthresh=float(linthresh), vmin=lo, vmax=hi)
