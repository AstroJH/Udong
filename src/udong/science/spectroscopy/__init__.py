"""Spectroscopy: measuring line/continuum properties from spectra or cubes.

Boundary:
* ``spectroscopy`` = per-spectrum / per-spaxel *measurements of the line
  profile* (flux, equivalent width, and the moments M0/M1/M2) plus
  basic spectral operations.  These need wavelength + continuum.
* ``udong.science.kinematics`` = analysis of already-produced velocity /
  dispersion *fields* (rotation curves, plane fits, LSF deconvolution,
  global statistics) — it operates on 2-D maps, not raw spectra.

So the moments M1 (velocity) and M2 (dispersion) are *estimated here* (from
spectra/cubes, model-free) and then *consumed by* ``kinematics``.
"""

from __future__ import annotations

from udong.science.spectroscopy.basic import (
    integrate_band,
    mask_wavelength_range,
    to_rest_frame,
)
from udong.science.spectroscopy.continuum import (
    SpaxelContinuumResult,
    StellarContinuumFit,
    fit_cube_spaxel,
    fit_stellar_continuum,
)
from udong.science.spectroscopy.lines import (
    HA_NII_WINDOW,
    HB_WINDOW,
    OIII_WINDOW,
    SII_WINDOW,
    GaussianComponent,
    Line,
    LineWindow,
    WindowFit,
    bic,
    fit_fixed_kinematics,
    fit_single_and_double,
    fit_window,
)
from udong.science.spectroscopy.moments import (
    line_moments,
    moment_maps,
    subtract_linear_continuum,
    velocity_axis,
)

__all__ = [
    "GaussianComponent",
    "HA_NII_WINDOW",
    "HB_WINDOW",
    "Line",
    "LineWindow",
    "OIII_WINDOW",
    "SII_WINDOW",
    "SpaxelContinuumResult",
    "StellarContinuumFit",
    "WindowFit",
    "fit_cube_spaxel",
    "bic",
    "fit_fixed_kinematics",
    "fit_single_and_double",
    "fit_stellar_continuum",
    "fit_window",
    "integrate_band",
    "line_moments",
    "mask_wavelength_range",
    "moment_maps",
    "subtract_linear_continuum",
    "to_rest_frame",
    "velocity_axis",
]
