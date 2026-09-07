"""MaNGA-side stellar-template preparation (E-MILES SSP library).

Boundary note
-------------
This module sits on the **data** side of the boundary and therefore must not
import ``udong.science``:

* it knows how to locate the E-MILES SSP model file used for MaNGA-style fits
  (:func:`default_sps_file`);
* it knows how to build rest-frame, log-rebinned, LSF-matched templates from
  that library (:func:`make_manga_templates`).

The actual *fitting* (subtracting the stellar continuum of a spectrum or of
one spaxel) is survey-agnostic and lives in
``udong.science.spectroscopy.continuum`` (``fit_stellar_continuum`` /
``fit_cube_spaxel``); pipelines import both sides and pass the templates in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "default_sps_file",
    "make_manga_templates",
]


def default_sps_file() -> Path:
    """Default E-MILES SSP model file (override with ``UDONG_SPS_MODELS``)."""
    import os

    env = os.environ.get("UDONG_SPS_MODELS")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Repository" / "udong_data" / "sps" / "spectra_emiles_9.0.npz"


def make_manga_templates(
    velscale: float,
    *,
    wave_rest: np.ndarray,
    fwhm_aa: np.ndarray,
    sps_file: Path | str | None = None,
    lam_range: tuple[float, float] = (3520.0, 7420.0),
    norm_range: tuple[float, float] | None = None,
):
    """Build log-rebinned, LSF-matched E-MILES templates for MaNGA spectra.

    Parameters
    ----------
    velscale : float
        km/s per pixel of the MaNGA cube.
    wave_rest, fwhm_aa : (n,) arrays
        Rest-frame wavelength (AA) and galaxy LSF FWHM (AA) of the cube
        pixels (used to convolve the templates to the MaNGA resolution).
    sps_file : path of the E-MILES ``.npz`` library
        (see :func:`default_sps_file`).
    lam_range : wavelength range (AA, rest) over which templates are kept.
    norm_range : optional normalisation band for light-weight output.

    Returns
    -------
    templates : (n_pix, n_templates) log-rebinned rest-frame templates
    lam_temp : (n_pix,) template wavelengths (AA)
    """
    from ppxf.sps_util import sps_lib  # lazy optional dependency

    sps = sps_lib(
        str(Path(sps_file if sps_file is not None else default_sps_file())),
        velscale,
        fwhm_gal={"lam": np.asarray(wave_rest), "fwhm": np.asarray(fwhm_aa)},
        lam_range=list(lam_range),
        norm_range=list(norm_range) if norm_range is not None else None,
    )
    templates = sps.templates.reshape(sps.templates.shape[0], -1)
    return templates, np.asarray(sps.lam_temp)
