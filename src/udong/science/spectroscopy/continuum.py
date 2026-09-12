"""Stellar-continuum fitting with pPXF (generic, survey-agnostic).

This module is a thin, explicit wrapper around ``ppxf`` (Cappellari &
Emsellem 2004; Cappellari 2017, 2023).  It takes a **rest-frame**,
log-rebinned spectrum (or one spaxel of an IFU cube) plus a set of
rest-frame stellar templates (already matched to the instrumental LSF by the
caller) and returns the best-fitting stellar continuum model, the residual
(continuum-subtracted spectrum) and the stellar kinematics.

Contents
--------
* :func:`fit_stellar_continuum` -- fit one 1-D spectrum segment.
* :func:`fit_cube_spaxel` -- convenience wrapper for one spaxel of a
  :class:`~udong.core.cube.Cube` (works with any IFU survey, including the
  MaNGA ``MangaCube`` which subclasses ``Cube``).

Layering
--------
* This module depends only on ``udong.core`` types plus numpy; ``ppxf`` is
  imported lazily so the rest of udong does not require it.
* Survey-specific template building (E-MILES library, MaNGA LSF convolution,
  default model file) lives in the *data layer*
  (``udong.data.manga.continuum.make_manga_templates``).  The data layer
  therefore never imports this module -- cross-layer orchestration happens
  in pipelines/examples, which import both sides.
* generic per-spaxel entry point for any core ``Cube`` (MaNGA or MUSE):
  :func:`fit_cube_spaxel` / :class:`SpaxelContinuumResult`, kept so notebooks
  written before the data/science split keep working with an import-path-only
  change.

Conventions
-----------
* The galaxy spectrum must have *constant* logarithmic wavelength sampling
  (``velscale = c * dln(lambda)``), as produced by ``ppxf.ppxf_util.log_rebin``
  or by MaNGA LOGCUBE (which is already log-sampled).
* Emission windows and bad pixels must be marked through ``goodpixels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from udong.core.cube import Cube

__all__ = [
    "C_KM",
    "EMISSION_REST_LINES",
    "StellarContinuumFit",
    "SpaxelContinuumResult",
    "fit_cube_spaxel",
    "fit_stellar_continuum",
]

C_KM = 299792.458

# rest-frame vacuum wavelengths (AA) of the strong lines masked in the stellar
# fit (generic optical IFU set; kept as module data for provenance)
EMISSION_REST_LINES: tuple[float, ...] = (
    3727.09, 3729.88, 3869.86, 4862.68, 4960.30, 5008.24,
    6302.05, 6365.54, 6549.86, 6564.61, 6585.27, 6718.29, 6732.67,
)


@dataclass
class StellarContinuumFit:
    """Result of a pPXF stellar fit on one spectrum segment.

    Attributes
    ----------
    velocity
        km/s (relative to the assumed rest frame).
    sigma
        km/s (observed, includes instrumental LSF).
    chi2
        Reduced chi-square reported by pPXF.
    weights
        Template weights (normalised by ppxf).
    model
        Best-fit continuum, same length as the input flux.
    residual
        ``flux - model`` (the continuum-subtracted spectrum).
    goodpixels
        Boolean array, pixels used in the fit.
    n_used
        Number of used pixels (``goodpixels.sum()``).
    success, message
        Fit status; reserved for future failure handling.
    """

    velocity: float
    sigma: float
    chi2: float
    weights: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    goodpixels: np.ndarray
    n_used: int = 0
    success: bool = True
    message: str = ""


def fit_stellar_continuum(
    wavelength_rest: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    goodpixels: np.ndarray,
    *,
    templates: np.ndarray,
    lam_temp: np.ndarray,
    velscale: float,
    start: tuple[float, float] = (0.0, 150.0),
    moments: int = 2,
    degree: int = 12,
    mdegree: int = 0,
    redshift: float = 0.0,
) -> StellarContinuumFit:
    """Fit the stellar continuum of one rest-frame spectrum segment.

    Parameters
    ----------
    wavelength_rest : (n,) float
        Rest-frame wavelengths (AA) of the segment; must be logarithmically
        sampled with the same ``velscale`` as the templates.
    flux, ivar : (n,) float
        Flux density and inverse variance (same units).
    goodpixels : (n,) bool
        Pixels to include in the fit (emission windows / bad pixels excluded).
    templates : (m, n_templates) float
        Log-rebinned rest-frame stellar templates (matched to the galaxy LSF).
    lam_temp : (m,) float
        Rest-frame wavelengths of the template pixels.
    velscale : float
        km/s per pixel (same for galaxy and templates).
    start, moments, degree, mdegree
        pPXF settings (see ppxf docs).
    redshift : float
        If ``> 0`` the input wavelengths are taken as *observed* and divided
        by ``(1 + redshift)``; otherwise they are assumed already rest-frame.
        In both cases the returned model/residual are on the **input** grid.
    """
    from ppxf.ppxf import ppxf  # lazy import (optional dependency)

    wavelength_rest = np.asarray(wavelength_rest, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    goodpixels = np.asarray(goodpixels, dtype=bool)

    lam = (
        wavelength_rest / (1.0 + redshift)
        if redshift and redshift > 0
        else wavelength_rest
    )  # rest-frame labelling

    # normalise to ~unity to avoid numerical issues (ppxf doc)
    med = float(np.nanmedian(flux[goodpixels & (flux > 0)]))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    gal = flux / med
    noise = np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 0.0)) / med, np.inf)

    pp = ppxf(templates, gal, noise, velscale,
              start=list(start), moments=moments, degree=degree,
              mdegree=mdegree, lam=lam, lam_temp=lam_temp,
              mask=goodpixels, quiet=True)

    sol = np.atleast_1d(pp.sol)
    vel = float(sol[0]) if sol.size else np.nan
    sig = float(sol[1]) if sol.size > 1 else np.nan
    return StellarContinuumFit(
        velocity=vel,
        sigma=sig,
        chi2=float(pp.chi2),
        weights=np.asarray(pp.weights),
        model=pp.bestfit * med,
        residual=(gal - pp.bestfit) * med,
        goodpixels=goodpixels,
        n_used=int(np.count_nonzero(goodpixels)),
        success=True,
        message="",
    )


@dataclass
class SpaxelContinuumResult:
    """Per-spaxel pPXF result, arrays aligned with the fitted rest-frame range.

    Attributes
    ----------
    x, y
        Pixel coordinates of the spaxel (``flux[:, y, x]``).
    lam_rest
        Rest-frame wavelengths (AA) of the fitted segment.
    residual
        Continuum-subtracted flux density (cube units).
    model
        Best-fitting stellar continuum model (cube units).
    noise
        1-sigma noise per pixel (cube units); NaN where IVAR <= 0.
    goodpixels
        Boolean pixels used in the stellar fit.
    velocity, sigma
        Stellar kinematics (km/s); ``sigma`` is observed (includes LSF).
    chi2
        Reduced chi-square of the stellar fit.
    n_used
        Number of pixels used in the fit.
    template_file
        Optional path of the template library (informational / provenance).
    meta
        Free-form fit parameters (redshift, velscale, fit range, ...).
    """

    x: int
    y: int
    lam_rest: np.ndarray
    residual: np.ndarray
    model: np.ndarray
    noise: np.ndarray
    goodpixels: np.ndarray
    velocity: float
    sigma: float
    chi2: float
    n_used: int
    template_file: str = ""
    meta: dict = field(default_factory=dict)


def fit_cube_spaxel(
    cube: Cube,
    x: int,
    y: int,
    templates: np.ndarray,
    lam_temp: np.ndarray,
    *,
    redshift: float,
    velscale: float,
    fit_range: tuple[float, float] = (3650.0, 7320.0),
    line_mask_half_width_km_s: float = 450.0,
    start_velocity: float | None = None,
    start_sigma: float | None = None,
    degree: int = 12,
    template_file: str = "",
) -> SpaxelContinuumResult | None:
    """Fit and subtract the stellar continuum of one spaxel of an IFU cube.

    Survey-agnostic: ``cube`` only needs the :class:`~udong.core.cube.Cube`
    interface (``wavelength``, ``flux``, ``ivar``, ``mask``, ``mask_defs``);
    a MaNGA ``MangaCube`` satisfies it directly.  The stellar templates must
    already be rest-frame, log-rebinned and LSF-matched for this survey
    (e.g. built with ``udong.data.manga.continuum.make_manga_templates``).

    Parameters
    ----------
    cube
        Datacube (``flux[:, y, x]`` convention, wavelength axis required).
    x, y
        Spaxel pixel coordinates.
    templates, lam_temp
        Log-rebinned rest-frame templates and their wavelengths.
    redshift
        Systemic redshift used to move observed wavelengths to rest frame.
    velscale
        km/s per pixel of the (log-sampled) cube.
    fit_range
        Rest-frame wavelength range (AA) used for the stellar fit.
    line_mask_half_width_km_s
        Half width (km/s) of the windows masked around each strong emission
        line during the stellar fit.
    start_velocity, start_sigma
        Optional pPXF starting values (km/s).
    degree
        pPXF multiplicative Legendre polynomial degree.
    template_file
        Optional template-library path recorded in the result (provenance).

    Returns
    -------
    SpaxelContinuumResult or None
        ``None`` when the spaxel has no usable continuum pixels.
    """
    wavelength = cube.wavelength
    if wavelength is None:
        raise RuntimeError("cube has no wavelength vector")
    flux3 = cube.flux  # Quantity
    ivar3 = cube.ivar
    mask3 = cube.mask
    if ivar3 is None or mask3 is None:
        raise RuntimeError("cube has no ivar/mask arrays")
    wave_obs = np.asarray(wavelength.value, dtype=float)
    lam_rest = wave_obs / (1.0 + redshift)
    sel = (lam_rest >= fit_range[0]) & (lam_rest <= fit_range[1])
    if sel.sum() < 200:
        return None

    flux = np.asarray(flux3.value)[:, y, x][sel].astype(float)
    ivar = np.asarray(ivar3)[:, y, x][sel].astype(float)
    rawmask = mask3[:, y, x][sel]
    bad = (
        cube.mask_defs.unmask(rawmask, "DONOTUSE")
        if cube.mask_defs is not None
        else (rawmask != 0)
    )
    # skip pure-noise / empty spaxels
    good0 = (~bad) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    if good0.sum() < 200:
        return None

    # mask emission-line windows (rest frame)
    lam = lam_rest[sel]
    emission = np.zeros(lam.size, dtype=bool)
    for l0 in EMISSION_REST_LINES:
        emission |= np.abs(C_KM * (lam - l0) / l0) < line_mask_half_width_km_s
    goodpixels = good0 & (~emission)

    start_v = start_velocity if start_velocity is not None else 0.0
    start_s = start_sigma if start_sigma is not None else 200.0
    fit = fit_stellar_continuum(
        lam, flux, ivar, goodpixels,
        templates=templates, lam_temp=lam_temp, velscale=velscale,
        start=(start_v, start_s), degree=degree,
    )
    if not np.isfinite(fit.velocity):
        return None
    return SpaxelContinuumResult(
        x=int(x), y=int(y),
        lam_rest=lam,
        residual=fit.residual,
        model=fit.model,
        noise=np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 0.0)), np.nan),
        goodpixels=goodpixels,
        velocity=float(fit.velocity),
        sigma=float(fit.sigma),
        chi2=float(fit.chi2),
        n_used=fit.n_used,
        template_file=template_file,
        meta={"degree": degree, "fit_range": fit_range,
              "redshift": redshift, "velscale": velscale},
    )
