"""Line-of-sight kinematics measurement helpers.

These functions operate on ``Map2D`` velocity / velocity-dispersion maps and
are tracer-agnostic (stellar or gas).  They implement the *measurement* layer:

* ``radial_velocity_profile`` / ``radial_dispersion_profile``: 1-D rotation
  curve / dispersion profile (rebinned from the 2-D maps).
* ``global_mean_velocity``: IVAR-weighted mean line-of-sight velocity and its
  uncertainty.
* ``global_dispersion``: area/IVAR (or user-supplied) weighted mean velocity
  dispersion.
* ``lsf_correct_sigma``: instrument-LSF deconvolution
  ``sigma_int = sqrt(sigma_obs^2 - sigma_lsf^2)``.
* ``fit_velocity_plane``: linear least-squares fit ``v = v0 + a x + b y`` to
  the velocity field (a first-order description of ordered rotation) plus a
  sky-frame position angle of the velocity gradient.

Masks are interpreted through each map's ``MaskDefs`` (DONOTUSE by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from udong.core.coordinates import wcs_center_pixel
from udong.core.map import Map2D
from udong.core.uncertainty import ivar_to_sigma, propagate_weighted_mean
from udong.science.spatial import RadialProfile, radial_profile

__all__ = [
    "lsf_correct_sigma",
    "radial_velocity_profile",
    "radial_dispersion_profile",
    "global_mean_velocity",
    "global_dispersion",
    "fit_velocity_plane",
    "VelocityPlaneFit",
]


# --------------------------------------------------------------------------- #
def lsf_correct_sigma(sigma_obs: Map2D, sigma_lsf) -> Map2D:
    """Subtract the instrumental line-spread function in quadrature.

    ``sigma_int = sqrt(max(sigma_obs^2 - sigma_lsf^2, 0))``.  Pixels where the
    observed dispersion is smaller than the instrumental LSF (or that are bad
    in either input) are flagged bad (their corrected value is undefined).
    """
    if isinstance(sigma_lsf, Map2D):
        if sigma_lsf.shape != sigma_obs.shape:
            raise ValueError("sigma_lsf map must have the same shape as sigma_obs")
        lsf = np.asarray(sigma_lsf.value.value)
        bad_lsf = sigma_lsf.bad
    else:
        lsf_scalar = float(Quantity(sigma_lsf).to_value(sigma_obs.value.unit))
        lsf = np.full(sigma_obs.shape, lsf_scalar)
        bad_lsf = np.zeros(sigma_obs.shape, dtype=bool)
    obs = np.asarray(sigma_obs.value.value)
    bad = sigma_obs.bad | bad_lsf | ~np.isfinite(obs) | ~np.isfinite(lsf)
    with np.errstate(invalid="ignore"):
        corr2 = obs**2 - lsf**2
    corr = np.sqrt(np.where(corr2 > 0, corr2, np.nan))
    bad |= corr2 <= 0
    return Map2D(
        value=Quantity(corr, sigma_obs.value.unit),
        uncertainty=None,
        mask=bad,
        wcs=sigma_obs.wcs,
        meta=dict(sigma_obs.meta, quantity="sigma_int"),
        provenance=sigma_obs.provenance,
    )


# --------------------------------------------------------------------------- #
def radial_velocity_profile(velocity: Map2D, **kwargs) -> RadialProfile:
    """Radial (rotation-curve) profile of a line-of-sight velocity map."""
    return radial_profile(velocity, **kwargs)


def radial_dispersion_profile(dispersion: Map2D, **kwargs) -> RadialProfile:
    """Radial profile of a velocity-dispersion map."""
    return radial_profile(dispersion, **kwargs)


# --------------------------------------------------------------------------- #
def global_mean_velocity(velocity: Map2D) -> tuple[Quantity, Quantity]:
    """IVAR-weighted mean line-of-sight velocity and its 1-sigma uncertainty."""
    if velocity.ivar is None:
        raise ValueError("velocity map must carry IVAR uncertainties")
    v = np.asarray(velocity.value.value)
    iv = np.asarray(velocity.ivar.value)
    mean, mivar = propagate_weighted_mean(v, iv, axis=None)
    if not np.isfinite(mean):
        return Quantity(np.nan, velocity.value.unit), Quantity(np.nan, velocity.value.unit)
    return Quantity(mean, velocity.value.unit), Quantity(ivar_to_sigma(mivar), velocity.value.unit)


def global_dispersion(
    dispersion: Map2D,
    weight_map: Map2D | None = None,
) -> tuple[Quantity, Quantity]:
    """Weighted mean velocity dispersion and its uncertainty.

    Default weighting is by the inverse variance of each dispersion estimate
    (appropriate when combining independent measurements).  Pass
    ``weight_map`` (e.g. a flux/luminosity map) for luminosity-weighted
    values.
    """
    s = np.asarray(dispersion.value.value)
    if weight_map is not None:
        if weight_map.shape != dispersion.shape:
            raise ValueError("weight_map must have the same shape as dispersion")
        w = np.asarray(weight_map.value.value)
        ok = np.isfinite(s) & (w > 0) & ~dispersion.bad & ~weight_map.bad
        wsum = float(np.sum(np.where(ok, w, 0.0)))
        if wsum <= 0:
            return Quantity(np.nan, dispersion.value.unit), Quantity(np.nan, dispersion.value.unit)
        mean_w = float(np.sum(np.where(ok, s, 0.0) * np.where(ok, w, 0.0)) / wsum)
        # weighted scatter; uncertainty ~ scatter/sqrt(N_eff - 1)
        n_eff = float(np.count_nonzero(ok))
        scatter = float(np.sqrt(np.sum(np.where(ok, w, 0.0) * (np.where(ok, s, 0.0) - mean_w) ** 2) / wsum))
        unc = scatter / np.sqrt(max(n_eff - 1, 1.0)) if n_eff > 1 else np.nan
        return Quantity(mean_w, dispersion.value.unit), Quantity(unc, dispersion.value.unit)
    if dispersion.ivar is None:
        raise ValueError("dispersion map must carry IVAR (or pass weight_map=)")
    iv = np.asarray(dispersion.ivar.value)
    ok = np.isfinite(s) & (iv > 0) & ~dispersion.bad
    mean, mivar = propagate_weighted_mean(s, iv, axis=None)
    return Quantity(mean, dispersion.value.unit), Quantity(ivar_to_sigma(mivar), dispersion.value.unit)


# --------------------------------------------------------------------------- #
@dataclass
class VelocityPlaneFit:
    """Linear fit ``v = v0 + a*(x-cx) + b*(y-cy)`` to a velocity field."""

    v0: Quantity  # velocity at the fit centre
    a: Quantity  # dV/dx in (map unit)/pixel
    b: Quantity  # dV/dy
    cx: float
    cy: float
    n: int
    gradient_pa: Quantity  # sky position angle of steepest gradient (E of N)
    residual_rms: Quantity

    @property
    def gradient_magnitude(self) -> Quantity:
        """|grad v| in map units per pixel."""
        return Quantity(np.hypot(self.a.value, self.b.value), self.a.unit)


def fit_velocity_plane(
    velocity: Map2D,
    center: tuple[float, float] | str = "auto",
) -> VelocityPlaneFit:
    """Least-squares plane fit to the (good) velocity field.

    Returns coefficients plus the sky-frame position angle of the velocity
    gradient.  For a rotating disc the line-of-sight velocity changes fastest
    along the kinematic major axis, so ``gradient_pa`` points along that axis
    (the receding side; modulo 180 deg it is the major-axis PA).
    """
    if center == "auto":
        if velocity.wcs is not None:
            cx, cy = wcs_center_pixel(velocity.wcs)
        else:
            ny, nx = velocity.shape
            cx, cy = (nx - 1) / 2, (ny - 1) / 2
    else:
        if isinstance(center, str):
            raise ValueError(f"invalid center {center!r}")
        cx, cy = center
    yy, xx = np.indices(velocity.shape, dtype=float)
    v = np.asarray(velocity.value.value)
    bad = velocity.bad | ~np.isfinite(v)
    X = np.column_stack([np.ones_like(v[~bad]), (xx[~bad] - cx), (yy[~bad] - cy)])
    y = v[~bad]
    if len(y) < 3:
        raise ValueError("too few good pixels to fit a velocity plane")
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    v0 = float(coef[0])
    a = float(coef[1])
    b = float(coef[2])
    resid = y - X @ coef
    rms = float(np.sqrt(np.mean(resid**2)))

    # sky position angle of the gradient (requires a WCS)
    pa = Quantity(np.nan, u.deg)
    if velocity.wcs is not None:
        try:
            wcs = velocity.wcs
            x0, y0 = cx, cy
            # gradient direction in pixel space (choose a step along it)
            step = 1.0
            x1, y1 = cx + step * a / max(np.hypot(a, b), 1e-12), cy + step * b / max(np.hypot(a, b), 1e-12)
            c0 = wcs.pixel_to_world(x0, y0)
            c1 = wcs.pixel_to_world(x1, y1)
            pa = c0.position_angle(c1).to(u.deg)
        except Exception:
            pa = Quantity(np.nan, u.deg)

    unit = velocity.value.unit
    return VelocityPlaneFit(
        v0=Quantity(v0, unit),
        a=Quantity(a, unit / u.pixel),
        b=Quantity(b, unit / u.pixel),
        cx=cx,
        cy=cy,
        n=int(len(y)),
        gradient_pa=pa,
        residual_rms=Quantity(rms, unit),
    )
