"""Uncertainty conventions and propagation primitives.

Convention
----------
All survey products store errors as *inverse variance* (IVAR); ``IVAR <= 0``
means the value is invalid (should be masked).  This module provides
conversions between representations and a small set of analytic propagation
primitives.  Monte-Carlo / bootstrap propagation is planned for later phase but
``propagate()`` already fixes the interface.
"""

from __future__ import annotations

import numpy as np
from astropy.nddata import StdDevUncertainty, VarianceUncertainty

# astropy >= 8 renamed InverseVarianceUncertainty -> InverseVariance
try:  # pragma: no cover - depends on astropy version
    from astropy.nddata import InverseVariance as InverseVarianceUncertainty
except ImportError:  # astropy < 8
    from astropy.nddata import InverseVarianceUncertainty

__all__ = [
    "InverseVarianceUncertainty",
    "ivar_to_sigma",
    "ivar_to_variance",
    "sigma_to_ivar",
    "variance_to_ivar",
    "as_uncertainty",
    "propagate_sum",
    "propagate_weighted_mean",
    "propagate_ratio",
]


def ivar_to_sigma(ivar) -> np.ndarray:
    """Convert inverse variance to standard deviation.

    ``sigma = 1 / sqrt(ivar)``.  Pixels with ``ivar <= 0`` (including NaN and
    -inf) are flagged invalid and return NaN.

    Parameters
    ----------
    ivar : array-like
        Inverse variance (``1/sigma**2``).

    Returns
    -------
    numpy.ndarray
        Standard deviation; NaN where the input is not a valid IVAR.
    """
    ivar = np.asarray(ivar, dtype=float)
    sigma = np.full(ivar.shape, np.nan)
    ok = ivar > 0
    sigma[ok] = 1.0 / np.sqrt(ivar[ok])
    return sigma


def ivar_to_variance(ivar) -> np.ndarray:
    """Convert inverse variance to variance.

    ``var = 1 / ivar``.  Pixels with ``ivar <= 0`` are invalid and return NaN.

    Parameters
    ----------
    ivar : array-like
        Inverse variance.

    Returns
    -------
    numpy.ndarray
        Variance; NaN where the input is not a valid IVAR.
    """
    ivar = np.asarray(ivar, dtype=float)
    var = np.full(ivar.shape, np.nan)
    ok = ivar > 0
    var[ok] = 1.0 / ivar[ok]
    return var


def sigma_to_ivar(sigma) -> np.ndarray:
    """Convert standard deviation to inverse variance.

    ``ivar = 1 / sigma**2``.  Non-finite or non-positive sigma (i.e. no
    meaningful error, e.g. masked pixels) maps to ``ivar = 0``, the convention
    survey products use to mark invalid values.

    Parameters
    ----------
    sigma : array-like
        Standard deviation.

    Returns
    -------
    numpy.ndarray
        Inverse variance (always >= 0).
    """
    sigma = np.asarray(sigma, dtype=float)
    ivar = np.zeros(sigma.shape)
    ok = np.isfinite(sigma) & (sigma > 0)
    ivar[ok] = 1.0 / sigma[ok] ** 2
    return ivar


def variance_to_ivar(variance) -> np.ndarray:
    """Convert variance to inverse variance.

    ``ivar = 1 / variance``; invalid (non-finite or <= 0) input maps to
    ``ivar = 0``.

    Parameters
    ----------
    variance : array-like
        Variance.

    Returns
    -------
    numpy.ndarray
        Inverse variance (always >= 0).
    """
    variance = np.asarray(variance, dtype=float)
    ivar = np.zeros(variance.shape)
    ok = np.isfinite(variance) & (variance > 0)
    ivar[ok] = 1.0 / variance[ok]
    return ivar


def as_uncertainty(
    ivar=None, sigma=None, variance=None
) -> StdDevUncertainty | VarianceUncertainty | InverseVarianceUncertainty | None:
    """Build an astropy uncertainty object from a single representation.

    Exactly one of ``ivar`` / ``sigma`` / ``variance`` may be given; passing
    more than one raises ``ValueError``, passing none returns ``None``.

    Returns
    -------
    InverseVarianceUncertainty | StdDevUncertainty | VarianceUncertainty | None
        The corresponding astropy uncertainty container.
    """
    provided = sum(x is not None for x in (ivar, sigma, variance))
    if provided == 0:
        return None
    if provided > 1:
        raise ValueError("provide exactly one of ivar/sigma/variance")
    if ivar is not None:
        return InverseVarianceUncertainty(np.asarray(ivar))
    if sigma is not None:
        return StdDevUncertainty(np.asarray(sigma))
    return VarianceUncertainty(np.asarray(variance))


# --------------------------------------------------------------------------- #
# Analytic propagation primitives.  All operate on IVAR and return
# (value, ivar).  Units are handled by the caller via astropy Quantity.
# --------------------------------------------------------------------------- #


def propagate_sum(values, ivar, axis=None) -> tuple[np.ndarray, np.ndarray]:
    """Sum of values with inverse-variance error propagation.

    Assumes independent Gaussian errors: the variance of the sum is the sum of
    the variances, ``var_sum = sum(1 / ivar_i)``, hence the returned IVAR is
    ``ivar_sum = 1 / var_sum``.

    Masking convention: pixels that are not finite or have ``ivar <= 0`` are
    *excluded* from the sum (they contribute neither value nor error).  When
    ``axis=None`` and no valid pixel remains, the sum is 0.0 and the IVAR 0.0.

    Parameters
    ----------
    values, ivar
        Same-shape arrays of values and inverse variances.
    axis
        Optional numpy axis along which to sum (None = global scalar sum).

    Returns
    -------
    (value, ivar)
        Sum and its inverse variance (same shape as the input when ``axis`` is
        given, scalars otherwise).
    """
    values = np.asarray(values, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    if ivar.shape != values.shape:
        raise ValueError("values and ivar must have the same shape")
    ok = np.isfinite(values) & (ivar > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(ok, 1.0 / np.where(ok, ivar, 1.0), 0.0)
    if axis is None:
        s_flat = float(np.sum(values[ok]))
        var_sum = float(np.sum(inv))
        si_flat = 1.0 / var_sum if var_sum > 0 else 0.0
        return np.array(s_flat), np.array(si_flat)
    s = np.sum(np.where(ok, values, 0.0), axis=axis)
    var_sum = np.sum(inv, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_ivar = np.where(var_sum > 0, 1.0 / var_sum, 0.0)
    return s, s_ivar


def propagate_weighted_mean(values, ivar, axis=None) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance weighted mean and its IVAR.

    ``mean = sum(ivar * value) / sum(ivar)`` with resulting IVAR
    ``wsum = sum(ivar)`` (valid for independent measurements).  Pixels that are
    not finite or have ``ivar <= 0`` are excluded.  Slices without any valid
    weight give ``mean = NaN``, ``ivar = 0``.

    Parameters
    ----------
    values, ivar
        Same-shape arrays of values and inverse variances.
    axis
        Optional numpy axis along which to average.

    Returns
    -------
    (mean, ivar)
        Weighted mean and its inverse variance.
    """
    values = np.asarray(values, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    if ivar.shape != values.shape:
        raise ValueError("values and ivar must have the same shape")
    ok = np.isfinite(values) & (ivar > 0)
    w = np.where(ok, ivar, 0.0)
    wsum = np.sum(w, axis=axis)
    if axis is None:
        wsum = float(wsum)
        if wsum == 0:
            return np.array(np.nan), np.array(0.0)
        mean = float(np.sum(np.where(ok, values, 0.0) * w) / wsum)
        return np.array(mean), np.array(wsum)
    mean = np.sum(np.where(ok, values, 0.0) * w, axis=axis) / np.where(wsum > 0, wsum, np.nan)
    return mean, wsum


def propagate_ratio(a, a_ivar, b, b_ivar) -> tuple[np.ndarray, np.ndarray]:
    """Ratio ``a/b`` with independent Gaussian errors.

    Relative errors add in quadrature:
    ``(sigma_r / r)**2 = (sigma_a / a)**2 + (sigma_b / b)**2`` with ``r = a / b``.

    Pixels where either input is invalid (``ivar <= 0``) or ``b == 0`` give
    ``ratio = NaN``, ``ratio_ivar = 0``.

    Parameters
    ----------
    a, b
        Numerator / denominator values.
    a_ivar, b_ivar
        Inverse variances of ``a`` and ``b``.

    Returns
    -------
    (ratio, ratio_ivar)
        Element-wise ratio and its inverse variance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_ivar = np.asarray(a_ivar, dtype=float)
    b_ivar = np.asarray(b_ivar, dtype=float)
    ok = (a_ivar > 0) & (b_ivar > 0) & (b != 0)
    r = np.full(np.broadcast(a, b).shape, np.nan)
    r_ivar = np.zeros(np.broadcast(a, b).shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel2 = 1.0 / (a_ivar * a**2) + 1.0 / (b_ivar * b**2)
        r_ivar[ok] = 1.0 / (rel2[ok] * (a[ok] / b[ok]) ** 2)
        r[ok] = a[ok] / b[ok]
    return r, r_ivar


def propagate(func, *inputs, method: str = "analytic", **kwargs):
    """Propagation entry point; currently only ``method="analytic"`` exists.

    The interface is fixed now so that downstream code does not change when
    Monte-Carlo / bootstrap propagation (``method="mc"``) is added later.

    Parameters
    ----------
    func
        One of the analytic primitives above, e.g. :func:`propagate_sum`.
    *inputs
        Positional arguments forwarded to ``func``.
    method
        Propagation strategy; anything other than ``"analytic"`` raises
        ``NotImplementedError`` (planned for a later phase).
    **kwargs
        Forwarded to ``func``.

    Returns
    -------
    Whatever ``func`` returns (typically ``(value, ivar)``).
    """
    if method != "analytic":
        raise NotImplementedError(
            f"propagation method {method!r} is planned for later phase"
        )
    return func(*inputs, **kwargs)
