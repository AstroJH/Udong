"""Masked statistics with IVAR-based error propagation."""

from __future__ import annotations

import numpy as np
from udong.core.uncertainty import propagate_sum, propagate_weighted_mean

__all__ = [
    "weighted_mean",
    "sum_with_uncertainty",
    "percentile",
    "sigma_clip",
    "snr",
]


def _good(values, ivar, mask):
    good = np.isfinite(values)
    if ivar is not None:
        good &= np.asarray(ivar) > 0
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool)
    return good


def weighted_mean(values, ivar, mask=None, axis=None):
    """Inverse-variance weighted mean -> (mean, mean_ivar)."""
    values = np.asarray(values, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    m = None if mask is None else ~np.asarray(mask, dtype=bool)
    ok = np.isfinite(values) & (ivar > 0)
    if m is not None:
        ok &= m
    v = np.where(ok, values, np.nan)
    i = np.where(ok, ivar, 0.0)
    return propagate_weighted_mean(v, i, axis=axis)


def sum_with_uncertainty(values, ivar, mask=None, axis=None):
    """Masked sum with IVAR propagation -> (sum, sum_ivar)."""
    values = np.asarray(values, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    ok = np.isfinite(values) & (ivar > 0)
    if mask is not None:
        ok &= ~np.asarray(mask, dtype=bool)
    v = np.where(ok, values, np.nan)
    i = np.where(ok, ivar, 0.0)
    return propagate_sum(v, i, axis=axis)


def percentile(values, q, mask=None, axis=None):
    """Percentile(s) ignoring masked/invalid values."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if mask is not None:
        ok &= ~np.asarray(mask, dtype=bool)
    if axis is None:
        return np.percentile(values[ok], q)
    # generic axis handling: operate on flattened along-axis slices
    return np.apply_along_axis(
        lambda a: np.percentile(a[np.isfinite(a)], q), axis, np.where(ok, values, np.nan)
    )


def sigma_clip(values, mask=None, sigma=3.0, maxiters=5, axis=None):
    """Iterative sigma clipping; returns (values, bad_mask).

    ``bad_mask`` is True for pixels rejected by clipping (or already masked).
    Uses a median/MAD robust estimator per clip iteration.
    """
    from astropy.stats import mad_std
    from astropy.stats import sigma_clip as astropy_sigma_clip

    values = np.asarray(values, dtype=float)
    bad = np.zeros(values.shape, dtype=bool)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)
    data = np.where(bad, np.nan, values)
    clipped = astropy_sigma_clip(data, sigma=sigma, maxiters=maxiters, axis=axis, cenfunc="median", stdfunc=mad_std)
    return clipped, clipped.mask
    # NOTE: returns the astropy masked array; `.data` and `.mask` hold results


def snr(values, ivar):
    """Signal-to-noise ``values * sqrt(ivar)`` (NaN where ivar<=0)."""
    values = np.asarray(values, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ivar > 0, values * np.sqrt(ivar), np.nan)
