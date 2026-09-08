"""Monte-Carlo and bootstrap uncertainty propagation.

``propagate_mc`` draws independent Gaussian samples for each input (from
per-element sigma or IVAR) and applies an arbitrary function, returning the
full sample distribution plus a median / 16-84% summary.  ``bootstrap``
resamples data with replacement to estimate the sampling distribution of a
statistic.  Both are Quantity-aware.

Design notes
------------
* Errors are assumed independent and Gaussian for ``propagate_mc``; correlated
  measurements are out of scope here.
* The number of samples is explicit and recorded so that results carry
  provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.units import Quantity

from udong.core.uncertainty import ivar_to_sigma

__all__ = [
    "mc_samples",
    "propagate_mc",
    "bootstrap",
    "summarize_mc",
    "MCResult",
]

_DEFAULT_PERCENTILES = (16, 50, 84)


@dataclass
class MCResult:
    """Summary of a Monte-Carlo / bootstrap distribution."""

    median: np.ndarray | Quantity
    sigma: np.ndarray | Quantity  # sample standard deviation
    p16: np.ndarray | Quantity
    p84: np.ndarray | Quantity
    samples: np.ndarray | Quantity | None = None
    n_samples: int = 0

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return (
            f"MCResult(n={self.n_samples}, median={self.median}, "
            f"sigma={self.sigma}, p16={self.p16}, p84={self.p84})"
        )


def _rng(rng=None, seed=None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng(seed)
    return rng


def mc_samples(
    values,
    sigma=None,
    ivar=None,
    n: int = 1000,
    rng=None,
    seed=None,
):
    """Draw ``n`` independent Gaussian samples around ``values``.

    Provide exactly one of ``sigma`` or ``ivar``.  ``values`` may be a
    Quantity (samples keep its unit) or a plain array/scalar.
    """
    if (sigma is None) == (ivar is None):
        raise ValueError("provide exactly one of sigma= or ivar=")
    if sigma is None:
        sigma = ivar_to_sigma(ivar)
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = _rng(rng, seed)

    unit = None
    if isinstance(values, Quantity):
        unit = values.unit
        v = np.asarray(values.value)
        s = np.asarray(Quantity(sigma, unit).value if isinstance(sigma, Quantity) else sigma)
    else:
        v = np.asarray(values)
        s = np.asarray(sigma)
    if s.shape != v.shape:
        raise ValueError("sigma/ivar shape must match values")
    draws = rng.normal(size=(n,) + v.shape)
    samples = v + s * draws
    if unit is not None:
        return Quantity(samples, unit)
    return samples


def summarize_mc(samples, percentiles=_DEFAULT_PERCENTILES) -> MCResult:
    """Median / std / percentile summary over the first (sample) axis."""
    unit = None
    arr = samples
    if isinstance(samples, Quantity):
        unit = samples.unit
        arr = samples.value
    arr = np.asarray(arr)
    lo, med, hi = np.percentile(arr, list(percentiles), axis=0)
    sigma = np.std(arr, axis=0)
    res = MCResult(
        median=med,
        sigma=sigma,
        p16=lo,
        p84=hi,
        samples=samples,
        n_samples=int(arr.shape[0]),
    )
    if unit is not None:
        for attr in ("median", "sigma", "p16", "p84"):
            setattr(res, attr, Quantity(getattr(res, attr), unit))
    return res


def propagate_mc(
    func,
    inputs,
    n: int = 1000,
    rng=None,
    seed=None,
) -> MCResult:
    """Monte-Carlo propagation of ``func`` over uncertain scalar/array inputs.

    Parameters
    ----------
    func
        Callable ``f(*sample_arrays) -> scalar or array``.  All inputs must
        share the same (non-sample) shape.
    inputs
        Sequence of ``(value, sigma)`` tuples; pass ``None`` as the sigma to
        keep an input fixed (no scatter).  Values may be scalars, arrays or
        Quantities (sigma then carries the same unit).
    n, rng, seed
        Sample count and random source (seed used when rng is None).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = _rng(rng, seed)

    sample_arrays: list[np.ndarray] = []
    data_shapes = set()
    for value, sigma in inputs:
        if sigma is None:
            v = np.asarray(value)
            arr = np.broadcast_to(v, (n,) + v.shape)
        else:
            arr = np.asarray(mc_samples(value, sigma=sigma, n=n, rng=rng))
        sample_arrays.append(arr)
        data_shapes.add(arr.shape[1:])
    if len(data_shapes) > 1:
        raise ValueError("all inputs must share the same shape (after scalar broadcast)")

    # evaluate func on each sample
    out = []
    for i in range(n):
        out.append(np.asarray(func(*[a[i] for a in sample_arrays])))
    stacked = np.stack(out)
    return summarize_mc(stacked)


def bootstrap(
    values,
    statistic,
    n: int = 1000,
    rng=None,
    seed=None,
    axis: int = 0,
) -> MCResult:
    """Bootstrap resampling of ``statistic`` over ``values`` (with replacement).

    ``statistic`` is a callable taking a single array (e.g. ``np.median``).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = _rng(rng, seed)
    unit = None
    v = values
    if isinstance(values, Quantity):
        unit = values.unit
        v = values.value
    v = np.asarray(v)
    m = v.shape[axis]
    moved = np.moveaxis(v, axis, 0)
    res = np.empty((n,) + moved.shape[1:])
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        res[i] = statistic(moved[idx])
    out = summarize_mc(Quantity(res, unit) if unit is not None else res)
    return out
