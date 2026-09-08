import numpy as np
import pytest
from astropy import units as u

from udong.science.uncertainty.montecarlo import (
    bootstrap,
    mc_samples,
    propagate_mc,
    summarize_mc,
)


def test_mc_samples_matches_input_distribution():
    samples = mc_samples(np.array([10.0, 100.0]), sigma=np.array([1.0, 10.0]),
                         n=200_000, seed=42)
    assert samples.shape == (200_000, 2)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    assert np.allclose(mean, [10.0, 100.0], atol=0.05)
    assert np.allclose(std, [1.0, 10.0], atol=0.1)


def test_mc_samples_quantity_and_ivar():
    v = np.array([2.0]) * u.km / u.s
    s = mc_samples(v, ivar=np.array([1.0]) / (u.km / u.s) ** 2, n=5000, seed=1)
    assert isinstance(s, u.Quantity)
    assert s.unit == u.km / u.s
    assert np.isclose(s.mean().to_value(u.km / u.s), 2.0, atol=0.1)


def test_mc_samples_requires_one_uncertainty():
    with pytest.raises(ValueError):
        mc_samples(np.array([1.0]), n=10)
    with pytest.raises(ValueError):
        mc_samples(np.array([1.0]), sigma=0.1, ivar=1.0, n=10)


def test_propagate_mc_ratio():
    # ratio of two independent gaussians, a=4+-1, b=2+-0.5
    res = propagate_mc(lambda a, b: a / b, [(4.0, 1.0), (2.0, 0.5)],
                       n=200_000, seed=42)
    assert np.isclose(res.median, 2.0, atol=0.05)
    # (p84-p16)/2 ~ sigma_r ~ 0.3536*ratio (small-angle approximation);
    # use percentiles because the ratio of Gaussians has heavy tails
    half_range = (res.p84 - res.p16) / 2.0
    assert np.isclose(half_range / res.median, 0.3536, atol=0.15)


def test_propagate_mc_fixed_input():
    res = propagate_mc(lambda a, b: a + b, [(1.0, None), (2.0, 0.5)],
                       n=50_000, seed=7)
    assert np.isclose(res.median, 3.0, atol=0.05)


def test_bootstrap_median():
    data = np.random.default_rng(0).normal(5.0, 1.0, size=2000)
    res = bootstrap(data, np.median, n=2000, seed=3)
    assert np.isclose(res.median, 5.0, atol=0.1)
    assert res.sigma > 0


def test_summarize_mc_repr():
    samples = np.random.default_rng(0).normal(0, 1, (100, 3))
    res = summarize_mc(samples)
    assert res.n_samples == 100
    assert res.samples.shape == (100, 3)
