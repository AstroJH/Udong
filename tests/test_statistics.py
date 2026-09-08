import numpy as np

from udong.science.statistics import (
    percentile,
    sigma_clip,
    snr,
    sum_with_uncertainty,
    weighted_mean,
)


def test_weighted_mean():
    v = np.array([1.0, 2.0, 3.0, np.nan])
    iv = np.array([1.0, 1.0, 1.0, 1.0])
    mean, mivar = weighted_mean(v, iv)
    assert np.isclose(mean, 2.0)
    assert np.isclose(mivar, 3.0)


def test_sum():
    s, siv = sum_with_uncertainty(np.array([1.0, 2.0, np.nan]), np.array([1.0, 0.0, 1.0]))
    assert np.isclose(s, 1.0)
    assert np.isclose(siv, 1.0)


def test_percentile_masked():
    v = np.arange(10.0)
    mask = np.zeros(10, dtype=bool)
    mask[0] = True
    assert np.isclose(percentile(v, 50, mask=mask), 5.0)


def test_sigma_clip():
    v = np.array([1.0, 1.1, 0.9, 1.05, 50.0])
    clipped, bad = sigma_clip(v)
    assert bad[4]
    assert not bad[:4].any()


def test_snr():
    s = snr(np.array([2.0]), np.array([4.0]))
    assert np.isclose(s, 4.0)
    assert np.isnan(snr(np.array([1.0]), np.array([0.0])))
