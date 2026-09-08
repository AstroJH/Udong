import numpy as np

from udong.core.uncertainty import (
    ivar_to_sigma,
    ivar_to_variance,
    propagate_ratio,
    propagate_sum,
    propagate_weighted_mean,
    sigma_to_ivar,
    variance_to_ivar,
)


def test_conversions():
    ivar = np.array([1.0, 4.0, 0.0, -1.0])
    sigma = ivar_to_sigma(ivar)
    assert np.allclose(sigma[:2], [1.0, 0.5])
    assert np.isnan(sigma[2:]).all()
    var = ivar_to_variance(ivar)
    assert np.allclose(var[:2], [1.0, 0.25])
    assert np.isnan(var[2:]).all()
    assert np.allclose(sigma_to_ivar([1.0, 2.0, 0.0, np.nan]), [1.0, 0.25, 0.0, 0.0])
    assert np.allclose(variance_to_ivar([1.0, 4.0, 0.0, np.nan]), [1.0, 0.25, 0.0, 0.0])


def test_propagate_sum():
    s, s_ivar = propagate_sum(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
    assert np.isclose(s, 6.0)
    assert np.isclose(s_ivar, 1.0 / 3.0)  # var=3 -> ivar=1/3
    # masked-ish: NaN/zero-ivar values excluded
    s2, s2_ivar = propagate_sum(np.array([1.0, 2.0, np.nan]), np.array([1.0, 0.0, 1.0]))
    assert np.isclose(s2, 1.0)
    assert np.isclose(s2_ivar, 1.0)


def test_propagate_weighted_mean():
    v = np.array([1.0, 3.0, 5.0])
    iv = np.array([1.0, 2.0, 1.0])
    mean, mivar = propagate_weighted_mean(v, iv)
    expected = (1 * 1 + 3 * 2 + 5 * 1) / 4.0
    assert np.isclose(mean, expected)
    assert np.isclose(mivar, 4.0)
    # all invalid -> NaN mean, zero ivar
    m2, i2 = propagate_weighted_mean(np.array([np.nan]), np.array([0.0]))
    assert np.isnan(m2)
    assert i2 == 0.0


def test_propagate_ratio():
    r, r_ivar = propagate_ratio(np.array([4.0]), np.array([1.0]), np.array([2.0]), np.array([4.0]))
    assert np.isclose(r, 2.0)
    # rel^2 = 1/(1*16) + 1/(4*4) = 0.125 -> ivar_r = 1/(rel^2 * r^2) = 1/(0.125*4) = 2.0
    assert np.isclose(r_ivar, 2.0)


def test_propagate_sum_independent_errors():
    """Sum of independent errors: var adds, so ivar_sum = 1/sum(1/ivar)."""
    # two values each with sigma=0.5 (ivar=4): sigma_sum = sqrt(0.5) -> ivar=2
    s, s_ivar = propagate_sum(np.array([1.0, 1.0]), np.array([4.0, 4.0]))
    assert np.isclose(s, 2.0)
    assert np.isclose(s_ivar, 2.0)
    # axis version
    s2, i2 = propagate_sum(np.ones((2, 4)) * 1.0, np.full((2, 4), 4.0), axis=1)
    assert np.allclose(s2, [4.0, 4.0])
    assert np.allclose(i2, 1.0)  # 4 elements, ivar 4 each -> 1/(4*0.25)=1
