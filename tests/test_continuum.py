"""Tests for the generic pPXF continuum wrapper (survey-agnostic)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from udong.science.spectroscopy.continuum import C_KM, fit_stellar_continuum

ppxf = pytest.importorskip("ppxf.ppxf")


def _log_grid(n=1200, lam0=4000.0, lam1=6000.0):
    ln = np.linspace(np.log(lam0), np.log(lam1), n)
    lam = np.exp(ln)
    velscale = C_KM * (ln[1] - ln[0])
    return lam, velscale


def _toy_star(lam, amp, center, fwhm=6.0):
    """flat continuum with one Gaussian absorption line."""
    spec = np.full_like(lam, 1.0)
    spec -= amp * np.exp(-0.5 * ((lam - center) / (fwhm / 2.355)) ** 2)
    return spec


def test_wrapper_recovers_kinematics_and_model():
    lam, velscale = _log_grid()
    t1 = _toy_star(lam, 0.3, 5000.0, 3.0)
    t2 = _toy_star(lam, 0.5, 5500.0, 5.0)
    templates = np.column_stack([t1, t2])

    # galaxy: template mixture, shifted by +80 km/s and broadened to 120 km/s
    v_true, sig_true = 80.0, 120.0
    shift_pix = v_true / velscale
    x = np.arange(lam.size)
    # interp(x - shift, ...) moves features to *longer* wavelength (+v)
    gal = 0.6 * np.interp(x - shift_pix, x, t1) + 0.4 * np.interp(x - shift_pix, x, t2)
    gal = gaussian_filter1d(gal, sig_true / velscale, mode="nearest")
    rng = np.random.default_rng(42)
    noise_lvl = 0.01
    gal = gal + rng.normal(0, noise_lvl, gal.size)
    ivar = np.full_like(gal, 1.0 / noise_lvl ** 2)

    # restrict the galaxy to the central range so the templates cover it
    # including ppxf's default +/-2900 km/s velocity margin
    use = (lam > 4100) & (lam < 5900)
    # skip a narrow "emission" window to mimic the real usage
    good = np.ones(gal.size, dtype=bool)
    good[(lam > 4990) & (lam < 5010)] = False
    good = good & use

    fit = fit_stellar_continuum(
        lam[use], gal[use], ivar[use], good[use], templates=templates,
        lam_temp=lam, velscale=velscale, start=(0.0, 100.0), degree=6,
    )
    assert np.isfinite(fit.velocity)
    assert abs(fit.velocity - v_true) < 30.0
    assert abs(fit.sigma - sig_true) < 50.0
    assert fit.model.shape == gal[use].shape
    assert fit.residual.shape == gal[use].shape
    # residual rms should be much smaller than the line depth (0.3-0.5)
    assert np.std(fit.residual[good[use]]) < 0.05
