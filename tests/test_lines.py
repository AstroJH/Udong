import numpy as np
import pytest

from udong.science.spectroscopy.lines import (
    C_KM,
    HA_NII_WINDOW,
    HB_WINDOW,
    OIII_WINDOW,
    SII_WINDOW,
    Line,
    LineWindow,
    bic,
    fit_fixed_kinematics,
    fit_single_and_double,
    fit_window,
)


def synth(window, components, noise=0.0, n=256, seed=0):
    """Build a synthetic spectrum on the window grid from [(v, sigma, {line: flux}), ...]."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(window.wavelength_lo, window.wavelength_hi, n)
    flux = np.zeros(n)
    for v, sig, flx in components:
        for line in window.lines:
            f = flx.get(line.name, 0.0)
            if f <= 0:
                continue
            lam_c = line.wavelength * (1 + v / C_KM)
            sig_aa = sig * lam_c / C_KM
            peak = f / (sig_aa * np.sqrt(2 * np.pi))
            flux += peak * np.exp(-0.5 * ((lam - lam_c) / sig_aa) ** 2)
    ivar = np.full(n, 1.0 / noise ** 2) if noise > 0 else np.full(n, 1e12)
    if noise > 0:
        flux = flux + rng.normal(0, noise, n)
    return lam, flux, ivar


def test_line_window_validation():
    with pytest.raises(ValueError):
        LineWindow("x", 6500, 6600, (Line("A", 6564, 2.0),))
    with pytest.raises(ValueError):
        LineWindow("x", 6500, 6600, (Line("A", 6564, 1.0, "bogus"),))


def test_single_ha_recovery():
    lam, flux, ivar = synth(
        HA_NII_WINDOW,
        [(40.0, 90.0, {"Ha": 100.0, "NII6585": 30.0, "NII6548": 30.0 / 2.96})],
        noise=0.3, seed=1)
    res = fit_window(lam, flux, ivar, HA_NII_WINDOW, n_components=1)
    assert res.success
    c = res.components[0]
    assert abs(c.velocity - 40.0) < 8.0
    assert abs(c.sigma - 90.0) < 10.0
    assert abs(c.fluxes["Ha"] - 100.0) / 100.0 < 0.1
    assert abs(c.fluxes["NII6585"] - 30.0) / 30.0 < 0.15
    assert c.flux_sigma is not None and c.flux_sigma["Ha"] > 0


def test_double_recovery_and_bic():
    comps = [
        (30.0, 80.0, {"Ha": 120.0, "NII6585": 40.0, "NII6548": 40.0 / 2.96}),
        (-60.0, 300.0, {"Ha": 70.0, "NII6585": 18.0, "NII6548": 18.0 / 2.96}),
    ]
    lam, flux, ivar = synth(HA_NII_WINDOW, comps, noise=0.4, seed=3)
    single, double = fit_single_and_double(lam, flux, ivar, HA_NII_WINDOW)
    assert single.success and double.success
    assert single.bic - double.bic > 10.0  # double strongly preferred
    assert len(double.components) == 2
    c0, c1 = double.components  # sorted by sigma
    assert c0.sigma < c1.sigma
    # narrow component near (30, 80), broad near (-60, 300)
    assert abs(c0.velocity - 30.0) < 12.0
    assert abs(c0.sigma - 80.0) < 25.0
    # broad Gaussians constrain v/sigma weakly: use realistic tolerances
    assert abs(c1.velocity + 60.0) < 55.0
    assert abs(c1.sigma - 300.0) < 70.0
    assert abs(c0.fluxes["Ha"] - 120.0) / 120.0 < 0.2
    assert abs(c1.fluxes["Ha"] - 70.0) / 70.0 < 0.35


def test_single_preferred_when_single():
    lam, flux, ivar = synth(
        HA_NII_WINDOW,
        [(10.0, 120.0, {"Ha": 80.0, "NII6585": 20.0, "NII6548": 20.0 / 2.96})],
        noise=1.0, seed=5)
    single, double = fit_single_and_double(lam, flux, ivar, HA_NII_WINDOW)
    assert single.success and double.success
    assert single.bic < double.bic  # no strong preference for a second component


def test_oiii_doublet_ratio():
    lam, flux, ivar = synth(
        OIII_WINDOW,
        [(120.0, 150.0, {"OIII5007": 200.0, "OIII4959": 200.0 / 2.98})],
        noise=0.5, seed=7)
    res = fit_window(lam, flux, ivar, OIII_WINDOW, n_components=1)
    assert res.success
    c = res.components[0]
    assert abs(c.velocity - 120.0) < 10.0
    assert abs(c.sigma - 150.0) < 15.0
    ratio = c.fluxes["OIII4959"] / c.fluxes["OIII5007"]
    assert abs(ratio - 1.0 / 2.98) < 0.02


def test_bic_formula():
    # k ln(n) behaviour
    assert bic(10.0, 2, 100) > bic(10.0, 1, 100)


def test_windows_defined():
    for w in (HA_NII_WINDOW, OIII_WINDOW, HB_WINDOW, SII_WINDOW):
        assert w.lines[0].name


def test_fixed_kinematics_hb():
    lam, flux, ivar = synth(HB_WINDOW, [(30.0, 90.0, {"Hb": 50.0})], noise=0.2, seed=11)
    res = fit_fixed_kinematics(lam, flux, ivar, HB_WINDOW, [(30.0, 90.0)])
    assert res.success
    assert len(res.components) == 1
    assert abs(res.components[0].fluxes["Hb"] - 50.0) / 50.0 < 0.1


def test_fixed_kinematics_sii_doublet():
    lam, flux, ivar = synth(
        SII_WINDOW,
        [(20.0, 100.0, {"SII6718": 30.0, "SII6732": 20.0})],
        noise=0.2, seed=13)
    res = fit_fixed_kinematics(lam, flux, ivar, SII_WINDOW, [(20.0, 100.0)])
    assert res.success
    c = res.components[0]
    assert abs(c.fluxes["SII6718"] - 30.0) / 30.0 < 0.12
    assert abs(c.fluxes["SII6732"] - 20.0) / 20.0 < 0.15


def test_fixed_kinematics_two_components():
    lam, flux, ivar = synth(
        HB_WINDOW,
        [(40.0, 90.0, {"Hb": 60.0}), (-50.0, 300.0, {"Hb": 30.0})],
        noise=0.3, seed=17)
    res = fit_fixed_kinematics(lam, flux, ivar, HB_WINDOW, [(40.0, 90.0), (-50.0, 300.0)])
    assert res.success
    assert len(res.components) == 2
    assert abs(res.components[0].fluxes["Hb"] - 60.0) / 60.0 < 0.15
    assert abs(res.components[1].fluxes["Hb"] - 30.0) / 30.0 < 0.2
