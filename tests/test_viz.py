"""Smoke tests for the viz layer (Agg backend)."""

import numpy as np
from astropy import units as u

from udong.core.map import Map2D
from udong.core.spectrum import Spectrum
from udong.science.spatial import RadialProfile
from udong.viz import plot_map, plot_profile, plot_spectrum


def test_plot_spectrum():
    wave = np.linspace(4000, 5000, 50) * u.AA
    flux = np.ones(50) * 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    s = Spectrum(flux=flux, spectral_axis=wave)
    ax = plot_spectrum(s)
    assert ax is not None


def test_plot_map():
    m = Map2D(value=np.random.default_rng(0).normal(size=(8, 8)) * u.km / u.s)
    ax = plot_map(m)
    assert ax is not None


def test_plot_profile():
    p = RadialProfile(radius=np.array([1.0, 2.0]) * u.arcsec,
                      value=np.array([1.0, 2.0]) * u.km / u.s)
    ax = plot_profile(p)
    assert ax is not None


def test_plot_map_plain_axes_shows_sky_ticks():
    """plot_map on a plain (non-WCS) axes must label ticks in sky coords."""
    import matplotlib.pyplot as plt
    from astropy.wcs import WCS

    hdr = {
        "CRPIX1": 4.0, "CRPIX2": 4.0, "CRVAL1": 232.5447, "CRVAL2": 48.690201,
        "CD1_1": -0.000138889, "CD2_2": 0.000138889,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg", "CUNIT2": "deg",
    }
    w = WCS(hdr)
    w.wcs.set()
    m = Map2D(value=np.random.default_rng(0).normal(size=(8, 8)) * u.km / u.s,
              wcs=w)
    fig, ax = plt.subplots()
    try:
        plot_map(m, ax=ax, colorbar=False)
        xlabels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        ylabels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        # sky labels are decimal degrees, not bare pixel indices
        assert xlabels and ylabels
        assert all("." in s for s in xlabels + ylabels)
        assert ax.get_xlabel() == "RA [deg]"
    finally:
        plt.close(fig)
