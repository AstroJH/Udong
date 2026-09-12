"""Smoke tests for the viz layer (Agg backend)."""

import numpy as np
import pytest
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


def test_plot_map_scales():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    unit = u.Unit("1e-17 erg/(s cm2 AA)")
    pos = Map2D(value=(np.abs(np.random.default_rng(1).normal(size=(8, 8))) + 0.01) * unit)
    try:
        ax = plot_map(pos, scale="log")
        norm = ax.images[0].norm
        assert isinstance(norm, LogNorm)
        plt.close(ax.figure)

        ax = plot_map(pos, scale="linear")
        assert isinstance(ax.images[0].norm, Normalize)
        plt.close(ax.figure)

        bipolar = Map2D(value=np.linspace(-10, 10, 64).reshape(8, 8) * u.km / u.s)
        ax = plot_map(bipolar, scale="symlog", linthresh=0.1)
        assert isinstance(ax.images[0].norm, SymLogNorm)
        plt.close(ax.figure)
    finally:
        plt.close("all")


def test_plot_map_scale_errors():
    import matplotlib.pyplot as plt

    neg = Map2D(value=np.full((4, 4), -1.0) * u.km / u.s)
    try:
        with pytest.raises(ValueError):
            plot_map(neg, scale="log")   # no positive data
        with pytest.raises(ValueError):
            plot_map(neg, scale="bogus")
    finally:
        plt.close("all")


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
