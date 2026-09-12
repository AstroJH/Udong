"""Tests for WCS-aware image overlays (`udong.viz.overlay`)."""

import numpy as np
import pytest
from astropy import units as u
from astropy.wcs import WCS

from udong.core.map import Map2D
from udong.viz import (
    contour_levels,
    overlay_contours,
    overlay_map,
    plot_map,
    plot_overlay,
    zoom_to,
)

FLUX = u.Unit("1e-17 erg/(s cm2 AA)")
VEL = u.km / u.s


def _wcs(n=8, crval=(150.0, 2.0), scale=1e-4):
    w = WCS(naxis=2)
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.crval = list(crval)
    w.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_contour_levels():
    m = Map2D(value=np.linspace(-10, 10, 64).reshape(8, 8) * VEL)
    lv = contour_levels(m, nlevels=9, symmetric=True)
    assert len(lv) == 9
    assert np.isclose(lv[0], -lv[-1])
    lv2 = contour_levels(m, nlevels=5, percentiles=(10, 90))
    assert len(lv2) == 5 and lv2[0] < 0 < lv2[-1]
    with pytest.raises(ValueError):
        contour_levels(m, nlevels=1)


def test_overlay_contours_same_grid():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    base = Map2D(value=rng.random((8, 8)) * FLUX, wcs=_wcs())
    over = Map2D(value=rng.normal(size=(8, 8)) * 50 * VEL, wcs=_wcs())
    fig = plt.figure()
    try:
        ax = plot_map(base, colorbar=False)
        overlay_contours(over, ax, symmetric=True, colorbar=True)
        assert len(ax.collections) >= 1
    finally:
        plt.close("all")


def test_overlay_contours_different_wcs():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    base = Map2D(value=rng.random((8, 8)) * FLUX, wcs=_wcs(crval=(150.0, 2.0)))
    over = Map2D(value=rng.normal(size=(8, 8)) * 50 * VEL,
                 wcs=_wcs(crval=(150.001, 2.001), scale=2e-4))
    try:
        ax = plot_map(base, colorbar=False)
        overlay_contours(over, ax, symmetric=True)
        assert len(ax.collections) >= 1
        # contours of a different WCS must use a WCS transform, not transData
        assert ax.collections[0].get_transform() is not ax.transData
    finally:
        plt.close("all")


def test_overlay_map_reprojects_to_base_grid():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    base = Map2D(value=rng.random((8, 8)) * FLUX, wcs=_wcs(n=8))
    # 16 px at HALF the base pixel scale -> smaller footprint than the base
    over = Map2D(value=rng.normal(size=(16, 16)) * 50 * VEL,
                 wcs=_wcs(n=16, crval=(150.0, 2.0), scale=2.5e-5))
    try:
        ax = plot_map(base, colorbar=False)
        overlay_map(over, ax, symmetric=True, alpha=0.5)
        arr = np.asarray(ax.images[-1].get_array())
        assert arr.shape == (8, 8)
        # reprojection leaves pixels outside the overlay footprint as NaN
        assert np.isnan(arr).any() and np.isfinite(arr).any()
    finally:
        plt.close("all")


def test_plot_overlay_convenience_and_shape_error():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    base = Map2D(value=rng.random((8, 8)) * FLUX, wcs=_wcs())
    over = Map2D(value=rng.normal(size=(8, 8)) * 50 * VEL, wcs=_wcs())
    try:
        ax = plot_overlay(base, over, filled=True, contours=True,
                          fill_kwargs={"symmetric": True, "alpha": 0.5},
                          contour_kwargs={"symmetric": True})
        assert len(ax.images) >= 2 and len(ax.collections) >= 1
        plt.close("all")

        # caller-supplied *empty* axes: base must still be drawn
        fig, ax0 = plt.subplots()
        ax0 = plot_overlay(base, over, ax=ax0, filled=True,
                           fill_kwargs={"symmetric": True},
                           contour_kwargs={"symmetric": True})
        assert len(ax0.images) >= 2 and len(ax0.collections) >= 1
        plt.close("all")

        # no-WCS overlay with a different shape cannot be aligned
        bad = Map2D(value=rng.normal(size=(4, 4)) * 50 * VEL, wcs=None)
        ax2 = plot_map(base, colorbar=False)
        with pytest.raises(ValueError):
            overlay_map(bad, ax2)
    finally:
        plt.close("all")


def test_plot_map_xlim_ylim():
    import matplotlib.pyplot as plt

    m = Map2D(value=np.random.default_rng(4).random((20, 30)) * FLUX, wcs=_wcs(n=20))
    try:
        ax = plot_map(m, colorbar=False, xlim=(5, 15), ylim=(8, 18))
        assert ax.get_xlim() == (5.0, 15.0)
        assert ax.get_ylim() == (8.0, 18.0)
    finally:
        plt.close("all")


def test_zoom_to_overlay_footprint():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(5)
    base = Map2D(value=rng.random((8, 8)) * FLUX, wcs=_wcs(n=8, scale=1e-4))
    # 16 px at 2.5e-5 deg/px -> 4 base pixels across
    over = Map2D(value=rng.normal(size=(16, 16)) * 50 * VEL,
                 wcs=_wcs(n=16, crval=(150.0, 2.0), scale=2.5e-5))
    try:
        ax = plot_map(base, colorbar=False)
        zoom_to(over, ax, margin=0.0)
        x0, x1 = ax.get_xlim()
        assert 0 < (x1 - x0) < 8          # zoomed in compared with the 8-px base
        assert abs((x0 + x1) / 2 - 3.5) < 1.0   # centred on the base image

        ax2 = plot_overlay(base, over, zoom=True, filled=True,
                           fill_kwargs={"symmetric": True},
                           contour_kwargs={"symmetric": True})
        x0, x1 = ax2.get_xlim()
        assert 0 < (x1 - x0) < 8
    finally:
        plt.close("all")


def test_overlay_fill_colorbar_scale_and_explicit_limits():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    rng = np.random.default_rng(6)
    base = Map2D(value=rng.random((8, 8)) * FLUX, wcs=_wcs())
    flux = Map2D(value=(rng.random((8, 8)) * 99 + 1) * FLUX, wcs=_wcs())
    fig, ax = plt.subplots()
    try:
        plot_map(base, ax=ax, colorbar=False)
        overlay_map(flux, ax, cmap="magma", scale="log", vmin=2.0, vmax=50.0,
                    colorbar=True, colorbar_label="flux")
        im = ax.images[-1]
        assert isinstance(im.norm, LogNorm)
        assert np.isclose(im.norm.vmin, 2.0) and np.isclose(im.norm.vmax, 50.0)
        assert len(fig.axes) >= 2          # an extra axes holds the colourbar
    finally:
        plt.close("all")


def test_overlay_fill_symmetric_and_plot_map_symmetric():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    base = Map2D(value=rng.random((8, 8)) * FLUX, wcs=_wcs())
    vel = Map2D(value=rng.normal(size=(8, 8)) * 100 * VEL, wcs=_wcs())
    try:
        ax = plot_map(base, colorbar=False)
        overlay_map(vel, ax, symmetric=True, colorbar=True)
        norm = ax.images[-1].norm
        assert np.isclose(norm.vmin, -norm.vmax)
        plt.close("all")

        ax2 = plot_map(vel, symmetric=True, colorbar=False)
        norm2 = ax2.images[0].norm
        assert np.isclose(norm2.vmin, -norm2.vmax)
    finally:
        plt.close("all")
