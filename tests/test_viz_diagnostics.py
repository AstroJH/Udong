"""Smoke tests for diagnostic-diagram plotting (Agg)."""

import numpy as np
from astropy import units as u

from udong.core.map import Map2D
from udong.science.diagnostics.bpt import nii_bpt_classification
from udong.viz import plot_bpt


def _flux(fill):
    return Map2D(
        value=np.full((6, 6), fill) * 1e-17 * u.erg / (u.s * u.cm**2),
        uncertainty=np.full((6, 6), 1e4) / (u.erg / (u.s * u.cm**2)) ** 2,
        mask=np.zeros((6, 6), dtype=bool),
    )


def test_plot_bpt():
    ha = _flux(100.0)
    hb = _flux(30.0)
    nii = _flux(10.0)
    oiii = _flux(60.0)
    res = nii_bpt_classification(oiii, hb, nii, ha)
    ax = plot_bpt(res)
    assert ax is not None
