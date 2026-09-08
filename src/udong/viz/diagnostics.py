"""Diagnostic-diagram plotting (matplotlib thin wrappers)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from udong.science.diagnostics.bpt import (
    AGN,
    CLASS_NAMES,
    COMPOSITE,
    STAR_FORMING,
    UNCLASSIFIED,
    BPTClassification,
    kauffmann_2003_nii,
    kewley_2001_nii,
)

__all__ = ["plot_bpt"]

_CLASS_COLORS = {
    STAR_FORMING: "#2ca02c",
    COMPOSITE: "#ff7f0e",
    AGN: "#d62728",
}


def plot_bpt(result: BPTClassification, ax=None, sample: int | None = 2000, **kwargs):
    """Scatter of the [NII]-BPT diagram colored by class, with boundaries.

    ``sample`` optionally limits the number of plotted (good) points.
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.asarray(result.log_nii_ha.value.value)
    y = np.asarray(result.log_oiii_hb.value.value)
    good = ~(result.log_nii_ha.bad | result.log_oiii_hb.bad)
    ids = result.class_id
    good &= ids != UNCLASSIFIED
    xi, yi, ci = x[good], y[good], ids[good]
    if sample is not None and len(xi) > sample:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xi), size=sample, replace=False)
        xi, yi, ci = xi[sel], yi[sel], ci[sel]
    for code, color in _CLASS_COLORS.items():
        m = ci == code
        if m.any():
            ax.scatter(xi[m], yi[m], s=6, c=color, label=CLASS_NAMES[code], **kwargs)
    xs = np.linspace(-1.5, 0.5, 300)
    with np.errstate(invalid="ignore", divide="ignore"):
        ax.plot(xs, kewley_2001_nii(xs), "k-", lw=1.2, label="Kewley+01")
        ax.plot(xs, kauffmann_2003_nii(xs), "k--", lw=1.0, label="Kauffmann+03")
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(-1.2, 1.5)
    ax.set_xlabel(r"$\log_{10}([\mathrm{N\,II}]6584/\mathrm{H}\alpha)$")
    ax.set_ylabel(r"$\log_{10}([\mathrm{O\,III}]5007/\mathrm{H}\beta)$")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return ax
