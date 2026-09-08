"""BPT (Baldwin-Phillips-Terlevich) ionization diagnostics.

Classifies emission-line regions as star-forming / composite / AGN from the
[NII]-BPT diagram:

    x = log10([NII] 6584 / Ha)      y = log10([OIII] 5007 / Hb)

The classification boundaries are **explicitly referenced** module constants
(never hidden in code):

* Kauffmann et al. (2003, MNRAS 346, 1055): the empirical upper envelope of
  star-forming galaxies,
  ``y = 0.61 / (x - 0.05) + 1.30``.
* Kewley et al. (2001, ApJ 556, 121): the theoretical maximum-starburst line,
  ``y = 0.61 / (x - 0.47) + 1.19``.
* Cid Fernandes et al. (2010, MNRAS 403, 1036): empirical Seyfert/LINER
  divider used to split the AGN wedge,
  ``y = 1.05 x + 0.45`` (optional; pass ``agn_split`` to ``classify_nii``).

Objects below the Kauffmann line are ``STAR_FORMING``; between Kauffmann and
Kewley are ``COMPOSITE`` (hybrid SF + AGN); above the Kewley line are ``AGN``.

These diagnostics operate on the survey-agnostic ``Map2D`` type; to build the
four line-flux maps from MaNGA DAP MAPS, use
``MangaMaps.emission_line_flux(...)`` (in ``udong.data.manga``).

References/assumptions
----------------------
* Ratios use Gaussian (independent) error propagation on the fluxes, then
  ``sigma_log10 = sigma_ratio / (ratio * ln10)``.
* The physical convention is that the [NII] line is 6584 A and [OIII] 5007 A;
  the DAP channel names are ``NII-6585`` and ``OIII-5008``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.units import Quantity

from udong.core.map import Map2D
from udong.core.uncertainty import ivar_to_sigma, propagate_ratio

__all__ = [
    "UNCLASSIFIED",
    "STAR_FORMING",
    "COMPOSITE",
    "AGN",
    "SEYFERT",
    "LINER",
    "AGN_STRONG",
    "AGN_WEAK",
    "RETIRED",
    "CLASS_NAMES",
    "kauffmann_2003_nii",
    "kewley_2001_nii",
    "cid_fernandes_2010_sl",
    "log10_flux_ratio",
    "classify_nii",
    "nii_bpt_classification",
    "BPTClassification",
]

# --------------------------------------------------------------------------- #
# Class codes and names
# --------------------------------------------------------------------------- #
UNCLASSIFIED = 0
STAR_FORMING = 1
COMPOSITE = 2
AGN = 3
SEYFERT = 4
LINER = 5
AGN_STRONG = 6  # WHAN: Seyfert-like (EW(Ha) >= 6 A)
AGN_WEAK = 7  # WHAN: LINER-like (3 <= EW(Ha) < 6 A)
RETIRED = 8  # WHAN: EW(Ha) < 3 A

CLASS_NAMES: dict[int, str] = {
    UNCLASSIFIED: "unclassified",
    STAR_FORMING: "star-forming",
    COMPOSITE: "composite",
    AGN: "AGN",
    SEYFERT: "seyfert",
    LINER: "liner",
    AGN_STRONG: "strong-AGN",
    AGN_WEAK: "weak-AGN",
    RETIRED: "retired",
}

# Boundary references (kept explicit for provenance).
_KAUFMANN_2003 = "Kauffmann+2003, MNRAS 346, 1055"
_KEWLEY_2001 = "Kewley+2001, ApJ 556, 121"


def kauffmann_2003_nii(x):
    """Kauffmann+03 empirical upper envelope of star-forming galaxies."""
    return 0.61 / (x - 0.05) + 1.30


def kewley_2001_nii(x):
    """Kewley+01 theoretical maximum-starburst line."""
    return 0.61 / (x - 0.47) + 1.19


def cid_fernandes_2010_sl(x):
    """Cid Fernandes+10 empirical Seyfert/LINER divider in the [NII]-BPT.

    ``y = 1.05 x + 0.45``; adopted with essentially identical parameters by
    Schawinski et al. (2007, ApJ 667, 852) and used e.g. as the dotted line
    in Fig. 9(a) of Cao+ (2026, arXiv:2606.24211, their ref. [20]).
    """
    return 1.05 * x + 0.45


# --------------------------------------------------------------------------- #
# Log flux ratios with error propagation
# --------------------------------------------------------------------------- #
def log10_flux_ratio(numerator: Map2D, denominator: Map2D) -> tuple[Map2D, Map2D]:
    """``log10(numerator/denominator)`` and its 1-sigma uncertainty.

    Both maps must carry inverse-variance uncertainties; the ratio uses
    independent-Gaussian propagation.  Pixels bad in either map (or with
    non-positive flux) are flagged bad in the output maps.
    """
    if numerator.shape != denominator.shape:
        raise ValueError("numerator and denominator must have the same shape")
    if numerator.ivar is None or denominator.ivar is None:
        raise ValueError("both maps must carry IVAR uncertainties")
    n = np.asarray(numerator.value.value)
    d = np.asarray(denominator.value.value)
    n_ivar = np.asarray(numerator.ivar.value)
    d_ivar = np.asarray(denominator.ivar.value)
    bad = numerator.bad | denominator.bad | (n <= 0) | (d <= 0) | (n_ivar <= 0) | (d_ivar <= 0)

    ratio, ratio_ivar = propagate_ratio(n, n_ivar, d, d_ivar)
    log_ratio = np.log10(np.where(ratio > 0, ratio, np.nan))
    # sigma_log10 = sigma_ratio/(ratio*ln10) ; sigma_ratio = 1/sqrt(ratio_ivar)
    sigma_ratio = ivar_to_sigma(ratio_ivar)
    sigma_log = np.where(ratio > 0, sigma_ratio / (ratio * np.log(10.0)), np.nan)

    meta = dict(numerator.meta)
    meta["quantity"] = f"log10({numerator.meta.get('quantity', 'num')}/{denominator.meta.get('quantity', 'den')})"
    prov = None
    log_map = Map2D(
        value=Quantity(log_ratio),
        uncertainty=None,
        mask=bad,
        wcs=numerator.wcs,
        meta=meta,
        provenance=prov,
    )
    sigma_map = Map2D(
        value=Quantity(np.where(bad, np.nan, sigma_log)),
        uncertainty=None,
        mask=bad,
        wcs=numerator.wcs,
        meta=meta,
    )
    return log_map, sigma_map


def log10_value(values: Map2D) -> tuple[Map2D, Map2D]:
    """``log10(values)`` and its 1-sigma uncertainty from the IVAR.

    ``sigma_log10 = sigma_value / (value * ln10)``.  Non-positive or bad
    pixels are flagged bad.
    """
    if values.ivar is None:
        raise ValueError("map must carry IVAR uncertainties")
    v = np.asarray(values.value.value)
    iv = np.asarray(values.ivar.value)
    bad = values.bad | (v <= 0) | (iv <= 0)
    logv = np.log10(np.where(v > 0, v, np.nan))
    sigma = ivar_to_sigma(iv)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_log = np.where(v > 0, sigma / (v * np.log(10.0)), np.nan)
    meta = dict(values.meta, quantity=f"log10({values.meta.get('quantity', 'value')})")
    log_map = Map2D(value=Quantity(logv), mask=bad, wcs=values.wcs, meta=meta)
    sig_map = Map2D(value=Quantity(np.where(bad, np.nan, sigma_log)), mask=bad, wcs=values.wcs, meta=meta)
    return log_map, sig_map


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify_nii(
    log_nii_ha,
    log_oiii_hb,
    *,
    allow_composite: bool = True,
    agn_split=None,
) -> np.ndarray:
    """Classify spaxels on the [NII]-BPT diagram.

    Parameters
    ----------
    log_nii_ha, log_oiii_hb
        Arrays of log10 line ratios.  NaN/inf values are left ``UNCLASSIFIED``.
    allow_composite
        If False, returns only star-forming / AGN (the Kewley line is the
        only boundary).
    agn_split
        Optional callable ``y = f(x)`` that splits the AGN wedge into Seyfert
        (above the line) and LINER (below), e.g.
        :func:`cid_fernandes_2010_sl`.  If None, all AGN pixels are returned
        with the generic ``AGN`` code.
    """
    x = np.asarray(log_nii_ha, dtype=float)
    y = np.asarray(log_oiii_hb, dtype=float)
    out = np.full(np.broadcast(x, y).shape, UNCLASSIFIED, dtype=int)
    valid = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        if allow_composite:
            sf = valid & (y < kauffmann_2003_nii(x))
            comp = valid & (y >= kauffmann_2003_nii(x)) & (y < kewley_2001_nii(x))
            agn = valid & (y >= kewley_2001_nii(x))
        else:
            sf = valid & (y < kewley_2001_nii(x))
            comp = np.zeros(valid.shape, dtype=bool)
            agn = valid & (y >= kewley_2001_nii(x))
    out[sf] = STAR_FORMING
    out[comp] = COMPOSITE
    if agn_split is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            sey = agn & (y > agn_split(x))
            lin = agn & ~sey
        out[sey] = SEYFERT
        out[lin] = LINER
    else:
        out[agn] = AGN
    return out


@dataclass
class BPTClassification:
    """Result of an [NII]-BPT classification."""

    class_map: Map2D
    log_nii_ha: Map2D
    log_oiii_hb: Map2D
    log_nii_ha_sigma: Map2D | None = None
    log_oiii_hb_sigma: Map2D | None = None
    allow_composite: bool = True
    references: tuple[str, ...] = field(
        default_factory=lambda: (_KAUFMANN_2003, _KEWLEY_2001)
    )

    @property
    def class_id(self) -> np.ndarray:
        return np.asarray(self.class_map.value.value, dtype=int)

    def class_fractions(self) -> dict[str, float]:
        """Fraction of classified pixels in each class (excludes bad)."""
        ids = self.class_id
        good = ~self.class_map.bad & (ids != UNCLASSIFIED)
        n = int(np.count_nonzero(good))
        if n == 0:
            return {}
        out = {}
        for code in np.unique(ids[good]):
            out[CLASS_NAMES[int(code)]] = float(np.count_nonzero(good & (ids == code)) / n)
        return out


def nii_bpt_classification(
    oiii_flux: Map2D,
    hb_flux: Map2D,
    nii_flux: Map2D,
    ha_flux: Map2D,
    *,
    allow_composite: bool = True,
    agn_split=None,
) -> BPTClassification:
    """Full [NII]-BPT pipeline on four line-flux ``Map2D``.

    ``oiii_flux`` = [OIII]5007, ``hb_flux`` = Hb, ``nii_flux`` = [NII]6584,
    ``ha_flux`` = Ha (e.g. from ``MangaMaps.emission_line_flux``).
    ``agn_split`` is passed to :func:`classify_nii`; pass
    :func:`cid_fernandes_2010_sl` to split AGN into Seyfert/LINER.
    """
    log_nii_ha, sig_nii_ha = log10_flux_ratio(nii_flux, ha_flux)
    log_oiii_hb, sig_oiii_hb = log10_flux_ratio(oiii_flux, hb_flux)
    bad = log_nii_ha.bad | log_oiii_hb.bad
    classes = classify_nii(
        np.where(bad, np.nan, log_nii_ha.value.value),
        np.where(bad, np.nan, log_oiii_hb.value.value),
        allow_composite=allow_composite,
        agn_split=agn_split,
    )
    class_map = Map2D(
        value=Quantity(classes),
        mask=bad,
        wcs=ha_flux.wcs,
        meta=dict(ha_flux.meta, quantity="BPT class"),
        provenance=ha_flux.provenance,
    )
    return BPTClassification(
        class_map=class_map,
        log_nii_ha=log_nii_ha,
        log_oiii_hb=log_oiii_hb,
        log_nii_ha_sigma=sig_nii_ha,
        log_oiii_hb_sigma=sig_oiii_hb,
        allow_composite=allow_composite,
    )
