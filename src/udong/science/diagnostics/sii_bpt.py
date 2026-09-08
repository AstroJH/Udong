"""[SII]-BPT diagram: separating Seyferts from LINERs.

x = log10([SII] 6716+6731 / Ha)      y = log10([OIII] 5007 / Hb)

Boundaries:

* Kewley et al. (2001, ApJ 556, 121), [SII] starburst line:
  ``y = 0.72 / (x - 0.32) + 1.30``.  Objects below it are star-forming.
* Kewley et al. (2006, MNRAS 372, 961) Seyfert/LINER divider:
  ``y = 1.89 x + 0.76``.  Above the starburst line, objects above this
  divider are Seyferts, below it are LINERs.

The [SII] doublet flux is the sum of [SII] 6716 and 6731 (DAP channels
``SII-6718`` / ``SII-6732``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.units import Quantity

from udong.core.map import Map2D
from udong.core.uncertainty import propagate_sum
from udong.science.diagnostics.bpt import (
    LINER,
    SEYFERT,
    STAR_FORMING,
    UNCLASSIFIED,
    log10_flux_ratio,
)

__all__ = [
    "kewley_2001_sii",
    "kewley_2006_sl",
    "total_sii_flux",
    "classify_sii",
    "sii_bpt_classification",
    "SIIBPTClassification",
]

_KEWLEY_2001_SII = "Kewley+2001, ApJ 556, 121 ([SII] starburst line)"
_KEWLEY_2006 = "Kewley+2006, MNRAS 372, 961 (Seyfert/LINER divider)"


def kewley_2001_sii(x):
    """Kewley+01 maximum-starburst line for the [SII]-BPT diagram."""
    return 0.72 / (x - 0.32) + 1.30


def kewley_2006_sl(x):
    """Kewley+06 Seyfert/LINER divider in the [SII]-BPT diagram."""
    return 1.89 * x + 0.76


def total_sii_flux(sii6716: Map2D, sii6731: Map2D) -> Map2D:
    """``[SII]6716 + [SII]6731`` with IVAR error propagation."""
    if sii6716.shape != sii6731.shape:
        raise ValueError("the two [SII] maps must have the same shape")
    v1, iv1 = np.asarray(sii6716.value.value), np.asarray(sii6716.ivar.value if sii6716.ivar is not None else 0)
    v2, iv2 = np.asarray(sii6731.value.value), np.asarray(sii6731.ivar.value if sii6731.ivar is not None else 0)
    if sii6716.ivar is None or sii6731.ivar is None:
        raise ValueError("both [SII] maps must carry IVAR uncertainties")
    s, s_ivar = propagate_sum(np.stack([v1, v2]), np.stack([iv1, iv2]), axis=0)
    bad = sii6716.bad | sii6731.bad
    return Map2D(
        value=Quantity(s, sii6716.value.unit),
        uncertainty=Quantity(np.where(bad, 0.0, s_ivar)),
        mask=bad,
        wcs=sii6716.wcs,
        meta=dict(sii6716.meta, quantity="[SII]6716+6731"),
        provenance=sii6716.provenance,
    )


def classify_sii(log_sii_ha, log_oiii_hb) -> np.ndarray:
    """[SII]-BPT classification into SF / Seyfert / LINER."""
    x = np.asarray(log_sii_ha, dtype=float)
    y = np.asarray(log_oiii_hb, dtype=float)
    out = np.full(np.broadcast(x, y).shape, UNCLASSIFIED, dtype=int)
    valid = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        sf = valid & (y < kewley_2001_sii(x))
        agn = valid & (y >= kewley_2001_sii(x))
        sey = agn & (y > kewley_2006_sl(x))
        lin = agn & (y <= kewley_2006_sl(x))
    out[sf] = STAR_FORMING
    out[sey] = SEYFERT
    out[lin] = LINER
    return out


@dataclass
class SIIBPTClassification:
    """Result of a [SII]-BPT classification."""

    class_map: Map2D
    log_sii_ha: Map2D
    log_oiii_hb: Map2D
    log_sii_ha_sigma: Map2D | None = None
    log_oiii_hb_sigma: Map2D | None = None
    references: tuple[str, ...] = field(default_factory=lambda: (_KEWLEY_2001_SII, _KEWLEY_2006))

    @property
    def class_id(self) -> np.ndarray:
        return np.asarray(self.class_map.value.value, dtype=int)

    def class_fractions(self) -> dict[str, float]:
        from udong.science.diagnostics.bpt import CLASS_NAMES

        ids = self.class_id
        good = ~self.class_map.bad & (ids != UNCLASSIFIED)
        n = int(np.count_nonzero(good))
        if n == 0:
            return {}
        out = {}
        for code in np.unique(ids[good]):
            out[CLASS_NAMES[int(code)]] = float(np.count_nonzero(good & (ids == code)) / n)
        return out


def sii_bpt_classification(
    sii_flux: Map2D,
    ha_flux: Map2D,
    oiii_flux: Map2D,
    hb_flux: Map2D,
) -> SIIBPTClassification:
    """Full [SII]-BPT pipeline on line-flux maps.

    ``sii_flux`` = [SII]6716+6731 (use :func:`total_sii_flux` on the two DAP
    channels), ``ha_flux`` = Ha, ``oiii_flux`` = [OIII]5007, ``hb_flux`` = Hb.
    """
    log_sii_ha, sig_sii_ha = log10_flux_ratio(sii_flux, ha_flux)
    log_oiii_hb, sig_oiii_hb = log10_flux_ratio(oiii_flux, hb_flux)
    bad = log_sii_ha.bad | log_oiii_hb.bad
    classes = classify_sii(
        np.where(bad, np.nan, log_sii_ha.value.value),
        np.where(bad, np.nan, log_oiii_hb.value.value),
    )
    class_map = Map2D(
        value=Quantity(classes),
        mask=bad,
        wcs=ha_flux.wcs,
        meta=dict(ha_flux.meta, quantity="[SII]-BPT class"),
        provenance=ha_flux.provenance,
    )
    return SIIBPTClassification(
        class_map=class_map,
        log_sii_ha=log_sii_ha,
        log_oiii_hb=log_oiii_hb,
        log_sii_ha_sigma=sig_sii_ha,
        log_oiii_hb_sigma=sig_oiii_hb,
    )
