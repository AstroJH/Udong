"""[OI]-BPT diagram: star-forming / Seyfert / LINER classification.

x = log10([OI] 6300 / Ha)      y = log10([OIII] 5007 / Hb)

Boundaries:

* Kewley et al. (2001, ApJ 556, 121), [OI] maximum-starburst line:
  ``y = 0.73 / (x + 0.59) + 1.33``.  Objects below it are star-forming.
* Kewley et al. (2006, MNRAS 372, 961) Seyfert/LINER divider:
  ``y = 1.18 x + 1.30``.  Above the starburst line, objects above this
  divider are Seyferts, below it are LINERs.

The [OI] line used here is the 6300 A transition (MaNGA DAP channel
``OI-6302``); the fainter 6364 A line (``OI-6365``) is not included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.units import Quantity

from udong.core.map import Map2D
from udong.science.diagnostics.bpt import (
    LINER,
    SEYFERT,
    STAR_FORMING,
    UNCLASSIFIED,
    log10_flux_ratio,
)

__all__ = [
    "kewley_2001_oi",
    "kewley_2006_sl",
    "classify_oi",
    "oi_bpt_classification",
    "OIBPTClassification",
]

_KEWLEY_2001_OI = "Kewley+2001, ApJ 556, 121 ([OI] starburst line)"
_KEWLEY_2006 = "Kewley+2006, MNRAS 372, 961 (Seyfert/LINER divider)"


def kewley_2001_oi(x):
    """Kewley+01 maximum-starburst line for the [OI]-BPT diagram."""
    return 0.73 / (x + 0.59) + 1.33


def kewley_2006_sl(x):
    """Kewley+06 Seyfert/LINER divider in the [OI]-BPT diagram."""
    return 1.18 * x + 1.30


def classify_oi(log_oi_ha, log_oiii_hb) -> np.ndarray:
    """[OI]-BPT classification into SF / Seyfert / LINER."""
    x = np.asarray(log_oi_ha, dtype=float)
    y = np.asarray(log_oiii_hb, dtype=float)
    out = np.full(np.broadcast(x, y).shape, UNCLASSIFIED, dtype=int)
    valid = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        sf = valid & (y < kewley_2001_oi(x))
        agn = valid & (y >= kewley_2001_oi(x))
        sey = agn & (y > kewley_2006_sl(x))
        lin = agn & (y <= kewley_2006_sl(x))
    out[sf] = STAR_FORMING
    out[sey] = SEYFERT
    out[lin] = LINER
    return out


@dataclass
class OIBPTClassification:
    """Result of an [OI]-BPT classification."""

    class_map: Map2D
    log_oi_ha: Map2D
    log_oiii_hb: Map2D
    log_oi_ha_sigma: Map2D | None = None
    log_oiii_hb_sigma: Map2D | None = None
    references: tuple[str, ...] = field(default_factory=lambda: (_KEWLEY_2001_OI, _KEWLEY_2006))

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


def oi_bpt_classification(
    oi_flux: Map2D,
    ha_flux: Map2D,
    oiii_flux: Map2D,
    hb_flux: Map2D,
) -> OIBPTClassification:
    """Full [OI]-BPT pipeline on line-flux maps.

    ``oi_flux`` = [OI]6300 (DAP channel ``OI-6302``), ``ha_flux`` = Ha,
    ``oiii_flux`` = [OIII]5007, ``hb_flux`` = Hb.
    """
    log_oi_ha, sig_oi_ha = log10_flux_ratio(oi_flux, ha_flux)
    log_oiii_hb, sig_oiii_hb = log10_flux_ratio(oiii_flux, hb_flux)
    bad = log_oi_ha.bad | log_oiii_hb.bad
    classes = classify_oi(
        np.where(bad, np.nan, log_oi_ha.value.value),
        np.where(bad, np.nan, log_oiii_hb.value.value),
    )
    class_map = Map2D(
        value=Quantity(classes),
        mask=bad,
        wcs=ha_flux.wcs,
        meta=dict(ha_flux.meta, quantity="OI-BPT class"),
        provenance=ha_flux.provenance,
    )
    return OIBPTClassification(
        class_map=class_map,
        log_oi_ha=log_oi_ha,
        log_oiii_hb=log_oiii_hb,
        log_oi_ha_sigma=sig_oi_ha,
        log_oiii_hb_sigma=sig_oiii_hb,
    )
