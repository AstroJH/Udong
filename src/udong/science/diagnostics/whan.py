"""WHAN diagram (Cid Fernandes et al. 2010, 2011).

The WHAN diagram replaces the [OIII]/Hb axis of the BPT with the H-alpha
*equivalent width*, plotting

    x = EW(Ha)  (rest frame, Angstrom)      y = log10([NII] 6584 / Ha)

It separates *retired* (passive) galaxies - whose weak emission is ionized by
old stellar populations rather than star formation or an AGN - from genuine
AGN, and splits AGN into strong (Seyfert-like) and weak (LINER-like):

* ``log10([NII]/Ha) < -0.4``  -> star-forming;
* ``EW(Ha) < 3 A``            -> retired (unless the line ratio already said SF);
* ``3 <= EW(Ha) < 6 A``       -> weak AGN (LINER-like);
* ``EW(Ha) >= 6 A``           -> strong AGN (Seyfert-like).

References: Cid Fernandes et al. 2010, MNRAS 403, 1036; 2011, MNRAS 413,
1687.

The equivalent width must be in the galaxy *rest frame*: the DAP GEW is
measured on observed-frame spectra, so divide by ``(1 + z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.units import Quantity

from udong.core.map import Map2D
from udong.science.diagnostics.bpt import (
    AGN_STRONG,
    AGN_WEAK,
    RETIRED,
    STAR_FORMING,
    UNCLASSIFIED,
    log10_flux_ratio,
    log10_value,
)

__all__ = [
    "classify_whan",
    "whan_classification",
    "WHANClassification",
]

_LOG_NII_SF = -0.4  # Cid Fernandes et al. 2011
_EW_WEAK = 3.0  # Angstrom (rest frame)
_EW_STRONG = 6.0  # Angstrom (rest frame)

_CF11 = "Cid Fernandes+2011, MNRAS 413, 1687"


def classify_whan(log_nii_ha, log_ew_ha_rest) -> np.ndarray:
    """WHAN classification (log10 [NII]/Ha vs log10 rest-frame EW(Ha))."""
    x = np.asarray(log_nii_ha, dtype=float)
    ew = np.asarray(log_ew_ha_rest, dtype=float)
    out = np.full(np.broadcast(x, ew).shape, UNCLASSIFIED, dtype=int)
    valid = np.isfinite(x) & np.isfinite(ew)
    sf = valid & (x < _LOG_NII_SF)
    retired = valid & (x >= _LOG_NII_SF) & (ew < np.log10(_EW_WEAK))
    weak = valid & (x >= _LOG_NII_SF) & (ew >= np.log10(_EW_WEAK)) & (ew < np.log10(_EW_STRONG))
    strong = valid & (x >= _LOG_NII_SF) & (ew >= np.log10(_EW_STRONG))
    out[sf] = STAR_FORMING
    out[retired] = RETIRED
    out[weak] = AGN_WEAK
    out[strong] = AGN_STRONG
    return out


@dataclass
class WHANClassification:
    """Result of a WHAN classification."""

    class_map: Map2D
    log_nii_ha: Map2D
    log_ew_ha_rest: Map2D
    log_nii_ha_sigma: Map2D | None = None
    log_ew_ha_sigma: Map2D | None = None
    redshift: float = 0.0
    references: tuple[str, ...] = field(default_factory=lambda: (_CF11,))

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


def whan_classification(
    nii_flux: Map2D,
    ha_flux: Map2D,
    ew_ha_obs: Map2D,
    z: float = 0.0,
) -> WHANClassification:
    """Full WHAN pipeline.

    ``ew_ha_obs`` is the *observed-frame* H-alpha equivalent width map (e.g.
    ``maps.emission_line_ew("Ha-6564")``); it is converted to rest frame via
    ``/(1+z)`` before classification.
    """
    log_nii_ha, sig_nii_ha = log10_flux_ratio(nii_flux, ha_flux)
    ew_rest = Map2D(
        value=ew_ha_obs.value / (1.0 + z),
        uncertainty=ew_ha_obs.uncertainty,
        mask=ew_ha_obs.mask,
        wcs=ew_ha_obs.wcs,
        meta=dict(ew_ha_obs.meta, quantity="EW(Ha) rest"),
    )
    log_ew, sig_ew = log10_value(ew_rest)
    bad = log_nii_ha.bad | log_ew.bad
    classes = classify_whan(
        np.where(bad, np.nan, log_nii_ha.value.value),
        np.where(bad, np.nan, log_ew.value.value),
    )
    class_map = Map2D(
        value=Quantity(classes),
        mask=bad,
        wcs=ha_flux.wcs,
        meta=dict(ha_flux.meta, quantity="WHAN class"),
        provenance=ha_flux.provenance,
    )
    return WHANClassification(
        class_map=class_map,
        log_nii_ha=log_nii_ha,
        log_ew_ha_rest=log_ew,
        log_nii_ha_sigma=sig_nii_ha,
        log_ew_ha_sigma=sig_ew,
        redshift=z,
    )
