"""Ionization / AGN diagnostics.

* ``bpt``: [NII]-BPT (star-forming / composite / AGN, optional Seyfert/LINER).
* ``sii_bpt``: [SII]-BPT (star-forming / Seyfert / LINER).
* ``oi_bpt``: [OI]-BPT (star-forming / Seyfert / LINER).
* ``whan``: WHAN diagram using EW(Ha) (SF / strong AGN / weak AGN / retired).
"""

from __future__ import annotations

from udong.science.diagnostics.bpt import (
    AGN,
    AGN_STRONG,
    AGN_WEAK,
    CLASS_NAMES,
    COMPOSITE,
    LINER,
    RETIRED,
    SEYFERT,
    STAR_FORMING,
    UNCLASSIFIED,
    BPTClassification,
    cid_fernandes_2010_sl,
    classify_nii,
    kauffmann_2003_nii,
    kewley_2001_nii,
    log10_flux_ratio,
    log10_value,
    nii_bpt_classification,
)
from udong.science.diagnostics.oi_bpt import (
    OIBPTClassification,
    classify_oi,
    kewley_2001_oi,
    oi_bpt_classification,
)
from udong.science.diagnostics.oi_bpt import (
    kewley_2006_sl as kewley_2006_oi,
)
from udong.science.diagnostics.sii_bpt import (
    SIIBPTClassification,
    classify_sii,
    kewley_2001_sii,
    kewley_2006_sl,
    sii_bpt_classification,
    total_sii_flux,
)
from udong.science.diagnostics.whan import (
    WHANClassification,
    classify_whan,
    whan_classification,
)

__all__ = [
    "AGN",
    "AGN_STRONG",
    "AGN_WEAK",
    "BPTClassification",
    "CLASS_NAMES",
    "COMPOSITE",
    "LINER",
    "OIBPTClassification",
    "RETIRED",
    "SEYFERT",
    "SIIBPTClassification",
    "STAR_FORMING",
    "UNCLASSIFIED",
    "WHANClassification",
    "cid_fernandes_2010_sl",
    "classify_nii",
    "classify_oi",
    "classify_sii",
    "classify_whan",
    "kauffmann_2003_nii",
    "kewley_2001_nii",
    "kewley_2001_oi",
    "kewley_2001_sii",
    "kewley_2006_oi",
    "kewley_2006_sl",
    "log10_flux_ratio",
    "log10_value",
    "nii_bpt_classification",
    "oi_bpt_classification",
    "sii_bpt_classification",
    "total_sii_flux",
    "whan_classification",
]
