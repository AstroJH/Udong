"""Survey-agnostic catalog layer.

* ``matching`` -- generic sky cross-matching primitives;
* ``obscore`` -- ESO Archive ObsCore (VO) product discovery (e.g. MUSE);
* ``surveys`` -- external survey registry + product/cone access (ADR-008).
* ``wise`` / ``ztf`` / ``desi`` -- easycat-Archive adapters (Phase B).
"""

from __future__ import annotations

from udong.catalogs.matching import (
    CrossMatchResult,
    crossmatch,
    crossmatch_tables,
    deduplicate,
    match_to_catalog,
)

from udong.catalogs.obscore import (
    EsoQueryError,
    pixel_scale_mode,
    query_obscore,
    search_muse_products,
)
from udong.catalogs.desi import fetch_cutout
from udong.catalogs.surveys import (
    SURVEYS,
    SurveyNotImplemented,
    SurveyProduct,
    SurveySpec,
    available_surveys,
    cone_search,
    fetch_cone,
    fetch_one,
    read_product_table,
    register,
    resolve,
)

__all__ = [
    "CrossMatchResult",
    "crossmatch",
    "crossmatch_tables",
    "deduplicate",
    "match_to_catalog",
    "EsoQueryError",
    "query_obscore",
    "search_muse_products",
    "pixel_scale_mode",
    "SURVEYS",
    "SurveySpec",
    "SurveyNotImplemented",
    "SurveyProduct",
    "available_surveys",
    "cone_search",
    "fetch_one",
    "fetch_cone",
    "fetch_cutout",
    "read_product_table",
    "register",
    "resolve",
]
