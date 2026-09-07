"""MaNGA-specific data access.

Public API:
    MangaDataset                              top-level access (locate / download / read)
    MangaCube, MangaMaps                      products returned by MangaDataset
    make_manga_templates / default_sps_file   MaNGA E-MILES stellar templates
    fetch_maskbits / ensure_maskbits          official sdssMaskbits.par (downloaded, not vendored)
"""

from __future__ import annotations

from udong.data.manga.dataset import MangaDataset
from udong.data.manga.products import MangaCube, MangaMaps

# MaNGA-side stellar-template preparation (E-MILES).  The generic per-spaxel
# pPXF fit lives in udong.science.spectroscopy.continuum.
from udong.data.manga.continuum import (
    default_sps_file,
    make_manga_templates,
)

# Mask-bit registry: definitions are maintained upstream by SDSS and fetched
# into a local cache.
from udong.data.manga.masks import (
    ensure_maskbits,
    fetch_maskbits,
    get_mask_defs,
    load_mask_registries,
)

__all__ = [
    "MangaDataset",
    "MangaCube",
    "MangaMaps",

    "default_sps_file",
    "make_manga_templates",

    "ensure_maskbits",
    "fetch_maskbits",
    "get_mask_defs",
    "load_mask_registries",
]
