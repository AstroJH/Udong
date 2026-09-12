"""MUSE (ESO) data access.

Public API:
    MuseDataset    top-level access (locate / download / read by ESO dp_id)
    MuseCube       product returned by MuseDataset (a udong.core.Cube)
    read_cube      low-level FITS reader
    MusePath       local/remote path templates
    muse_archive    easycat MUSEArchive factory (ESO dataportal downloads)
    muse_unit      MUSE BUNIT string parser

Everything MUSE-specific (FITS layout, STAT-as-variance, NaN-as-mask, AWAV wavelengths) 
is confined to this package.
"""

from __future__ import annotations

from udong.data.muse.archive import muse_archive
from udong.data.muse.dataset import MuseDataset, default_root
from udong.data.muse.locator import MusePath
from udong.data.muse.products import MuseCube
from udong.data.muse.reader import (
    dp_id_from_filename,
    read_cube,
    read_exposure_map,
    read_whitelight,
)
from udong.data.muse.units import muse_unit

__all__ = [
    "MuseDataset",
    "default_root",
    "MuseCube",
    "read_cube",
    "read_whitelight",
    "read_exposure_map",
    "dp_id_from_filename",
    "MusePath",
    "muse_archive",
    "muse_unit",
]
