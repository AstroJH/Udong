"""File-location templates for MaNGA SAS products.

Mirrors the official Science Archive Server layout (validated against DR17):

* cubes / RSS:   ``manga/spectro/redux/{drpver}/{plate}/stack/manga-{plate}-{ifu}-LOGCUBE.fits.gz``
* DAP products:  ``manga/spectro/analysis/{drpver}/{dapver}/{daptype}/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-{daptype}.fits.gz``
* drpall:        ``manga/spectro/redux/{drpver}/drpall-{drpver}.fits``
* dapall:        ``manga/spectro/analysis/{drpver}/{dapver}/dapall-{drpver}-{dapver}.fits``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = ["MangaPath", "parse_plateifu", "DAPTYPES"]

DAPTYPES = (
    "SPX-MILESHC-MASTARSSP",
    "VOR10-MILESHC-MASTARSSP",
    "HYB10-MILESHC-MASTARSSP",
    "HYB10-MILESHC-MASTARHC2",
)

_PLATEIFU_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_plateifu(plateifu: str) -> tuple[str, str]:
    """Split a ``plate-ifu`` identifier into (plate, ifu)."""
    m = _PLATEIFU_RE.match(plateifu)
    if m is None:
        raise ValueError(f"invalid plateifu {plateifu!r}; expected e.g. '8485-1901'")
    return m.group(1), m.group(2)


@dataclass(frozen=True)
class MangaPath:
    """Builds local/remote paths for MaNGA products under a data root."""

    root: Path | str
    release: str = "dr17"
    drpver: str = "v3_1_1"
    dapver: str = "3.1.0"

    # ------------------------------------------------------------------ #
    def _rel(self, *parts: str) -> Path:
        return Path(self.release, *parts)

    def _local(self, *parts: str) -> Path:
        return Path(self.root, *parts)

    def _url(self, *parts: str) -> str:
        return f"https://data.sdss.org/sas/{Path(*parts)}"

    # ------------------------------------------------------------------ #
    def cube_rel(self, plateifu: str, wave: Literal["LOG", "LIN"] = "LOG") -> Path:
        plate, ifu = parse_plateifu(plateifu)
        return self._rel("manga", "spectro", "redux", self.drpver, plate, "stack",
                         f"manga-{plate}-{ifu}-{wave}CUBE.fits.gz")

    def rss_rel(self, plateifu: str, wave: Literal["LOG", "LIN"] = "LOG") -> Path:
        plate, ifu = parse_plateifu(plateifu)
        return self._rel("manga", "spectro", "redux", self.drpver, plate, "stack",
                         f"manga-{plate}-{ifu}-{wave}RSS.fits.gz")

    def maps_rel(self, plateifu: str, daptype: str) -> Path:
        plate, ifu = parse_plateifu(plateifu)
        return self._rel("manga", "spectro", "analysis", self.drpver, self.dapver,
                         daptype, plate, ifu,
                         f"manga-{plate}-{ifu}-MAPS-{daptype}.fits.gz")

    def model_cube_rel(self, plateifu: str, daptype: str) -> Path:
        plate, ifu = parse_plateifu(plateifu)
        return self._rel("manga", "spectro", "analysis", self.drpver, self.dapver,
                         daptype, plate, ifu,
                         f"manga-{plate}-{ifu}-LOGCUBE-{daptype}.fits.gz")

    def drpall_rel(self) -> Path:
        return self._rel("manga", "spectro", "redux", self.drpver, f"drpall-{self.drpver}.fits")

    def dapall_rel(self) -> Path:
        return self._rel("manga", "spectro", "analysis", self.drpver, self.dapver,
                         f"dapall-{self.drpver}-{self.dapver}.fits")

    # ------------------------------------------------------------------ #
    def cube_local(self, plateifu: str, wave: Literal["LOG", "LIN"] = "LOG") -> Path:
        return self._local(*self.cube_rel(plateifu, wave).parts)

    def rss_local(self, plateifu: str, wave: Literal["LOG", "LIN"] = "LOG") -> Path:
        return self._local(*self.rss_rel(plateifu, wave).parts)

    def maps_local(self, plateifu: str, daptype: str) -> Path:
        return self._local(*self.maps_rel(plateifu, daptype).parts)

    def model_cube_local(self, plateifu: str, daptype: str) -> Path:
        return self._local(*self.model_cube_rel(plateifu, daptype).parts)

    def drpall_local(self) -> Path:
        return self._local(*self.drpall_rel().parts)

    def dapall_local(self) -> Path:
        return self._local(*self.dapall_rel().parts)

    # ------------------------------------------------------------------ #
    def cube_url(self, plateifu: str, wave: Literal["LOG", "LIN"] = "LOG") -> str:
        return self._url(*self.cube_rel(plateifu, wave).parts)

    def rss_url(self, plateifu: str, wave: Literal["LOG", "LIN"] = "LOG") -> str:
        return self._url(*self.rss_rel(plateifu, wave).parts)

    def maps_url(self, plateifu: str, daptype: str) -> str:
        return self._url(*self.maps_rel(plateifu, daptype).parts)

    def model_cube_url(self, plateifu: str, daptype: str) -> str:
        return self._url(*self.model_cube_rel(plateifu, daptype).parts)

    def drpall_url(self) -> str:
        return self._url(*self.drpall_rel().parts)

    def dapall_url(self) -> str:
        return self._url(*self.dapall_rel().parts)
