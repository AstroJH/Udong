"""Top-level MUSE data access: :class:`MuseDataset`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.table import Table

from udong.core.map import Map2D

from udong.data.base import IFUDataset
from udong.data.muse.archive import muse_archive
from udong.data.muse.locator import MusePath
from udong.data.net import progress_enabled
from udong.data.muse.products import MuseCube
from udong.data.muse.reader import (
    dp_id_from_filename,
    read_cube,
    read_exposure_map,
    read_whitelight,
)

__all__ = ["MuseDataset", "default_root"]


def default_root() -> Path:
    """Default MUSE data root: ``$UDONG_MUSE_ROOT`` or ``~/Repository/muse``."""
    env = os.environ.get("UDONG_MUSE_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Repository" / "muse"


class MuseDataset:
    """Locate, download (optionally) and read ESO Phase 3 MUSE products.

    A MUSE product is identified by its ESO archive dataset id, ``dp_id``,
    e.g. ``"ADP.2024-04-30T18:20:44.624"`` (Ton S 180 WFM deep cube).  A
    ``dp_id`` is the ESO equivalent of a MaNGA ``plateifu``: a stable,
    archive-wide identifier (here ``ADP.<UTC-timestamp>``) that also encodes
    the direct download URL ``https://dataportal.eso.org/dataPortal/file/<dp_id>``.

    Implements the cube-centric :class:`~udong.data.base.IFUDataset`
    protocol.  Resolving human-readable target names to ``dp_id`` is a
    catalog concern; :meth:`find_target` provides a *local* helper that scans
    already-downloaded cubes (remote ObsCore discovery lives in the catalog
    layer).

    Examples
    --------
    >>> muse = MuseDataset()                       # doctest: +SKIP
    >>> cube = muse.load_cube("ADP.2024-04-30T18:20:44.624")  # doctest: +SKIP
    >>> spec = cube.spectrum(x=164, y=159)         # doctest: +SKIP

    Products live at the canonical layout ``<root>/<dp_id>.fits``; pass
    ``path=`` only for files kept elsewhere:

    >>> cube = muse.load_cube(
    ...     "ADP.2024-04-30T18:20:44.624",
    ...     path="~/somewhere/else/cube.fits",
    ...     download=False)                        # doctest: +SKIP

    Find already-downloaded cubes by (substring of) target name:

    >>> muse.find_target("Ton S 180")              # doctest: +SKIP
    [('ADP.2024-04-30T18:20:44.624', PosixPath('.../ADP....fits'))]
    """

    def __init__(
        self,
        root: Path | str | None = None,
        allow_download: bool = True,
    ) -> None:
        self.root = Path(root or default_root()).expanduser()
        self.allow_download = allow_download
        self.paths = MusePath(self.root)
        self._archives: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # IFUDataset
    # ------------------------------------------------------------------ #
    def _archive(self, mode: str = "cube"):
        """Cached easycat MUSEArchive for a product mode."""
        if mode not in self._archives:
            self._archives[mode] = muse_archive(mode=mode)
        return self._archives[mode]

    def _fetch(self, dp_id: str, dest: Path, mode: str,
               progress=None) -> Path:
        """Download one product via easycat ``MUSEArchive.fetch_one``.

        ``dest`` keeps Udong's own local naming; errors carry the dataportal
        URL from ``MusePath.url``.
        """
        import sys

        sys.stderr.write(f"[udong] downloading {dp_id}\n")
        sys.stderr.write(f"         -> {dest}\n")
        item = self._archive(mode).fetch_one(
            dp_id,
            dest=dest,
            download=True,
            progress=progress_enabled() if progress is None else progress,
            validate="fits",
        )
        if not getattr(item, "success", False):
            raise OSError(
                f"download failed for {dp_id} ({mode}): "
                f"{getattr(item, 'error', '') or 'unknown error'}"
            )
        if getattr(item, "data", None) is None:
            raise FileNotFoundError(
                f"no MUSE product {dp_id} ({mode}) in the ESO archive\n"
                f"  (url: {self.paths.url(dp_id)})"
            )
        return dest

    def load_cube(
        self,
        identifier: str,
        path: Path | str | None = None,
        download: bool = True,
        progress=None,
    ) -> MuseCube:
        """Load a MUSE datacube identified by its ESO ``dp_id``.

        When the local file is missing it is fetched with easycat's
        ``MUSEArchive`` (atomic ``.part`` download, FITS validation); pass
        ``download=False`` to require a local file.
        """
        local = Path(path).expanduser() if path is not None else self.paths.local(identifier)

        if not local.exists():
            if not download or not self.allow_download:
                raise FileNotFoundError(
                    f"expected file not found: {local}\n"
                    f"  (dp_id: {identifier}; url: {self.paths.url(identifier)}) - "
                    "set allow_download=True / download=True or place the file there"
                )
            local.parent.mkdir(parents=True, exist_ok=True)
            self._fetch(identifier, local, "cube", progress=progress)
        return read_cube(local, dp_id=identifier)

    # ------------------------------------------------------------------ #
    # ancillary 2-D images (whitelight / exposure map)
    # ------------------------------------------------------------------ #
    def _load_image(self, kind: str, reader, dp_id: str, path,
                    download: bool = False, progress=None) -> Map2D:
        local = Path(path).expanduser() if path is not None else (
            self.root / f"{dp_id}.{kind}.fits"
        )
        if not local.exists():
            if not download or not self.allow_download:
                raise FileNotFoundError(
                    f"expected file not found: {local}\n"
                    f"  (dp_id: {dp_id}; url: {self.paths.url(dp_id)}) - pass "
                    f"path= or set download=True"
                )
            local.parent.mkdir(parents=True, exist_ok=True)
            self._fetch(dp_id, local, kind, progress=progress)
        return reader(local)

    def load_whitelight(self, dp_id: str, path: Path | str | None = None,
                        download: bool = False, progress=None) -> Map2D:
        """Load a MUSE white-light image as a Map2D.

        ``dp_id`` is the white-light product's **own** ESO id (the cube has a
        different one, e.g. ``ADP....625`` vs ``ADP....624``).  The default
        local path is ``root/<dp_id>.whitelight.fits``.
        """
        return self._load_image("whitelight", read_whitelight, dp_id, path,
                                download=download, progress=progress)

    def load_exposure_map(self, dp_id: str, path: Path | str | None = None,
                          download: bool = False, progress=None) -> Map2D:
        """Load a MUSE exposure map as a Map2D (its own ``dp_id``; unit s)."""
        return self._load_image("expmap", read_exposure_map, dp_id, path,
                                download=download, progress=progress)

    # ------------------------------------------------------------------ #
    # remote product discovery (ESO ObsCore, catalog layer)
    # ------------------------------------------------------------------ #
    def discover(
        self,
        *,
        target: str | None = None,
        ra=None,
        dec=None,
        radius=None,
        dataproduct_type: str | None = "cube",
        top: int | None = None,
        timeout: float = 120.0,
    ) -> Table:
        """Query ESO ObsCore for MUSE reduced products (requires network).

        Thin wrapper around
        :func:`udong.catalogs.obscore.search_muse_products`; rows can be fed
        straight back into :meth:`load_cube` via their ``dp_id``.  ``target``
        is a free-text, case-insensitive substring match on the archive's
        (PI-provided) target name -- aliases differ, so prefer ``ra/dec/
        radius`` when the coordinates are known.
        """
        from udong.catalogs.obscore import search_muse_products  # lazy: catalog layer

        return search_muse_products(
            target=target,
            ra=ra,
            dec=dec,
            radius=radius,
            dataproduct_type=dataproduct_type,
            top=top,
            timeout=timeout,
        )

    # ------------------------------------------------------------------ #
    # local target-name discovery
    # ------------------------------------------------------------------ #
    def local_cubes(self) -> list[dict[str, Any]]:
        """Scan ``root/*.fits`` for MUSE cubes (header-only, no data load).

        Returns a list of ``{"dp_id", "path", "target"}`` for every file that
        looks like a MUSE datacube (has DATA + STAT extensions).
        """
        out: list[dict[str, Any]] = []
        for p in sorted(self.root.glob("*.fits")):
            try:
                with fits.open(p, memmap=True) as hdul:
                    prim = hdul[0].header
                    if "DATA" not in hdul or "STAT" not in hdul:
                        continue
                    target = prim.get("ESO OBS TARG NAME") or prim.get("OBJECT") or ""
                    out.append(
                        {
                            "dp_id": dp_id_from_filename(p),
                            "path": p,
                            "target": str(target),
                        }
                    )
            except Exception:
                continue
        return out

    def find_target(
        self, target: str, case_sensitive: bool = False
    ) -> list[tuple[str, Path]]:
        """Return ``(dp_id, path)`` of local cubes whose target matches.

        Matching is a case-insensitive substring test on the header target
        name (e.g. ``"Ton S 180"``, alias ``"HE 0054-2239"``).  Only scans
        ``root`` non-recursively; remote discovery needs the catalog layer.
        """
        name = target if case_sensitive else target.lower()
        hits: list[tuple[str, Path]] = []
        for item in self.local_cubes():
            t = item["target"] if case_sensitive else item["target"].lower()
            if name in t:
                hits.append((item["dp_id"] or "", item["path"]))
        return hits
