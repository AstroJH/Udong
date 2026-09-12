"""Top-level MaNGA data access: :class:`MangaDataset`."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity

from .locator import DAPTYPES, MangaPath
from .products import MangaCube, MangaMaps
from .archive import manga_archive
from udong.data.net import progress_enabled

from .downloader import download_file, gzip_ok
from .reader import read_cube, read_dapall, read_drpall, read_maps

__all__ = ["MangaDataset", "default_root"]


def default_root() -> Path:
    """Default MaNGA data root: ``$UDONG_MANGA_ROOT`` or ``~/Repository/manga``."""
    env = os.environ.get("UDONG_MANGA_ROOT")

    if env:
        return Path(env).expanduser()
    
    return Path.home() / "Repository" / "manga"


class MangaDataset:
    """Locate, download (optionally) and read MaNGA DR17 products.

    Examples
    --------
    >>> manga = MangaDataset()
    >>> cube = manga.load_cube("8485-1901")             # doctest: +SKIP
    >>> spec = cube.spectrum(x=18, y=18)                # doctest: +SKIP
    >>> maps = manga.load_maps("8485-1901")             # doctest: +SKIP
    >>> vel = maps["STELLAR_VEL"]                       # doctest: +SKIP
    """

    def __init__(
        self,
        root: Path | str | None = None,
        release: str = "dr17",
        drpver: str = "v3_1_1",
        dapver: str = "3.1.0",
        default_daptype: str = "HYB10-MILESHC-MASTARSSP",
        allow_download: bool = True,
    ) -> None:
        if default_daptype not in DAPTYPES:
            raise ValueError(f"unknown DAPTYPE {default_daptype!r}; use one of {DAPTYPES}")
        self.root = Path(root or default_root()).expanduser()
        self.release = release
        self.drpver = drpver
        self.dapver = dapver
        self.default_daptype = default_daptype
        self.allow_download = allow_download
        self.paths = MangaPath(self.root, release, drpver, dapver)
        self._drpall: Table | None = None
        self._dapall: Table | None = None
        self._mask_cache: dict[str, object] = {}
        self._archives: dict[tuple[str, str | None], object] = {}

    # ------------------------------------------------------------------ #
    def _resolve(
        self, local: Path, url: str, download: bool, verify_gzip: bool = True
    ) -> Path:
        if local.exists():
            if not (verify_gzip and local.suffix == ".gz") or gzip_ok(local):
                return local
            
            # stale/truncated partial download: refuse to read it, then
            # re-fetch.  download_file() resumes from the existing bytes when
            # the partial is a valid prefix of the file.
            if not download or not self.allow_download:
                raise FileNotFoundError(
                    f"existing file is truncated/corrupt: {local}\n"
                    f"  (url: {url}) - delete the file or enable download to re-fetch it"
                )
        else:
            if not download or not self.allow_download:
                raise FileNotFoundError(
                    f"expected file not found: {local}\n"
                    f"  (url: {url}) - set allow_download=True or place the file there"
                )
            
        download_file(url, local)
        return local

    # ------------------------------------------------------------------ #
    def _mask_defs(self, group: str):
        """Resolve a MaNGA mask registry (download once when enabled).

        Returns ``None`` when ``allow_download=False`` and the registry file
        is not cached locally (containers then keep raw masks).
        """
        if group in self._mask_cache:
            return self._mask_cache[group]
        from udong.data.manga.masks import get_mask_defs

        try:
            md = get_mask_defs(group, download=self.allow_download)
        except FileNotFoundError:
            if self.allow_download:
                raise
            md = None  # offline: keep raw masks (bad = any non-zero bit)
        self._mask_cache[group] = md
        return md

    # ------------------------------------------------------------------ #
    # easycat archive plumbing (product URLs + single-target downloads)
    # ------------------------------------------------------------------ #
    def _archive(self, product: str, dap: str | None = None):
        """Cached easycat SDSSArchive for a MaNGA product."""
        key = (product, dap)
        if key not in self._archives:
            self._archives[key] = manga_archive(
                release=self.release, drpver=self.drpver, dapver=self.dapver,
                product=product, dap=dap,
            )
        return self._archives[key]

    def _product_url(self, product: str, dap: str | None, plateifu: str,
                     local: Path) -> str:
        """Planned URL for a product (no download, no side effects)."""
        try:
            item = self._archive(product, dap).fetch_one(
                plateifu, dest=local, download=False
            )
        except Exception:
            return ""
        return str((getattr(item, "meta", {}) or {}).get("url", ""))

    def _resolve_product(self, local: Path, plateifu: str, product: str,
                         dap: str | None, download: bool) -> Path:
        """Locate or download one MaNGA product, keeping Udong's local layout.

        The transfer is delegated to easycat's ``SurveyArchive.fetch_one``
        (atomic ``.part`` download, size + gzip verification, progress), while
        ``dest=`` keeps our SAS-mirrored path and ``ItemResult.meta`` supplies
        the URL for error messages.
        """
        local = Path(local)
        if local.exists():
            if local.suffix != ".gz" or gzip_ok(local):
                return local
            if not download or not self.allow_download:
                raise FileNotFoundError(
                    f"existing file is truncated/corrupt: {local}\n"
                    f"  (url: {self._product_url(product, dap, plateifu, local)}) - "
                    "delete the file or enable download to re-fetch it"
                )

        if not download or not self.allow_download:
            raise FileNotFoundError(
                f"expected file not found: {local}\n"
                f"  (url: {self._product_url(product, dap, plateifu, local)}) - "
                "set allow_download=True or place the file there"
            )

        local.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(f"[udong] downloading {local.name}\n")
        sys.stderr.write(f"         -> {local}\n")
        item = self._archive(product, dap).fetch_one(
            plateifu,
            dest=local,
            download=True,
            progress=progress_enabled(),
            validate="gzip" if local.suffix == ".gz" else None,
        )
        if not getattr(item, "success", False):
            raise OSError(
                f"download failed for {plateifu} ({product}): "
                f"{getattr(item, 'error', '') or 'unknown error'}"
            )
        if getattr(item, "data", None) is None:
            url = (getattr(item, "meta", {}) or {}).get("url", "")
            raise FileNotFoundError(
                f"no {product} product for {plateifu} in the archive\n  (url: {url})"
            )
        return local

    # ------------------------------------------------------------------ #
    def load_cube(
        self,
        plateifu: str,
        wave: Literal["LOG", "LIN"] = "LOG",
        download: bool = True,
    ) -> MangaCube:
        """Load a DRP datacube for ``plateifu`` (e.g. ``"8485-1901"``)."""
        local = self.paths.cube_local(plateifu, wave)
        path = self._resolve_product(local, plateifu, f"{wave}CUBE", None, download)
        return read_cube(path, mask_defs=self._mask_defs("MANGA_DRP3PIXMASK"),
                         release=self.release)

    def load_maps(
        self,
        plateifu: str,
        daptype: str | None = None,
        download: bool = True,
    ) -> MangaMaps:
        """Load DAP MAPS for ``plateifu`` (default DAPTYPE if not given)."""
        daptype = daptype or self.default_daptype
        local = self.paths.maps_local(plateifu, daptype)
        path = self._resolve_product(local, plateifu, "MAPS", daptype, download)
        return read_maps(path, mask_defs=self._mask_defs("MANGA_DAPPIXMASK"),
                         release=self.release)

    # ------------------------------------------------------------------ #
    @property
    def drpall(self) -> Table:
        """DRPall summary catalog (cached)."""
        if self._drpall is None:
            local = self.paths.drpall_local()
            path = self._resolve(local, self.paths.drpall_url(), download=True)
            self._drpall = read_drpall(path)
        return self._drpall

    @property
    def dapall(self) -> Table:
        """DAPall summary catalog (cached)."""
        if self._dapall is None:
            local = self.paths.dapall_local()
            path = self._resolve(local, self.paths.dapall_url(), download=True)
            self._dapall = read_dapall(path)
        return self._dapall

    def galaxies_near(
        self,
        coord: SkyCoord,
        radius: Quantity,
        *,
        ra_col: str = "ifura",
        dec_col: str = "ifudec",
        sort_by_separation: bool = True,
    ) -> Table:
        """Return drpall rows whose IFU centre lies within ``radius`` of ``coord``.

        Convenience wrapper on top of the cached ``drpall`` table so that a
        target coordinate (e.g. from an AGN/catalog cross-match) can be turned
        into candidate ``plateifu`` values without hand-written table queries.

        Parameters
        ----------
        coord
            Sky position to search around.
        radius
            Matching radius (any angular unit).
        ra_col, dec_col
            drpall columns holding the coordinates used for the match.
            Defaults to the IFU centre columns ``ifura`` / ``ifudec``; use
            ``objra`` / ``objdec`` to search on the science-target position.
        sort_by_separation
            Sort the returned rows by angular separation (closest first).

        Returns
        -------
        astropy.table.Table
            A copy of the matched drpall rows with an added
            ``sep_arcsec`` column; provenance of the query is stored in
            ``meta`` (``match_coord`` / ``match_radius_arcsec``).
        """
        d = self.drpall
        ra = np.asarray(d[ra_col], dtype=float)
        dec = np.asarray(d[dec_col], dtype=float)
        sky = SkyCoord(ra, dec, unit="deg")
        sep = coord.separation(sky)
        keep = sep < Quantity(radius)
        out = d[keep].copy()
        out["sep_arcsec"] = sep[keep].to_value(u.arcsec)
        if sort_by_separation:
            order = np.argsort(out["sep_arcsec"], kind="stable")
            out = out[order]
        out.meta["match_coord"] = coord.to_string("hmsdms", precision=2)
        out.meta["match_radius_arcsec"] = Quantity(radius).to_value(u.arcsec)
        return out
