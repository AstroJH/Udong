"""FITS readers for MaNGA products.

These are the *only* place in the code base that knows about MaNGA FITS
extension names / numbers.  Everything above this layer works with the core
types (``MangaCube``, ``MangaMaps``, ``Map2D``, ...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.wcs import WCS

from udong.core.map import Map2D
from udong.core.mask import MaskDefs
from udong.core.provenance import Provenance

from udong.data.manga.products import MangaCube, MangaMaps
from udong.data.manga.masks import mask_defs_or_none
from udong.data.manga.downloader import gzip_ok
from udong.data.manga.units import manga_unit

__all__ = ["read_cube", "read_maps", "read_drpall", "read_dapall"]

# -- WCS helpers ------------------------------------------------------------- #
_WCS_KEY_PREFIXES = (
    "WCSAXES", "CTYPE", "CRVAL", "CRPIX", "CD", "PC", "CDELT", "CUNIT",
    "RADESYS", "EQUINOX", "NAXIS", "LONPOLE", "LATPOLE", "MJDREF",
    "DATE-OBS", "MJD-OBS", "CROTA",
)


def _wcs_only_header(header: fits.Header, naxis: int) -> fits.Header:
    """A copy of ``header`` containing only WCS-relevant keywords.

    Passing a full survey header to ``WCS()`` produces spurious warnings
    (e.g. ``PLATEID`` parsed as a string); filtering avoids that.
    """
    out = fits.Header()
    for key, value in header.items():
        if key.startswith(_WCS_KEY_PREFIXES):
            out[key] = value
    out["NAXIS"] = naxis
    for i in range(1, naxis + 1):
        nk = f"NAXIS{i}"
        if nk in header:
            out[nk] = header[nk]
    return out


def _wcs_from_header(header: fits.Header, naxis: int) -> WCS:
    """Build a WCS from a FITS header, forcing ``naxis`` axes."""
    try:
        w = WCS(_wcs_only_header(header, naxis), naxis=naxis)
        if w.pixel_n_dim == naxis:
            return w
    except Exception:
        pass
    # manual fallback (e.g. NAXIS=0 primary headers)
    w = WCS(naxis=naxis)
    for i in range(1, naxis + 1):
        w.wcs.crpix[i - 1] = header.get(f"CRPIX{i}", 1.0)
        w.wcs.crval[i - 1] = header.get(f"CRVAL{i}", 0.0)
        w.wcs.ctype[i - 1] = header.get(f"CTYPE{i}", "")
    cd = np.eye(naxis)
    for i in range(1, naxis + 1):
        for j in range(1, naxis + 1):
            if f"CD{i}_{j}" in header:
                cd[i - 1, j - 1] = header[f"CD{i}_{j}"]
            elif f"PC{i}_{j}" in header and f"CDELT{j}" in header:
                cd[i - 1, j - 1] = header[f"PC{i}_{j}"] * header[f"CDELT{j}"]
            elif i == j and f"CDELT{i}" in header:
                cd[i - 1, i - 1] = header[f"CDELT{i}"]
    w.wcs.pc = cd
    w.wcs.set()
    return w


def _spatial_wcs(header: fits.Header) -> WCS | None:
    """2-D spatial WCS from an extension/primary header."""
    try:
        w = WCS(header)
        if w.pixel_n_dim >= 2:
            return w.sub(["longitude", "latitude"])
    except Exception:
        pass
    try:
        return _wcs_from_header(header, 2)
    except Exception:
        return None


def _channels(header: fits.Header) -> tuple[list[str], list[str | None]]:
    """Channel names and units from ``C##``/``U##`` header cards."""
    names: list[str] = []
    units: list[str | None] = []
    n = header.get("NAXIS3", 0)
    for i in range(1, n + 1):
        cname = header.get(f"C{i:02d}") or header.get(f"C{i}")
        uname = header.get(f"U{i:02d}") or header.get(f"U{i}")
        if cname is None and i > 0:
            cname = f"channel{i}"
        names.append(str(cname))
        units.append(None if uname is None else str(uname))
    return names, units


# -- provenance -------------------------------------------------------------- #
_META_KEYS = (
    "PLATEIFU", "MANGAID", "PLATEID", "IFUDSGN", "OBJRA", "OBJDEC", "IFURA",
    "IFUDEC", "Z", "EBVGAL", "DRP3QUAL", "DAPQUAL", "MNGTARG1", "MNGTARG2",
    "MNGTARG3", "NEXP", "EXPTIME", "SEEMED", "SEEING", "VERSDRP2", "VERSDRP3",
    "VERSDAP", "DAPTYPE", "MASKNAME", "BUNIT", "REFF", "ECOOPA", "ECOOELL",
    "BINKEY", "SCKEY", "ELMKEY", "ELFKEY", "SIKEY", "GEXTLAW", "RVGAL",
)


def _meta(header: fits.Header) -> dict[str, Any]:
    return {k: header[k] for k in _META_KEYS if k in header}


def _versions(header: fits.Header) -> dict[str, str]:
    return {k: str(header[k]) for k in header if k.startswith("VERS")}


def _checksums(header: fits.Header) -> dict[str, str]:
    out = {}
    if "DATASUM" in header:
        out["datasum"] = str(header["DATASUM"])
    if "CHECKSUM" in header:
        out["checksum"] = str(header["CHECKSUM"])
    return out



def _verify_local_file(path: Path) -> Path:
    """Raise a clear error for truncated/corrupt local files (esp. .gz)."""
    path = Path(path)
    if path.suffix == ".gz" and not gzip_ok(path):
        raise OSError(
            f"{path} is truncated or corrupt (gzip integrity check failed). "
            "Delete the file and re-download it, e.g. "
            "MangaDataset.load_cube('...', download=True)."
        )
    return path


def _open_fits(path: Path):
    """fits.open with a clear message on truncation."""
    try:
        return fits.open(path)
    except (OSError, EOFError, KeyError) as exc:
        raise OSError(
            f"failed to read FITS file {path}: {exc}. "
            "The file may be truncated or incomplete; re-download it."
        ) from exc

# -- readers ----------------------------------------------------------------- #
def read_cube(
    path: Path | str,
    mask_defs: MaskDefs | None = None,
    release: str | None = None,
) -> MangaCube:
    """Read a MaNGA DRP LOGCUBE (or LINCUBE) file."""
    path = _verify_local_file(Path(path))
    with _open_fits(path) as hdul:
        prim = hdul[0].header
        flux_hdu = hdul["FLUX"]
        unit = manga_unit(prim.get("BUNIT"))
        flux = Quantity(flux_hdu.data, unit, copy=False)
        ivar = np.asarray(hdul["IVAR"].data, dtype=float)
        mask = np.asarray(hdul["MASK"].data)
        wave = Quantity(hdul["WAVE"].data, u.AA, copy=False)
        lsf_pre = hdul["LSFPRE"].data
        lsf_post = hdul["LSFPOST"].data
        wcs = _wcs_from_header(prim, 3)

    meta = _meta(prim)
    # Registry is resolved lazily (downloaded once by MangaDataset when needed);
    # without a local par file, readers keep raw masks and treat any non-zero
    # bit as bad (mask_defs_or_none warns once).
    if mask_defs is None:
        mask_defs = mask_defs_or_none("MANGA_DRP3PIXMASK")
    prov = Provenance(
        product="manga-cube",
        source=str(path),
        survey="MaNGA",
        release=release,
        drpver=meta.get("VERSDRP3"),
        checksums=_checksums(prim),
        versions=_versions(prim),
        meta={"header": dict(prim)},
    )
    return MangaCube(
        flux=flux,
        ivar=ivar,
        mask=mask,
        wavelength=wave,
        wcs=wcs,
        meta=meta,
        provenance=prov,
        mask_defs=mask_defs,
        lsf_pre=lsf_pre,
        lsf_post=lsf_post,
    )


def read_maps(
    path: Path | str,
    mask_defs: MaskDefs | None = None,
    release: str | None = None,
) -> MangaMaps:
    """Read a DAP MAPS file into :class:`MangaMaps`."""
    path = _verify_local_file(Path(path))
    if mask_defs is None:
        mask_defs = mask_defs_or_none("MANGA_DAPPIXMASK")
    maps: dict[str, Map2D] = {}
    channel_maps: dict[str, dict[str, Any]] = {}
    with _open_fits(path) as hdul:
        prim = hdul[0].header
        for hdu in hdul:
            name = hdu.name
            if not name or name == "PRIMARY" or hdu.data is None:
                continue
            if name.endswith("_IVAR") or name.endswith("_MASK"):
                continue
            base = name
            # data-only extensions (e.g. SPX_ELLCOO) and partial triples are
            # both kept; _IVAR/_MASK are attached when present.
            ivar = None
            mask = None
            if base + "_IVAR" in hdul:
                ivar = np.asarray(hdul[base + "_IVAR"].data, dtype=float)
            if base + "_MASK" in hdul:
                mask = np.asarray(hdul[base + "_MASK"].data)
            bunit = manga_unit(hdu.header.get("BUNIT"))
            wcs = _spatial_wcs(hdu.header)
            if hdu.data.ndim == 2:
                maps[base] = Map2D(
                    value=Quantity(hdu.data, bunit, copy=False),
                    uncertainty=ivar,
                    mask=mask,
                    wcs=wcs,
                    meta=dict(_meta(hdu.header), extname=base),
                    mask_defs=mask_defs,
                    provenance=None,
                )
            else:
                ch_names, ch_units = _channels(hdu.header)
                channel_maps[base] = {
                    "value": hdu.data,
                    "ivar": ivar,
                    "mask": mask,
                    "wcs": wcs,
                    "bunit": bunit,
                    "channels": ch_names,
                    "channel_units": ch_units,
                    "meta": dict(_meta(hdu.header), extname=base),
                }

    prov = Provenance(
        product="manga-maps",
        source=str(path),
        survey="MaNGA",
        release=release,
        drpver=_versions(prim).get("VERSDRP3"),
        dapver=_versions(prim).get("VERSDAP"),
        checksums=_checksums(prim),
        versions=_versions(prim),
        meta={"header": dict(prim)},
    )
    return MangaMaps(
        maps=maps,
        channel_maps=channel_maps,
        meta=_meta(prim),
        provenance=prov,
        mask_defs=mask_defs,
    )


def read_drpall(path: Path | str) -> Table:
    """Read the MaNGA DRPall summary catalog (MaNGA cubes: HDU 1)."""
    path = _verify_local_file(Path(path))
    return Table.read(path, hdu=1)


def read_dapall(path: Path | str) -> Table:
    """Read the MaNGA DAPall summary catalog."""
    path = _verify_local_file(Path(path))
    return Table.read(path, hdu=1)
