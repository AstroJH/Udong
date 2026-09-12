"""Survey-agnostic FITS/WCS helpers shared by data adapters.

These helpers know nothing about MaNGA/MUSE product semantics; they only deal
with FITS headers and astropy WCS construction.  Survey readers
(``udong.data.manga.reader``, ``udong.data.muse.reader``) keep their own
extension/unit knowledge and call into these for WCS parsing.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

__all__ = ["wcs_only_header", "wcs_from_header", "spatial_wcs", "open_fits"]

_WCS_KEY_PREFIXES = (
    "WCSAXES", "CTYPE", "CRVAL", "CRPIX", "CD", "PC", "CDELT", "CUNIT",
    "RADESYS", "EQUINOX", "NAXIS", "LONPOLE", "LATPOLE", "MJDREF",
    "DATE-OBS", "MJD-OBS", "CROTA",
)


def wcs_only_header(header: fits.Header, naxis: int) -> fits.Header:
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


def wcs_from_header(header: fits.Header, naxis: int) -> WCS:
    """Build a WCS from a FITS header, forcing ``naxis`` axes.

    Falls back to manual keyword assembly when ``WCS(header)`` fails (e.g. for
    NAXIS=0 primary headers that carry spatial keywords only).
    """
    try:
        w = WCS(wcs_only_header(header, naxis), naxis=naxis)
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


def spatial_wcs(header: fits.Header) -> WCS | None:
    """2-D spatial (RA/Dec) WCS from an extension/primary header."""
    try:
        w = WCS(header)
        if w.pixel_n_dim >= 2:
            return w.sub(["longitude", "latitude"])
    except Exception:
        pass
    try:
        return wcs_from_header(header, 2)
    except Exception:
        return None


def open_fits(path):
    """``fits.open`` with a clear message on truncation."""
    try:
        return fits.open(path)
    except (OSError, EOFError, KeyError) as exc:
        raise OSError(
            f"failed to read FITS file {path}: {exc}. "
            "The file may be truncated or incomplete; re-download it."
        ) from exc
