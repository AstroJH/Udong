"""FITS reader for ESO Phase 3 MUSE datacubes.

This is the *only* place that knows the MUSE FITS layout (validated against a
real WFM cube, `docs/guides/muse_golden_file_notes.md`):

* PRIMARY: metadata header only (no data);
* DATA: float32 cube ``(nwave, ny, nx)`` in ``10**(-20) erg/s/cm2/AA``;
* STAT: float32 **variance** cube (same shape/units squared);
* no DQ extension: bad/no-data pixels are ``NaN`` in DATA and STAT.

Conversions applied here (and recorded in provenance):
  STAT (variance)  -> core IVAR = 1/variance (``<= 0`` marks invalid);
  NaN in DATA/STAT -> boolean mask + zero IVAR;
  wavelength       -> explicit 1-D vector from the DATA header WCS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.units import Quantity
from astropy.wcs import WCS

from udong.data.fitsutil import open_fits, spatial_wcs, wcs_from_header
from udong.core.map import Map2D
from udong.core.provenance import ProcessingStep, Provenance
from udong.data.muse.products import MuseCube
from udong.data.muse.units import muse_unit

__all__ = ["read_cube", "read_whitelight", "read_exposure_map", "dp_id_from_filename"]

_DPID_RE = re.compile(r"(ADP\.\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})")

# Curated primary-header keys promoted to cube.meta (the full header is kept
# in provenance["meta"]["header"]).
_META_KEYS = (
    "TARGNAME", "INSMODE", "SKY_RES", "NCOMBINE", "EXPTIME", "PIPEVERS",
    "PROGID", "PROCATG", "MJD-OBS", "DATE-OBS", "WAVELMIN", "WAVELMAX",
    "SPEC_RES", "RADESYS", "SPECSYS", "dp_id",
)


def dp_id_from_filename(path: Path | str) -> str | None:
    """Extract an ESO ``dp_id`` (``ADP....``) from a file name, if present."""
    m = _DPID_RE.search(Path(path).name)
    return m.group(1) if m else None


def _primary_meta(header: fits.Header, dp_id: str | None) -> dict[str, Any]:
    """Curated metadata from the primary header (+ archive dp_id)."""
    def first(*keys: str):
        for k in keys:
            if k in header and header[k] is not None:
                return header[k]
        return None

    meta = {
        "TARGNAME": first("ESO OBS TARG NAME", "OBJECT", "ESO TEL TARG NAME"),
        "INSMODE": first("ESO INS MODE", "INSMODE"),
        "SKY_RES": header.get("SKY_RES"),
        "NCOMBINE": header.get("NCOMBINE"),
        "EXPTIME": header.get("EXPTIME"),
        "PIPEVERS": first("ESO PRO REC3 PIPE ID", "ESO PRO REC2 PIPE ID",
                          "ESO PRO REC1 PIPE ID"),
        "PROGID": header.get("ESO OBS PROG ID", header.get("PROG_ID")),
        "PROCATG": header.get("ESO PRO CATG"),
        "MJD-OBS": header.get("MJD-OBS"),
        "DATE-OBS": header.get("DATE-OBS"),
        "WAVELMIN": header.get("WAVELMIN"),
        "WAVELMAX": header.get("WAVELMAX"),
        "SPEC_RES": header.get("SPEC_RES"),
        "RADESYS": header.get("RADESYS"),
        "SPECSYS": header.get("SPECSYS"),
        "dp_id": dp_id,
    }
    return {k: v for k, v in meta.items() if v is not None}


def _wavelength_frame(header: fits.Header) -> str:
    """Air/vacuum flag from the spectral CTYPE (``AWAV`` = air)."""
    ctype = str(header.get("CTYPE3", "")).upper()
    if ctype.startswith("AWAV") or "AIR" in ctype:
        return "air"
    return "vacuum"


def _wavelength(header: fits.Header, nwave: int) -> Quantity:
    """Explicit wavelength vector from ``CRVAL3/CRPIX3/CD3_3`` (linear)."""
    crval = header.get("CRVAL3", 0.0)
    crpix = header.get("CRPIX3", 1.0)
    cd = header.get("CD3_3", header.get("CDELT3", None))
    if cd is None:
        raise ValueError("MUSE cube header has no CD3_3/CDELT3 spectral step")
    lam = crval + (np.arange(nwave) + 1.0 - crpix) * cd
    return Quantity(lam, u.AA, copy=False)


def _versions(header: fits.Header) -> dict[str, str]:
    out = {}
    for k in header:
        if k.startswith("ESO PRO REC") and k.endswith("PIPE ID"):
            out[k] = str(header[k])
    return out


def read_cube(path: Path | str, dp_id: str | None = None) -> MuseCube:
    """Read an ESO Phase 3 MUSE datacube file into a :class:`MuseCube`.

    Parameters
    ----------
    path
        Path to a MUSE ``*_OBJ.fits`` science cube (DATA + STAT extensions).
    dp_id
        Optional ESO archive dataset id; when ``None`` it is guessed from the
        file name (``ADP....``) and recorded in ``meta["dp_id"]``.
    """
    path = Path(path)
    dp_id = dp_id or dp_id_from_filename(path)
    with open_fits(path) as hdul:
        prim = hdul[0].header
        data_hdu = hdul["DATA"]
        stat_hdu = hdul["STAT"]
        if data_hdu.data is None or stat_hdu.data is None:
            raise OSError(f"{path}: MUSE cube missing DATA/STAT data arrays")

        # keep float32 end-to-end (real files are big-endian '>f4');
        # native conversion happens lazily in numpy ops.
        raw = np.asarray(data_hdu.data)
        var = np.asarray(stat_hdu.data, dtype=np.float32)
        if raw.ndim != 3 or var.shape != raw.shape:
            raise OSError(
                f"{path}: expected DATA/STAT shape (nwave, ny, nx), got "
                f"{raw.shape} / {var.shape}"
            )

        unit = muse_unit(data_hdu.header.get("BUNIT"))
        flux = Quantity(raw, unit, copy=False)
        header_wcs = data_hdu.header

        # variance -> IVAR; NaN in DATA/STAT and non-positive variance are bad.
        with np.errstate(all="ignore"):
            bad = ~np.isfinite(raw) | ~np.isfinite(var) | (var <= 0)
            ivar = np.zeros(raw.shape, dtype=np.float32)
            ivar[~bad] = (1.0 / var[~bad]).astype(np.float32)

        wavelength = _wavelength(header_wcs, raw.shape[0])
        wcs = wcs_from_header(header_wcs, 3)

    meta = _primary_meta(prim, dp_id)
    frame = _wavelength_frame(header_wcs)
    meta.setdefault("wavelength_frame", frame)
    prov = Provenance(
        product="muse-cube",
        source=str(path),
        survey="MUSE",
        versions=_versions(prim),
        meta={"header": dict(prim)},
        steps=[],
    )
    prov.steps.append(
        ProcessingStep(
            name="read_muse_cube",
            params={
                "stat_to_ivar": "1/variance",
                "nan_mask": "DATA|STAT NaN or var<=0 -> bad/ivar=0",
                "wavelength": "CRVAL3+(arange-CRPIX3)*CD3_3 (linear)",
                "wavelength_frame": frame,
            },
        )
    )
    return MuseCube(
        flux=flux,
        ivar=ivar,
        mask=bad,
        wavelength=wavelength,
        wcs=wcs,
        meta=meta,
        provenance=prov,
        wavelength_frame=frame,
    )


# -- ancillary 2-D images (whitelight / exposure map) ---------------- #
def _read_image2d(path: Path | str, kind: str, product: str) -> Map2D:
    """Read a MUSE ancillary 2-D image (PRIMARY metadata + DATA) into a Map2D."""
    path = Path(path)
    with open_fits(path) as hdul:
        prim = hdul[0].header
        if "DATA" not in hdul or hdul["DATA"].data is None:
            raise OSError(f"{path}: expected a DATA image extension")
        hdu = hdul["DATA"]
        raw = np.asarray(hdu.data)
        if raw.ndim != 2:
            raise OSError(f"{path}: expected a 2-D image, got shape {raw.shape}")
        unit = muse_unit(hdu.header.get("BUNIT"))
        wcs = spatial_wcs(hdu.header)

    value = Quantity(raw, unit, copy=False)
    bad = ~np.isfinite(raw)
    meta = _primary_meta(prim, dp_id_from_filename(path))
    meta["kind"] = kind
    prov = Provenance(
        product=product,
        source=str(path),
        survey="MUSE",
        meta={"header": dict(prim)},
    )
    return Map2D(value=value, mask=bad, wcs=wcs, meta=meta, provenance=prov)


def read_whitelight(path: Path | str) -> Map2D:
    """Read a MUSE white-light image (``MU_SIMD_...``) into a Map2D.

    The white-light image is a 2-D collapsed flux image on the *same* grid as
    the datacube (verified for the Ton S 180 golden file).
    """
    return _read_image2d(path, kind="whitelight", product="muse-whitelight")


def read_exposure_map(path: Path | str) -> Map2D:
    """Read a MUSE exposure map (``MU_SXPM_...``) into a Map2D.

    The exposure map gives on-source time per pixel (unit ``s``).  Note its
    grid is *not* pixel-identical to the datacube (one pixel smaller for the
    Ton S 180 file) -- use its own WCS.
    """
    return _read_image2d(path, kind="exposure-map", product="muse-exposure-map")
