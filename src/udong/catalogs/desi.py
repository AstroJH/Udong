"""DESI Legacy Surveys products via easycat's ``DESIArchive``.

Modes (easycat): ``photometry`` (Table), ``spectra`` (coadd FITS), ``image``
(cutout FITS/JPG).  Only ``photometry`` has a tabular representation; the
file-only modes require ``dest=`` or ``store_dir=`` so the download is kept.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from udong.catalogs.surveys import (
    SurveyProduct,
    coerce_product_data,
    register,
)

__all__ = ["desi_archive", "fetch_desi", "fetch_cutout"]


def desi_archive(*, mode: str = "photometry", radius_arcsec: float = 3.0,
                 ls_release: str = "dr9", cache_dir: Path | str | None = None,
                 image_layer: str = "ls-dr10", image_format: str = "fits",
                 image_size: int = 256, image_pixscale: float = 0.262,
                 download_kwargs: dict[str, Any] | None = None):
    """Configured easycat ``DESIArchive`` (monkeypatchable in tests)."""
    from easycat.download import DESIArchive

    return DESIArchive(mode=mode, radius_arcsec=radius_arcsec,
                       ls_release=ls_release, cache_dir=cache_dir,
                       image_layer=image_layer, image_format=image_format,
                       image_size=image_size, image_pixscale=image_pixscale,
                       download_kwargs=download_kwargs)


@register("desi")
def fetch_desi(
    target: Any,
    *,
    mode: str = "photometry",
    radius_arcsec: float = 3.0,
    ls_release: str = "dr9",
    image_layer: str = "ls-dr10",
    image_format: str = "fits",
    image_size: int = 256,
    image_pixscale: float = 0.262,
    cache_dir: Path | str | None = None,
    dest: Path | str | None = None,
    store_dir: Path | str | None = None,
    download: bool = True,
    progress: Any = None,
    download_kwargs: dict[str, Any] | None = None,
    **download_options: Any,
) -> SurveyProduct:
    """Download (or resolve) a DESI Legacy Surveys product for one source."""
    if mode not in ("photometry", "spectra", "image"):
        raise ValueError(f"unknown DESI mode {mode!r}")
    tabular = mode == "photometry"
    if download and not tabular and dest is None and store_dir is None:
        raise ValueError(
            f"DESI mode={mode!r} returns a file; pass dest= or store_dir= to keep it"
        )

    if download and not tabular:
        # image/spectra are FITS files: validate them on download
        download_options.setdefault("validate", "fits")

    archive = desi_archive(mode=mode, radius_arcsec=radius_arcsec,
                           ls_release=ls_release, cache_dir=cache_dir,
                           image_layer=image_layer, image_format=image_format,
                           image_size=image_size, image_pixscale=image_pixscale,
                           download_kwargs=download_kwargs)
    tmpdir = None
    if download and dest is None and store_dir is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="udong-desi-")
        store_dir = Path(tmpdir.name)

    try:
        item = archive.fetch_one(target, dest=dest, store_dir=store_dir,
                                 download=download, progress=progress,
                                 **download_options)
        if tabular:
            path, table = coerce_product_data(item.data)
        else:
            path = Path(item.data) if isinstance(item.data, (str, Path)) else None
            table = None
        if tmpdir is not None:
            path = None
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return SurveyProduct(
        survey="desi",
        obj_id=item.obj_id,
        path=path,
        table=table,
        meta=dict(item.meta or {}, product=f"desi-{mode}", mode=mode,
                  ls_release=ls_release),
        success=bool(item.success),
        error=item.error or "",
    )


def fetch_cutout(
    ra: float,
    dec: float,
    *,
    dest: Path | str | None = None,
    store_dir: Path | str | None = None,
    size: int = 360,
    pixscale: float = 0.25,
    layer: str = "ls-dr10",
    obj_id: str | None = None,
    progress: Any = None,
    **kwargs: Any,
) -> SurveyProduct:
    """Download a DESI Legacy Surveys optical cutout (FITS, griz) for a position.

    Convenience wrapper around :func:`fetch_desi` with ``mode="image"`` and
    ``image_format="fits"``: one call, one file, one Table-free product::

        p = fetch_cutout(ra, dec, dest="cutout.fits", size=360, pixscale=0.25)
    """
    target = {
        "obj_id": obj_id or f"{float(ra):.6f}{float(dec):+.6f}",
        "raj2000": float(ra),
        "dej2000": float(dec),
    }
    return fetch_desi(
        target,
        mode="image",
        dest=dest,
        store_dir=store_dir,
        image_layer=layer,
        image_format="fits",
        image_size=size,
        image_pixscale=pixscale,
        progress=progress,
        **kwargs,
    )
