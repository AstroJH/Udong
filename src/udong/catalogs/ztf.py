"""ZTF light curves via easycat's ``ZTFArchive``."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from udong.catalogs.surveys import (
    SurveyProduct,
    coerce_product_data,
    register,
)

__all__ = ["ztf_archive", "fetch_ztf"]


def ztf_archive(*, bands: str = "g,r,i", radius_arcsec: float = 3.0,
                max_objects: int | None = None, store_format: str = "csv",
                download_kwargs: dict[str, Any] | None = None):
    """Configured easycat ``ZTFArchive`` (monkeypatchable in tests)."""
    from easycat.download import ZTFArchive

    return ZTFArchive(bands=bands, radius_arcsec=radius_arcsec,
                      max_objects=max_objects, store_format=store_format,
                      download_kwargs=download_kwargs)


@register("ztf")
def fetch_ztf(
    target: Any,
    *,
    bands: str = "g,r,i",
    radius_arcsec: float = 3.0,
    max_objects: int | None = None,
    store_format: str = "csv",
    dest: Path | str | None = None,
    store_dir: Path | str | None = None,
    download: bool = True,
    progress: Any = None,
    download_kwargs: dict[str, Any] | None = None,
    **download_options: Any,
) -> SurveyProduct:
    """Download (or resolve) the ZTF light curve for one source.

    The CSV light curve is parsed into ``SurveyProduct.table``.
    """
    archive = ztf_archive(bands=bands, radius_arcsec=radius_arcsec,
                          max_objects=max_objects, store_format=store_format,
                          download_kwargs=download_kwargs)
    tmpdir = None
    if download and dest is None and store_dir is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="udong-ztf-")
        store_dir = Path(tmpdir.name)

    try:
        item = archive.fetch_one(target, dest=dest, store_dir=store_dir,
                                 download=download, progress=progress,
                                 **download_options)
        path, table = coerce_product_data(item.data)
        if tmpdir is not None:
            path = None
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return SurveyProduct(
        survey="ztf",
        obj_id=item.obj_id,
        path=path,
        table=table,
        meta=dict(item.meta or {}, product="ztf-lightcurve", bands=bands),
        success=bool(item.success),
        error=item.error or "",
    )
