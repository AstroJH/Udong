"""WISE/AllWISE photometry via easycat's ``WISEArchive``."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from udong.catalogs.surveys import (
    SurveyProduct,
    coerce_product_data,
    register,
)

__all__ = ["wise_archive", "fetch_wise"]


def wise_archive(*, radius_arcsec: float = 3.0, store_format: str = "fits",
                 download_kwargs: dict[str, Any] | None = None):
    """Configured easycat ``WISEArchive`` (monkeypatchable in tests)."""
    from easycat.download import WISEArchive

    return WISEArchive(radius_arcsec=radius_arcsec, store_format=store_format,
                       download_kwargs=download_kwargs)


@register("wise")
def fetch_wise(
    target: Any,
    *,
    radius_arcsec: float = 3.0,
    store_format: str = "fits",
    dest: Path | str | None = None,
    store_dir: Path | str | None = None,
    download: bool = True,
    progress: Any = None,
    download_kwargs: dict[str, Any] | None = None,
    **download_options: Any,
) -> SurveyProduct:
    """Download (or resolve) the WISE photometry product for one source.

    Without ``dest``/``store_dir`` the file is fetched into a temporary
    directory, parsed into ``SurveyProduct.table`` and removed afterwards.
    """
    archive = wise_archive(radius_arcsec=radius_arcsec, store_format=store_format,
                           download_kwargs=download_kwargs)
    tmpdir = None
    if download and dest is None and store_dir is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="udong-wise-")
        store_dir = Path(tmpdir.name)

    try:
        item = archive.fetch_one(target, dest=dest, store_dir=store_dir,
                                 download=download, progress=progress,
                                 **download_options)
        path, table = coerce_product_data(item.data)
        if tmpdir is not None:
            path = None  # temporary file removed below
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return SurveyProduct(
        survey="wise",
        obj_id=item.obj_id,
        path=path,
        table=table,
        meta=dict(item.meta or {}, product="wise-photometry"),
        success=bool(item.success),
        error=item.error or "",
    )
