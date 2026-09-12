"""easycat ``SDSSArchive`` configuration for Udong (MaNGA).

Udong keeps its own on-disk layout and provenance; easycat's Archive is used
for product knowledge (URLs, supported products) and for single-target
downloads via :meth:`SurveyArchive.fetch_one` (atomic ``.part`` downloads,
size/gzip verification, progress reporting).
"""

from __future__ import annotations

from typing import Any

from easycat.download import SDSSArchive

__all__ = ["manga_archive"]


def manga_archive(
    *,
    release: str = "dr17",
    drpver: str = "v3_1_1",
    dapver: str = "3.1.0",
    product: str = "LOGCUBE",
    dap: str | None = None,
    download_kwargs: dict[str, Any] | None = None,
):
    """Return an easycat ``SDSSArchive(mode="manga")`` for this configuration."""
    return SDSSArchive(
        mode="manga",
        manga_product=product,
        manga_dap=dap,
        manga_sas=f"https://data.sdss.org/sas/{release}",
        manga_version=drpver,
        manga_dap_version=dapver,
        download_kwargs=dict(download_kwargs or {}),
    )
