"""easycat ``MUSEArchive`` configuration for Udong.

The Archive knows the ESO dataportal URL layout and the dataportal Range
quirk (``resume="never"`` + FITS validation by default); Udong keeps its own
local layout via ``dest=`` and its own readers.
"""

from __future__ import annotations

from typing import Any

from easycat.download import MUSEArchive

__all__ = ["muse_archive"]


def muse_archive(
    *,
    mode: str = "cube",
    dataportal_base: str = "https://dataportal.eso.org/dataPortal/file",
    download_kwargs: dict[str, Any] | None = None,
) -> MUSEArchive:
    """Return an easycat ``MUSEArchive`` for a MUSE product mode.

    ``mode`` is a provenance marker only (``cube`` / ``whitelight`` /
    ``exmap`` / ``any``) -- the dataportal URL depends solely on the ``dp_id``.
    """
    return MUSEArchive(
        mode=mode,
        dataportal_base=dataportal_base,
        download_kwargs=dict(download_kwargs or {}),
    )
