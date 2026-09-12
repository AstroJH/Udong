"""Local/remote location for ESO Phase 3 MUSE products.

Layout (ADR-006, decision D2):

    $UDONG_MUSE_ROOT/<dp_id>.fits          local file for a product
    https://dataportal.eso.org/dataPortal/file/<dp_id>    remote URL

No per-release directory tree: an ESO product is a single self-describing
file identified by its ``dp_id``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MusePath", "DATAPORTAL_URL", "valid_dp_id"]

DATAPORTAL_URL = "https://dataportal.eso.org/dataPortal/file"

# Only allow safe file-name characters (dp_id contains '.', ':' and '-'-like
# timestamps, e.g. ``ADP.2024-04-30T18:20:44.624``).
_SAFE = re.compile(r"^[A-Za-z0-9._:-]+$")


def valid_dp_id(dp_id: str) -> bool:
    """True for a plausible, path-safe ESO dataset id."""
    return bool(dp_id) and _SAFE.match(dp_id) is not None


@dataclass(frozen=True)
class MusePath:
    """Builds local/remote paths for MUSE products under a data root."""

    root: Path | str

    def local(self, dp_id: str) -> Path:
        if not valid_dp_id(dp_id):
            raise ValueError(f"invalid dp_id {dp_id!r}")
        return Path(self.root, f"{dp_id}.fits")

    def url(self, dp_id: str) -> str:
        if not valid_dp_id(dp_id):
            raise ValueError(f"invalid dp_id {dp_id!r}")
        return f"{DATAPORTAL_URL}/{dp_id}"
