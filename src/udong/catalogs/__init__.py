"""Survey-agnostic catalog layer.

* ``matching`` -- generic sky cross-matching primitives.
"""

from __future__ import annotations

from udong.catalogs.matching import (
    CrossMatchResult,
    crossmatch,
    crossmatch_tables,
    deduplicate,
    match_to_catalog,
)

__all__ = [
    "CrossMatchResult",
    "crossmatch",
    "crossmatch_tables",
    "deduplicate",
    "match_to_catalog",
]
