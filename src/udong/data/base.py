"""Survey-agnostic dataset interface.

Science code should only rely on the methods/properties declared here, so a
future MUSE (or other IFU survey) adapter can be dropped in without touching
``udong.science``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from astropy.table import Table
from udong.core.cube import Cube


@runtime_checkable
class IFUDataset(Protocol):
    """Common interface implemented by survey datasets (e.g. MaNGA)."""

    def load_cube(self, plateifu: str) -> Cube:
        """Load a reduced datacube for an observation identifier."""
        ...

    @property
    def drpall(self) -> Table:
        """Survey summary catalog (per-observation metadata)."""
        ...
