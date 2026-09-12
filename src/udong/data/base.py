"""Survey-agnostic dataset interfaces.

Science code should only rely on the cube-centric :class:`IFUDataset`
interface, so a future survey adapter (e.g. MUSE) can be dropped in without
touching ``udong.science``: every IFU survey can return a
:class:`~udong.core.cube.Cube` from a data-layer identifier
(MaNGA ``plateifu``, MUSE ``dp_id``, ...).

Surveys that additionally provide a *local* per-observation summary catalog
(e.g. MaNGA drpall) implement the optional :class:`CatalogDataset` protocol.
It is intentionally separate: MUSE has no single local summary table (ESO's
ObsCore is a remote VO service and belongs to the catalog layer), so MUSE
adapters implement ``IFUDataset`` only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from astropy.table import Table
from udong.core.cube import Cube


@runtime_checkable
class IFUDataset(Protocol):
    """Cube-centric interface implemented by IFU survey datasets.

    ``identifier`` is a survey-specific observation/product id (MaNGA
    ``plateifu`` such as ``"8485-1901"``; MUSE ESO ``dp_id`` such as
    ``"ADP.2024-04-30T18:20:44.624"``).  Resolving human-readable target
    names to identifiers is a catalog-layer concern, not part of this
    protocol.
    """

    def load_cube(self, identifier: str, **kwargs) -> Cube:
        """Load a reduced datacube for a survey observation identifier."""
        ...


@runtime_checkable
class CatalogDataset(Protocol):
    """Optional interface for surveys with a local summary catalog."""

    @property
    def drpall(self) -> Table:
        """Survey summary catalog (per-observation metadata)."""
        ...
