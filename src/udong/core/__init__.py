"""Survey-agnostic core abstractions.

Nothing in this subpackage may import data-specific code (e.g., MaNGA).
The public types are: :class:`~udong.core.spectrum.Spectrum`,
:class:`~udong.core.map.Map2D`, :class:`~udong.core.cube.Cube`,
:class:`~udong.core.mask.MaskDefs`, :class:`~udong.core.provenance.Provenance`,
plus uncertainty and coordinate helpers.
"""

from __future__ import annotations

from udong.core.cube import Cube
from udong.core.map import Map2D
from udong.core.mask import Bit, MaskDefs
from udong.core.provenance import ProcessingStep, Provenance
from udong.core.spectrum import Spectrum

__all__ = [
    "Bit",
    "Cube",
    "Map2D",
    "MaskDefs",
    "ProcessingStep",
    "Provenance",
    "Spectrum",
]
