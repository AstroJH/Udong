"""Bitmask handling.

MaNGA (and most survey data) stores data-quality information as integer
bitmasks (one bit per flag, e.g. ``DONOTUSE``, ``LOWCOV``, ...).  The design
rules in ``udong.core`` are:

* **raw masks are never discarded** -- containers keep the original integer
array so that every flag can be recovered later;
* survey-specific bit definitions live in the *data layer*
(``udong.data.manga.masks``), never here;

while this module provides the generic registry (:class:`Bit`,
:class:`MaskDefs`) and selection helpers (:meth:`MaskDefs.unmask` /
:meth:`MaskDefs.good`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["Bit", "MaskDefs"]


@dataclass(frozen=True)
class Bit:
    """One named bit of a mask group.

    Attributes
    ----------
    name
        Short machine-readable flag name (e.g. ``"DONOTUSE"``).
    position
        0-based bit position; the integer value of the flag is
        ``1 << position``.
    description
        Human-readable meaning of the flag.
    """

    name: str
    position: int  # 0-based bit position; the value is 1 << position
    description: str = ""


class MaskDefs:
    """Registry of named bits for a single mask group (e.g. ``MANGA_DAPPIXMASK``).

    Parameters
    ----------
    name
        Group name, e.g. ``"MANGA_DAPPIXMASK"``.
    bits
        Iterable of :class:`Bit` (or of ``(position, name, description)``
        tuples / plain strings, in which case positions are assigned
        sequentially).
    dtype
        numpy dtype of the stored masks, e.g. ``"uint32"`` or ``"uint64"``.
    description
        Optional description of the mask group.
    source
        Optional provenance note (URL / file) for the bit definitions.
    """

    def __init__(
        self,
        name: str,
        bits: Iterable[Bit | tuple[int, str, str] | str],
        dtype: str = "uint64",
        description: str = "",
        source: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.source = source
        self._dtype = np.dtype(dtype)
        self._bits: dict[str, Bit] = {}
        for i, b in enumerate(bits):
            if isinstance(b, str):
                b = Bit(name=b, position=i)
            elif isinstance(b, tuple):
                b = Bit(name=b[1], position=b[0], description=b[2])
            self._bits[b.name] = b

    # ------------------------------------------------------------------ #
    @property
    def dtype(self) -> np.dtype:
        """numpy dtype of the stored integer masks."""
        return self._dtype

    @property
    def bits(self) -> dict[str, Bit]:
        """A copy of the ``{name: Bit}`` registry."""
        return dict(self._bits)

    def __contains__(self, name: object) -> bool:
        """True when ``name`` is a defined bit."""
        return name in self._bits

    def __iter__(self):
        """Iterate over the defined bit names."""
        return iter(self._bits)

    def __len__(self) -> int:
        """Number of defined bits."""
        return len(self._bits)

    def __getitem__(self, name: str) -> Bit:
        """Return the :class:`Bit` registered under ``name``."""
        return self._bits[name]

    def names(self) -> list[str]:
        """List of defined bit names in insertion order."""
        return list(self._bits)

    def value(self, name: str) -> int:
        """Integer value (``1 << position``) of a named bit."""
        return 1 << self._bits[name].position

    # ------------------------------------------------------------------ #
    def unmask(
        self,
        masks: np.ndarray,
        *names: str,
        keep_any: bool = True,
        keep_all: bool = False,
    ) -> np.ndarray:
        """Return a boolean array flagging pixels with the named bits set.

        If no names are given, flags every non-zero pixel.  With
        ``keep_any=True`` (default) a pixel is bad if *any* requested bit is
        set; with ``keep_all=True`` all requested bits must be set.
        """
        masks = np.asarray(masks)
        if not names:
            return masks != 0
        flag = 0
        for n in names:
            if n not in self._bits:
                raise KeyError(f"bit {n!r} not defined in {self.name}")
            flag |= self.value(n)
        if keep_all:
            return (masks & flag) == flag
        return (masks & flag) != 0

    def good(
        self,
        masks: np.ndarray,
        *names: str,
        keep_any: bool = True,
        keep_all: bool = False,
    ) -> np.ndarray:
        """Inverse of :meth:`unmask`: True where none of the named bits is set.

        See :meth:`unmask` for the meaning of the arguments.
        """
        return ~self.unmask(masks, *names, keep_any=keep_any, keep_all=keep_all)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Serialize the registry to a JSON/YAML-friendly dict."""
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "dtype": str(self._dtype),
            "bits": [
                {"position": b.position, "name": b.name, "description": b.description}
                for b in self._bits.values()
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> MaskDefs:
        """Re-build a :class:`MaskDefs` from the dict of :meth:`to_dict`."""
        return cls(
            name=d["name"],
            bits=[Bit(**b) for b in d["bits"]],
            dtype=d.get("dtype", "uint64"),
            description=d.get("description", ""),
            source=d.get("source"),
        )
