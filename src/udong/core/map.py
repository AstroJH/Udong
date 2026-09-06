"""2-D map container.

There is no standard astropy 2-D "map" container, so ``Map2D`` is a thin,
explicit wrapper around numpy arrays + an astropy WCS.  The same conventions
as :class:`~udong.core.spectrum.Spectrum` apply: native IVAR uncertainty,
raw masks interpreted through a ``MaskDefs`` registry, and full metadata +
provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS

from udong.core.mask import MaskDefs
from udong.core.provenance import Provenance
from udong.core.uncertainty import (
    InverseVarianceUncertainty,
    ivar_to_sigma,
    sigma_to_ivar,
)

__all__ = ["Map2D"]


class Map2D:
    """A 2-D map of a measured quantity with uncertainty, mask, WCS, metadata.

    ``Map2D`` is the 2-D sibling of :class:`~udong.core.spectrum.Spectrum` (and
    :class:`~udong.core.cube.Cube`): a quantity per spatial pixel, an
    IVAR/native uncertainty, raw masks interpreted through a ``MaskDefs``
    registry, a (spatial) WCS, metadata and provenance.  Science and viz code
    consume ``Map2D`` so that survey adapters only need to produce it.
    """

    def __init__(
        self,
        value: Quantity,
        uncertainty=None,
        mask=None,
        wcs: WCS | None = None,
        unit=None,
        meta: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
        mask_defs: MaskDefs | None = None,
        channel: str | None = None,
        channel_unit=None,
    ) -> None:
        """Build a 2-D map.

        Parameters
        ----------
        value
            2-D array of measured values; an astropy ``Quantity``, or a plain array
            together with ``unit``.
        uncertainty
            Optional uncertainty: an astropy uncertainty object, a raw IVAR array
            (interpreted as inverse variance), or ``None``.
        mask
            Optional raw integer (or boolean) mask array, same shape as ``value``;
            interpreted through ``mask_defs``.
        wcs
            Optional 2-D celestial WCS.
        unit
            Value unit used when ``value`` is not already a ``Quantity``.
        meta
            Free-form metadata dict (copied on input).
        provenance
            Optional :class:`~udong.core.provenance.Provenance` record.
        mask_defs
            Optional :class:`~udong.core.mask.MaskDefs` registry.
        channel
            Optional channel/quantity name (e.g. ``"EMLINE_GFLUX"``) this map was
            extracted from.
        channel_unit
            Optional per-channel unit annotation (informational).
        """

        if not isinstance(value, Quantity):
            if unit is None:
                raise TypeError("value must be an astropy Quantity (or provide unit=)")
            value = value * unit

        if value.ndim != 2:
            raise ValueError(f"value must be 2-D, got shape {value.shape}")

        self._value = Quantity(value)
        self._uncertainty = _coerce_uncertainty(uncertainty)
        self._mask = None if mask is None else np.asarray(mask)
        self._wcs = wcs
        self._meta = dict(meta or {})
        self._provenance = provenance
        self._mask_defs = mask_defs
        self._channel = channel
        self._channel_unit = channel_unit

    # ------------------------------------------------------------------ #
    @property
    def value(self) -> Quantity:
        """2-D value array as a Quantity."""
        return Quantity(self._value)

    @property
    def unit(self):
        """Unit of the mapped quantity."""
        return self._value.unit

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape ``(ny, nx)``."""
        return self._value.shape

    @property
    def uncertainty(self):
        """The stored astropy uncertainty object (native representation)."""
        return self._uncertainty

    @property
    def ivar(self) -> Quantity | None:
        """Inverse variance as a Quantity (``unit**-2``); ``None`` if unset.

        Converts on demand from whatever representation is stored (native IVAR or
        sigma-based).
        """
        if self._uncertainty is None:
            return None
        if isinstance(self._uncertainty, InverseVarianceUncertainty):
            a = self._uncertainty.array
        else:
            a = sigma_to_ivar(self._uncertainty.array)
        return Quantity(a, self._value.unit**-2)

    @property
    def sigma(self) -> Quantity | None:
        """Standard deviation as a Quantity (same unit as the value)."""
        if self._uncertainty is None:
            return None
        if isinstance(self._uncertainty, InverseVarianceUncertainty):
            a = ivar_to_sigma(self._uncertainty.array)
        else:
            a = np.asarray(self._uncertainty.array, dtype=float)
        return Quantity(a, self._value.unit)

    @property
    def mask(self) -> np.ndarray | None:
        """Raw mask array (a copy); ``None`` if unset."""
        return None if self._mask is None else self._mask.copy()

    @property
    def mask_defs(self) -> MaskDefs | None:
        """Bit registry used to interpret :attr:`mask` (may be ``None``)."""
        return self._mask_defs

    @property
    def bad(self) -> np.ndarray:
        """Boolean bad-pixel flag (``False`` everywhere if no mask).

        When a ``MaskDefs`` registry is attached, only the ``DONOTUSE`` bit makes a
        pixel bad; otherwise any non-zero mask value is bad.
        """
        if self._mask is None:
            return np.zeros(self._value.shape, dtype=bool)
        if self._mask_defs is not None:
            names = ("DONOTUSE",) if "DONOTUSE" in self._mask_defs else ()
            return self._mask_defs.unmask(self._mask, *names)
        return (self._mask != 0).astype(bool)

    @property
    def wcs(self) -> WCS | None:
        """2-D celestial WCS (may be ``None`` for plain grids)."""
        return self._wcs

    @property
    def meta(self) -> dict[str, Any]:
        """Free-form metadata dict (shared reference, not a copy)."""
        return self._meta

    @property
    def provenance(self) -> Provenance | None:
        """Provenance record (may be ``None``)."""
        return self._provenance

    @property
    def channel(self) -> str | None:
        """Optional channel/quantity name this map was extracted from."""
        return self._channel

    # ------------------------------------------------------------------ #
    def masked_value(self, fill: float = np.nan) -> np.ndarray:
        """Value array with bad pixels replaced by ``fill``.

        Useful for quick ``imshow`` / contour plotting without touching the
        underlying data.

        Parameters
        ----------
        fill
            Replacement value for bad pixels (default ``np.nan``).

        Returns
        -------
        numpy.ndarray
            Plain (unitless) float array of shape ``self.shape``.
        """
        v = np.asarray(self._value.value)
        v = v.copy()
        v[self.bad] = fill
        return v

    def pixel_to_sky(self, x, y) -> SkyCoord:
        """Convert pixel coordinates to sky coordinates (requires WCS).

        Parameters
        ----------
        x, y
            Pixel coordinates (floats allowed for sub-pixel positions).

        Returns
        -------
        SkyCoord

        Raises
        ------
        RuntimeError
            If the map has no WCS.
        """
        if self._wcs is None:
            raise RuntimeError("Map2D has no WCS")
        return self._wcs.pixel_to_world(x, y)

    def sky_to_pixel(self, coord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
        """Convert a sky coordinate to pixel coordinates (requires WCS).

        Parameters
        ----------
        coord
            Sky position.

        Returns
        -------
        (x, y)
            Pixel coordinates as float arrays.

        Raises
        ------
        RuntimeError
            If the map has no WCS.
        """
        if self._wcs is None:
            raise RuntimeError("Map2D has no WCS")
        return self._wcs.world_to_pixel(coord)

    def copy(self) -> Map2D:
        """Return a copy of this map.

        The provenance record is deep-copied; value/uncertainty/mask arrays and
        metadata are passed through by reference (treat the copy as read-only if you
        need to guarantee isolation).
        """
        return Map2D(
            value=self._value,
            uncertainty=self._uncertainty,
            mask=self._mask,
            wcs=self._wcs,
            meta=self._meta,
            provenance=None if self._provenance is None else self._provenance.copy(),
            mask_defs=self._mask_defs,
            channel=self._channel,
            channel_unit=self._channel_unit,
        )


def _coerce_uncertainty(uncertainty):
    if uncertainty is None:
        return None
    if isinstance(uncertainty, InverseVarianceUncertainty):
        return uncertainty
    if hasattr(uncertainty, "array") and hasattr(uncertainty, "uncertainty_type"):
        return uncertainty
    return InverseVarianceUncertainty(np.asarray(uncertainty, dtype=float))
