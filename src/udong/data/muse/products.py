"""MUSE product containers: :class:`MuseCube`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from astropy import units as u
from astropy.units import Quantity
from astropy.wcs import WCS

from udong.core.cube import Cube
from udong.core.provenance import Provenance

__all__ = ["MuseCube"]


class MuseCube(Cube):
    """An ESO Phase 3 MUSE datacube with survey-specific extras.

    Array convention is inherited from :class:`~udong.core.cube.Cube`:
    ``flux`` has shape ``(nwave, ny, nx)``.  The wavelength axis is air
    wavelength (linear, 1.25 A/pixel for WFM) unless stated otherwise in
    :attr:`wavelength_frame`.
    """

    def __init__(
        self,
        flux: Quantity,
        ivar=None,
        mask=None,
        wavelength: Quantity | None = None,
        wcs: WCS | None = None,
        meta: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
        wavelength_frame: str = "air",
    ) -> None:
        super().__init__(
            flux=flux,
            ivar=ivar,
            mask=mask,
            wavelength=wavelength,
            wcs=wcs,
            meta=meta,
            provenance=provenance,
        )
        self._wavelength_frame = wavelength_frame

    # -- convenience metadata ------------------------------------------------ #
    @property
    def wavelength_frame(self) -> str:
        """Reference frame of :attr:`wavelength`: ``"air"`` or ``"vacuum"``."""
        return self._wavelength_frame

    @property
    def dp_id(self) -> str:
        """ESO archive dataset id (e.g. ``ADP.2024-04-30T18:20:44.624``)."""
        return str(self._meta.get("dp_id", ""))

    @property
    def target_name(self) -> str:
        """Target name as recorded in the primary header (may be empty)."""
        return str(self._meta.get("TARGNAME", ""))

    @property
    def mode(self) -> str:
        """Observing mode (e.g. ``WFM-AO-N``)."""
        return str(self._meta.get("INSMODE", ""))

    @property
    def sky_res(self) -> Quantity | None:
        """Official spatial resolution estimate ``SKY_RES`` (arcsec)."""
        v = self._meta.get("SKY_RES")
        return None if v is None else Quantity(float(v), u.arcsec)

    @property
    def ncombine(self) -> int | None:
        """Number of exposures combined into the cube."""
        v = self._meta.get("NCOMBINE")
        return None if v is None else int(v)

    @property
    def exptime(self) -> Quantity | None:
        """Total on-source exposure time (s)."""
        v = self._meta.get("EXPTIME")
        return None if v is None else Quantity(float(v), u.s)

    @property
    def pipeline_version(self) -> str:
        """MUSE pipeline version used to produce the cube."""
        return str(self._meta.get("PIPEVERS", ""))
