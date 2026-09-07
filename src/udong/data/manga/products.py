"""MaNGA product containers: :class:`MangaCube` and :class:`MangaMaps`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from astropy.units import Quantity
from astropy.wcs import WCS

from udong.core.cube import Cube
from udong.core.map import Map2D
from udong.core.mask import MaskDefs
from udong.core.provenance import Provenance
from udong.data.manga.channels import resolve_emission_line
from udong.data.manga.units import manga_unit

__all__ = ["MangaCube", "MangaMaps"]


class MangaCube(Cube):
    """A MaNGA DRP LOGCUBE with survey-specific extras (LSF etc.)."""

    def __init__(
        self,
        flux: Quantity,
        ivar=None,
        mask=None,
        wavelength: Quantity | None = None,
        wcs: WCS | None = None,
        meta: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
        mask_defs: MaskDefs | None = None,
        lsf_pre=None,
        lsf_post=None,
    ) -> None:
        super().__init__(
            flux=flux,
            ivar=ivar,
            mask=mask,
            wavelength=wavelength,
            wcs=wcs,
            meta=meta,
            provenance=provenance,
            mask_defs=mask_defs,
        )
        self._lsf_pre = lsf_pre
        self._lsf_post = lsf_post

    @property
    def lsf_pre(self):
        """Pre-pixellized line-spread function (per spaxel / wavelength)."""
        return self._lsf_pre

    @property
    def lsf_post(self):
        """Post-pixellized line-spread function."""
        return self._lsf_post

    # -- convenience metadata ------------------------------------------------- #
    @property
    def plateifu(self) -> str:
        return str(self._meta.get("PLATEIFU", ""))

    @property
    def mangaid(self) -> str:
        return str(self._meta.get("MANGAID", ""))

    @property
    def redshift(self) -> float | None:
        z = self._meta.get("Z")
        return None if z is None else float(z)

    @property
    def drp3qual(self) -> int | None:
        q = self._meta.get("DRP3QUAL")
        return None if q is None else int(q)


class MangaMaps:
    """Container for all quantities in a DAP MAPS file.

    * 2-D quantities (e.g. ``STELLAR_VEL``) are exposed directly as
      :class:`~udong.core.map.Map2D` via ``maps["STELLAR_VEL"]``.
    * Multi-channel quantities (e.g. ``EMLINE_GFLUX`` with 35 channels) are
      selected with :meth:`channel`, or through the convenience accessors
      :meth:`emission_line_flux` etc.
    """

    def __init__(
        self,
        maps: Mapping[str, Map2D],
        channel_maps: Mapping[str, dict[str, Any]] | None = None,
        meta: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
        mask_defs: MaskDefs | None = None,
    ) -> None:
        self._maps: dict[str, Map2D] = dict(maps)
        self._channel_maps: dict[str, dict[str, Any]] = dict(channel_maps or {})
        self._meta = dict(meta or {})
        self._provenance = provenance
        self._mask_defs = mask_defs

    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> Map2D:
        try:
            return self._maps[key]
        except KeyError:
            raise KeyError(
                f"{key!r} not in MAPS. Available: {sorted(self.keys())}"
            ) from None

    def __contains__(self, key: object) -> bool:
        return key in self._maps

    def __iter__(self) -> Iterator[str]:
        return iter(self._maps)

    def keys(self) -> list[str]:
        return sorted(self._maps)

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def provenance(self) -> Provenance | None:
        return self._provenance

    @property
    def plateifu(self) -> str:
        return str(self._meta.get("PLATEIFU", ""))

    @property
    def mangaid(self) -> str:
        return str(self._meta.get("MANGAID", ""))

    @property
    def daptype(self) -> str:
        return str(self._meta.get("DAPTYPE", ""))

    @property
    def drp3qual(self) -> int | None:
        q = self._meta.get("DRP3QUAL")
        return None if q is None else int(q)

    @property
    def dapqual(self) -> int | None:
        q = self._meta.get("DAPQUAL")
        return None if q is None else int(q)

    # ------------------------------------------------------------------ #
    def channel(self, extname: str, channel: str) -> Map2D:
        """Select a single channel from a multi-channel extension as Map2D."""
        cm = self._channel_maps.get(extname)
        if cm is None:
            raise KeyError(f"no multi-channel extension {extname!r}")
        names = cm["channels"]
        try:
            idx = names.index(channel)
        except ValueError:
            raise KeyError(
                f"channel {channel!r} not in {extname}; available: {names}"
            ) from None
        units = cm.get("channel_units") or [None] * len(names)
        unit = manga_unit(units[idx]) if units[idx] is not None else cm["bunit"]
        ivar = cm["ivar"][idx] if cm["ivar"] is not None else None
        mask = cm["mask"][idx] if cm["mask"] is not None else None
        return Map2D(
            value=cm["value"][idx] * unit,
            uncertainty=ivar,
            mask=mask,
            wcs=cm["wcs"],
            meta=dict(cm["meta"], channel=channel),
            provenance=cm.get("provenance"),
            mask_defs=self._mask_defs,
            channel=channel,
            channel_unit=unit,
        )

    def emission_line_flux(self, line: str) -> Map2D:
        """Gaussian-fit line flux (``EMLINE_GFLUX``) for an emission line."""
        return self.channel("EMLINE_GFLUX", resolve_emission_line(line))

    def emission_line_ew(self, line: str) -> Map2D:
        """Gaussian-fit equivalent width (``EMLINE_GEW``)."""
        return self.channel("EMLINE_GEW", resolve_emission_line(line))

    def emission_line_gvel(self, line: str) -> Map2D:
        """Gaussian-fit line-of-sight velocity (``EMLINE_GVEL``)."""
        return self.channel("EMLINE_GVEL", resolve_emission_line(line))

    def emission_line_gsigma(self, line: str) -> Map2D:
        """Gaussian-fit velocity dispersion (``EMLINE_GSIGMA``)."""
        return self.channel("EMLINE_GSIGMA", resolve_emission_line(line))

    def emission_line_sflux(self, line: str) -> Map2D:
        """Summed (passband) line flux (``EMLINE_SFLUX``)."""
        return self.channel("EMLINE_SFLUX", resolve_emission_line(line))

    def spectral_index(self, name: str) -> Map2D:
        """A spectral index (``SPECINDEX``), with per-channel units applied."""
        return self.channel("SPECINDEX", name)

    @property
    def emission_lines(self) -> list[str]:
        """Emission-line channel names present in this file."""
        cm = self._channel_maps.get("EMLINE_GFLUX")
        if cm is None:
            return []
        return list(cm["channels"])
