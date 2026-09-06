"""Generic 3-D IFU datacube container.

Array convention
----------------
The flux array has shape ``(nwave, ny, nx)`` (matching the native MaNGA FITS
layout, where axis 0 is wavelength, axis 1 is the y/dec pixel axis and axis 2
is the x/ra pixel axis).  ``cube.spectrum(x, y)`` extracts
``flux[:, y, x]``.

The wavelength axis is always taken from an explicit wavelength vector
(never reconstructed from header dispersion keywords, which are only
approximate for MaNGA).
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
from udong.core.spectrum import Spectrum

__all__ = ["Cube"]

class Cube:
    """A 3-D IFU datacube with ``flux`` of shape ``(nwave, ny, nx)``.

    ``Cube`` is the survey-agnostic counterpart of a MaNGA LOGCUBE: a
    wavelength-resolved image stack with per-pixel uncertainty (IVAR), raw
    masks, a (3-D) WCS, free-form metadata and provenance.  It knows nothing
    about MaNGA file layout; survey adapters in ``udong.data`` build it.

    Main accessors:
    * :meth:`~Cube.spectrum` / :meth:`~Cube.spectrum_at` extract 1-D
    :class:`~udong.core.spectrum.Spectrum` objects;
    * ``wavelength`` / ``spatial_wcs`` give the spectral and sky axes;
    * index the flux directly with ``cube.flux[:, y, x]`` (wavelength first)
    or use :meth:`~Cube.spectrum`.
    """

    def __init__(
        self,
        flux: Quantity,
        ivar=None,
        mask=None,
        wavelength: Quantity | None = None,
        wcs: WCS | None = None,
        unit=None,
        meta: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
        mask_defs: MaskDefs | None = None,
    ) -> None:
        """Build a datacube.

        Parameters
        ----------
        flux
            3-D flux-density array ``(nwave, ny, nx)``; an astropy ``Quantity``, or
            a plain array together with ``unit``.
        ivar
            Optional inverse-variance array, same shape as ``flux`` (``ivar <= 0``
            marks invalid pixels).
        mask
            Optional raw integer (or boolean) mask array, same shape as ``flux``;
            interpreted through ``mask_defs``.
        wavelength
            Optional 1-D spectral axis (length ``nwave``).  A spectrum can only be
            extracted when this is provided.
        wcs
            Optional WCS (3-D celestial+spectral, or spatial-only; see
            :attr:`spatial_wcs`).
        unit
            Flux unit used when ``flux`` is not already a ``Quantity``.
        meta
            Free-form metadata dict (copied on input).
        provenance
            Optional :class:`~udong.core.provenance.Provenance` record.
        mask_defs
            Optional :class:`~udong.core.mask.MaskDefs` registry for interpreting
            ``mask``.
        """
        if not isinstance(flux, Quantity):
            if unit is None:
                raise TypeError("flux must be an astropy Quantity (or provide unit=)")
            flux = flux * unit
        flux = Quantity(flux)

        if flux.ndim != 3:
            raise ValueError(f"flux must be 3-D (nwave, ny, nx), got {flux.shape}")
        
        self._flux = flux
        self._ivar = None if ivar is None else np.asarray(ivar, dtype=float)
        self._mask = None if mask is None else np.asarray(mask)
        self._wavelength = None if wavelength is None else Quantity(wavelength)
        self._wcs = wcs
        self._meta = dict(meta or {})
        self._provenance = provenance
        self._mask_defs = mask_defs

        nwave = flux.shape[0]
        if self._wavelength is not None and len(self._wavelength) != nwave:
            raise ValueError("wavelength length does not match nwave")
        if self._ivar is not None and self._ivar.shape != flux.shape:
            raise ValueError("ivar shape does not match flux")
        if self._mask is not None and self._mask.shape != flux.shape:
            raise ValueError("mask shape does not match flux")

    # ------------------------------------------------------------------ #
    @property
    def flux(self) -> Quantity:
        """Flux-density cube ``(nwave, ny, nx)`` as a Quantity."""
        return Quantity(self._flux)

    @property
    def ivar(self) -> np.ndarray | None:
        """Inverse-variance cube (unitless ndarray); ``None`` if unset."""
        return None if self._ivar is None else self._ivar.copy()

    @property
    def mask(self) -> np.ndarray | None:
        """Raw mask cube; ``None`` if unset (a copy is returned)."""
        return None if self._mask is None else self._mask.copy()

    @property
    def mask_defs(self) -> MaskDefs | None:
        """Bit registry used to interpret :attr:`mask` (may be ``None``)."""
        return self._mask_defs

    @property
    def wavelength(self) -> Quantity | None:
        """1-D spectral axis (e.g. vacuum wavelengths in Angstrom)."""
        return None if self._wavelength is None else Quantity(self._wavelength)

    @property
    def wcs(self) -> WCS | None:
        """The stored (possibly 3-D) WCS object."""
        return self._wcs

    @property
    def spatial_wcs(self) -> WCS | None:
        """2-D spatial (RA/Dec) WCS derived from the stored 3-D WCS.

        Returns ``None`` when the cube carries no WCS.  Used by :meth:`spectrum_at`
        and by map-making code.
        """
        if self._wcs is None:
            return None
        try:
            return self._wcs.sub(["longitude", "latitude"])
        except Exception:
            return self._wcs.sub(2)

    @property
    def unit(self):
        """Flux-density unit of the cube (from the flux Quantity)."""
        return self._flux.unit

    @property
    def meta(self) -> dict[str, Any]:
        """Free-form metadata dict (shared reference, not a copy)."""
        return self._meta

    @property
    def provenance(self) -> Provenance | None:
        """Provenance record attached to this cube (may be ``None``)."""
        return self._provenance

    @property
    def shape(self) -> tuple[int, int, int]:
        """Array shape ``(nwave, ny, nx)``."""
        return self._flux.shape

    @property
    def nwave(self) -> int:
        """Number of wavelength channels."""
        return self._flux.shape[0]

    @property
    def ny(self) -> int:
        """Number of pixels along the y (Dec) axis."""
        return self._flux.shape[1]

    @property
    def nx(self) -> int:
        """Number of pixels along the x (RA) axis."""
        return self._flux.shape[2]

    # ------------------------------------------------------------------ #
    def spectrum(self, x: int, y: int) -> Spectrum:
        """Extract the spectrum at integer pixel ``(x, y)`` as a Spectrum.

        The returned :class:`~udong.core.spectrum.Spectrum` inherits the cube's
        wavelength vector, uncertainty (as inverse variance), mask (with the same
        ``mask_defs``), metadata and provenance.  Pixel coordinates are added to
        ``meta["x"]`` / ``meta["y"]``.

        Parameters
        ----------
        x, y
            Integer pixel coordinates (``flux[:, y, x]``).

        Returns
        -------
        Spectrum
            Requires the cube to have a wavelength vector.
        """
        if not (0 <= int(x) < self.nx and 0 <= int(y) < self.ny):
            raise IndexError(f"(x, y)=({x}, {y}) out of bounds ({self.nx} x {self.ny})")
        x, y = int(x), int(y)
        if self._wavelength is None:
            raise RuntimeError("cube has no wavelength vector; cannot build Spectrum")
        unc = None
        if self._ivar is not None:
            from udong.core.uncertainty import InverseVarianceUncertainty

            unc = InverseVarianceUncertainty(self._ivar[:, y, x])
        msk = None if self._mask is None else self._mask[:, y, x]
        meta = dict(self._meta)
        meta.setdefault("x", x)
        meta.setdefault("y", y)
        return Spectrum(
            flux=self._flux[:, y, x],
            spectral_axis=self._wavelength,
            uncertainty=unc,
            mask=msk,
            meta=meta,
            provenance=self._provenance,
            mask_defs=self._mask_defs,
        )

    def spectrum_at(self, coord: SkyCoord) -> Spectrum:
        """Extract the spectrum nearest to a sky coordinate.

        Parameters
        ----------
        coord
            Target sky position.

        Returns
        -------
        Spectrum
            Spectrum of the spaxel nearest to ``coord`` (rounds the fractional pixel
            position).

        Raises
        ------
        RuntimeError
            If the cube has no spatial WCS.
        """
        wcs2d = self.spatial_wcs
        if wcs2d is None:
            raise RuntimeError("cube has no spatial WCS; use spectrum(x, y)")
        
        x, y = wcs2d.world_to_pixel(coord)
        return self.spectrum(int(round(float(x))), int(round(float(y))))
