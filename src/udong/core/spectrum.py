"""1-D spectrum container.

``Spectrum`` is a thin, stable wrapper around the specutils spectrum class
(``specutils.Spectrum`` in specutils >= 2.x, previously ``Spectrum1D``).  We
wrap (rather than subclass) so that survey code never depends on specutils
API churn directly; the specutils object remains available through
:attr:`Spectrum.specutils_spectrum` for advanced use.

Conventions
-----------
* ``flux`` is an astropy ``Quantity`` (e.g. ``1e-17 erg/s/cm2/AA``).
* Uncertainty is stored natively as inverse variance where the source
  provides IVAR; :attr:`~Spectrum.ivar` and :attr:`~Spectrum.sigma` convert on
  demand.
* ``mask`` is kept in its original form (raw integer bitmask or boolean
  array).  :attr:`~Spectrum.bad` interprets it when a ``MaskDefs`` registry is
  attached.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy.units import Quantity

from udong.core.mask import MaskDefs
from udong.core.provenance import Provenance
from udong.core.uncertainty import (
    InverseVarianceUncertainty,
    ivar_to_sigma,
    sigma_to_ivar,
)

try:  # specutils >= 2.x
    from specutils import Spectrum as _SpecutilsSpectrum
except ImportError:  # pragma: no cover - older specutils
    from specutils import Spectrum1D as _SpecutilsSpectrum

__all__ = ["Spectrum"]


class Spectrum:
    """A 1-D spectrum with flux, uncertainty, mask, coordinates and metadata.

    ``Spectrum`` is the basic unit of all spectral work in Udong.  It wraps a
    specutils spectrum (see the module docstring) and adds:

    * a stable, survey-agnostic API: ``flux``, ``spectral_axis``, ``ivar`` /
    ``sigma``, ``mask`` / ``bad``, ``meta``, ``provenance``;
    * native handling of MaNGA-style inverse-variance uncertainty
    (``ivar <= 0`` marks invalid pixels);
    * raw mask preservation plus a ``MaskDefs`` registry for interpreting it.

    Spectra are *not* tied to any survey: MaNGA, MUSE or hand-made synthetic
    data all enter through the same constructor.
    """

    def __init__(
        self,
        flux: Quantity,
        spectral_axis: Quantity | None = None,
        uncertainty=None,
        mask=None,
        wcs=None,
        meta: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
        mask_defs: MaskDefs | None = None,
        redshift: float | None = None,
        unit=None,
    ) -> None:
        """Build a spectrum.

        Parameters
        ----------
        flux
            1-D flux-density array; an astropy ``Quantity`` or a plain array
            together with ``unit``.
        spectral_axis
            1-D spectral axis (wavelength, frequency or velocity) as an astropy
            ``Quantity``.  Required.
        uncertainty
            Optional uncertainty: an astropy uncertainty object, a raw IVAR array
            (interpreted as inverse variance), or ``None``.
        mask
            Optional raw integer (or boolean) mask array; interpreted through
            ``mask_defs`` (also passed to the specutils object as a boolean
            "good pixel" mask).
        wcs
            Optional WCS for the spectrum (informational / advanced use).
        meta
            Free-form metadata dict (copied on input).
        provenance
            Optional :class:`~udong.core.provenance.Provenance` record.
        mask_defs
            Optional :class:`~udong.core.mask.MaskDefs` registry.
        redshift
            Optional redshift of the source (used by specutils helpers).
        unit
            Flux unit used when ``flux`` is not already a ``Quantity``.
        """
        if not isinstance(flux, Quantity):
            if unit is None:
                raise TypeError("flux must be an astropy Quantity (or provide unit=)")
            flux = flux * unit
        flux = Quantity(flux)
        if spectral_axis is None:
            raise ValueError("spectral_axis is required")
        if not isinstance(spectral_axis, Quantity):
            raise TypeError("spectral_axis must be an astropy Quantity")

        self._uncertainty = _coerce_uncertainty(uncertainty)
        self._mask = None if mask is None else np.asarray(mask)
        self._meta = dict(meta or {})
        self._provenance = provenance
        self._mask_defs = mask_defs
        self._redshift = redshift

        self._specutils = _SpecutilsSpectrum(
            flux=flux,
            spectral_axis=spectral_axis,
            uncertainty=self._uncertainty,
            mask=None if self._mask is None else (self._mask != 0).astype(bool),
            wcs=wcs,
            meta=self._meta,
            redshift=redshift,
        )
        # keep a private copy of the plain spectral axis for fast slicing
        self._spectral_axis = Quantity(spectral_axis)

    # ------------------------------------------------------------------ #
    # wavelength reference frame
    # ------------------------------------------------------------------ #
    @property
    def wavelength_frame(self) -> str | None:
        """Wavelength reference frame from ``meta``: ``"air"`` or ``"vacuum"``."""
        f = self._meta.get("wavelength_frame")
        return None if f is None else str(f)

    def to_vacuum(self) -> "Spectrum":
        """Return a copy whose wavelength axis is in the **vacuum** frame.

        No-op when the spectrum is already vacuum (or the frame is unknown).
        Use this before comparing against vacuum rest wavelengths (the Udong
        emission-line catalogues are vacuum).  Records a provenance step.
        """
        return self._with_frame("vacuum")

    def to_air(self) -> "Spectrum":
        """Return a copy whose wavelength axis is in the **air** frame."""
        return self._with_frame("air")

    def _with_frame(self, frame: str) -> "Spectrum":
        current = self.wavelength_frame
        if current == frame or current is None:
            return self
        from udong.core.provenance import ProcessingStep
        from udong.core.wavelength import air_to_vacuum, vacuum_to_air
        from dataclasses import replace

        if frame == "vacuum":
            axis = air_to_vacuum(self._spectral_axis)
            op = "air_to_vacuum"
        else:
            axis = vacuum_to_air(self._spectral_axis)
            op = "vacuum_to_air"
        meta = dict(self._meta)
        meta["wavelength_frame"] = frame
        prov = self._provenance
        if prov is not None:
            step = ProcessingStep(name=op, params={"frame_from": current, "frame_to": frame})
            prov = replace(prov, steps=list(prov.steps) + [step])
        return Spectrum(
            flux=self.flux,
            spectral_axis=axis,
            uncertainty=self._uncertainty,
            mask=self._mask,
            meta=meta,
            provenance=prov,
            mask_defs=self._mask_defs,
            redshift=self._redshift,
        )

    # ------------------------------------------------------------------ #
    @property
    def specutils_spectrum(self):
        """The underlying specutils spectrum object (advanced use).

        Use this to access the wider specutils ecosystem (e.g. models, fitting);
        Udong's own API intentionally stays small.
        """
        return self._specutils

    @property
    def spectrum1d(self):
        """Deprecated alias of :attr:`specutils_spectrum` (kept for 
        specutils < 2.x compatibility).
        """
        return self._specutils

    @property
    def flux(self) -> Quantity:
        """Flux-density array as a Quantity."""
        return Quantity(self._specutils.flux)

    @property
    def spectral_axis(self) -> Quantity:
        """Spectral axis (wavelength/frequency/velocity) as a Quantity."""
        return Quantity(self._spectral_axis)

    @property
    def uncertainty(self):
        """The stored astropy uncertainty object (native representation)."""
        return self._uncertainty

    @property
    def ivar(self) -> Quantity | None:
        """Inverse variance as a Quantity (``flux.unit**-2``); ``None`` if unset.

        Converts on demand from whatever representation is stored.
        """
        if self._uncertainty is None:
            return None
        if isinstance(self._uncertainty, InverseVarianceUncertainty):
            a = self._uncertainty.array
        else:
            a = sigma_to_ivar(self._uncertainty.array)
        return Quantity(a, self.flux.unit**-2)

    @property
    def sigma(self) -> Quantity | None:
        """Standard deviation as a Quantity (same unit as the flux)."""
        if self._uncertainty is None:
            return None
        if isinstance(self._uncertainty, InverseVarianceUncertainty):
            a = ivar_to_sigma(self._uncertainty.array)
        else:
            a = np.asarray(self._uncertainty.array, dtype=float)
        return Quantity(a, self.flux.unit)

    @property
    def mask(self) -> np.ndarray | None:
        """Raw mask array (a copy); ``None`` if unset."""
        return None if self._mask is None else self._mask.copy()

    @property
    def mask_defs(self) -> MaskDefs | None:
        """Bit registry used to interpret :attr:`mask` (may be ``None``)."""
        return self._mask_defs

    @property
    def bad(self) -> np.ndarray | None:
        """Boolean bad-pixel flag (``None`` if no mask).

        When a ``MaskDefs`` registry is attached, only the ``DONOTUSE`` bit marks a
        pixel bad; otherwise any non-zero mask value is bad.
        """
        if self._mask is None:
            return None
        if self._mask_defs is not None:
            names = ("DONOTUSE",) if "DONOTUSE" in self._mask_defs else ()
            return self._mask_defs.unmask(self._mask, *names)
        return (self._mask != 0).astype(bool)

    @property
    def unit(self):
        """Flux-density unit of the spectrum."""
        return self.flux.unit

    @property
    def meta(self) -> dict[str, Any]:
        """Free-form metadata dict (shared reference, not a copy)."""
        return self._meta

    @property
    def provenance(self) -> Provenance | None:
        """Provenance record (may be ``None``)."""
        return self._provenance

    @property
    def redshift(self) -> float | None:
        """Redshift assigned to the spectrum (may be ``None``)."""
        return self._redshift

    def __len__(self) -> int:
        """Number of spectral pixels."""
        return len(self._spectral_axis)

    # ------------------------------------------------------------------ #
    def copy(self) -> Spectrum:
        """Return a copy of this spectrum.

        The provenance record is deep-copied; flux/uncertainty/mask arrays and
        metadata are passed by reference (a read-only copy unless you replace the
        underlying arrays yourself).
        """
        return Spectrum(
            flux=self.flux,
            spectral_axis=self.spectral_axis,
            uncertainty=self._uncertainty,
            mask=self._mask,
            meta=self._meta,
            provenance=None if self._provenance is None else self._provenance.copy(),
            mask_defs=self._mask_defs,
            redshift=self._redshift,
        )

    def with_mask(self, additional_bad: np.ndarray) -> Spectrum:
        """Return a copy with additional bad pixels OR-ed into the mask.

        Parameters
        ----------
        additional_bad
            Boolean array of the same length as the spectrum (``True`` = bad).

        Notes
        -----
        With a raw integer mask and a ``MaskDefs`` registry containing ``DONOTUSE``,
        the new bad pixels are encoded by setting that bit; otherwise a boolean mask
        is OR-ed (or bit 0 is set for plain integer masks).  The returned spectrum
        shares the metadata/provenance of this one.
        """
        additional_bad = np.asarray(additional_bad, dtype=bool)
        if additional_bad.shape != (len(self),):
            raise ValueError("additional_bad must have the same length as the spectrum")
        if self._mask is None:
            new_mask = additional_bad.astype(np.uint64) if self._mask_defs is not None else additional_bad
        elif np.issubdtype(self._mask.dtype, np.bool_):
            new_mask = self._mask | additional_bad
        else:
            # raw integer mask: encode "bad" as DONOTUSE when the registry has it
            if self._mask_defs is not None and "DONOTUSE" in self._mask_defs:
                add = np.where(additional_bad, self._mask_defs.value("DONOTUSE"), 0).astype(self._mask.dtype)
            else:
                add = np.where(additional_bad, 1, 0).astype(self._mask.dtype)
            new_mask = self._mask | add
        return Spectrum(
            flux=self.flux,
            spectral_axis=self.spectral_axis,
            uncertainty=self._uncertainty,
            mask=new_mask,
            meta=self._meta,
            provenance=self._provenance,
            mask_defs=self._mask_defs,
            redshift=self._redshift,
        )

    def slice(self, wlo: Quantity, whi: Quantity) -> Spectrum:
        """Return a copy restricted to ``wlo <= wave <= whi`` (inclusive).

        Uncertainty and mask are sliced consistently with the flux.

        Parameters
        ----------
        wlo, whi
            Lower / upper bounds of the spectral window (Quantity).

        Returns
        -------
        Spectrum
            Sub-spectrum over the requested window.
        """
        wave = self.spectral_axis
        sel = (wave >= wlo) & (wave <= whi)
        unc = None
        if self._uncertainty is not None:
            unc = type(self._uncertainty)(self._uncertainty.array[sel])
        msk = None if self._mask is None else self._mask[sel]
        return Spectrum(
            flux=self.flux[sel],
            spectral_axis=wave[sel],
            uncertainty=unc,
            mask=msk,
            meta=self._meta,
            provenance=self._provenance,
            mask_defs=self._mask_defs,
            redshift=self._redshift,
        )


def _coerce_uncertainty(uncertainty):
    """Accept astropy uncertainty objects or raw IVAR arrays."""
    if uncertainty is None:
        return None
    if isinstance(uncertainty, (InverseVarianceUncertainty,)):
        return uncertainty
    # any other astropy uncertainty type
    if hasattr(uncertainty, "array") and hasattr(uncertainty, "uncertainty_type"):
        return uncertainty
    # raw array: interpret as inverse variance
    return InverseVarianceUncertainty(np.asarray(uncertainty, dtype=float))
