"""Basic spectral operations (spectroscopy package).

Masked band integration, rest-frame shifting, masking.  Line-profile moments
(M0/M1/M2) and continuum subtraction live in :mod:`udong.science.spectroscopy.moments`.
"""

from __future__ import annotations

import numpy as np
from astropy.units import Quantity

from udong.core.spectrum import Spectrum
from udong.core.uncertainty import propagate_sum

__all__ = ["integrate_band", "to_rest_frame", "mask_wavelength_range"]


def integrate_band(spectrum: Spectrum, wlo: Quantity, whi: Quantity):
    """Integrate flux over ``[wlo, whi]`` -> ``(flux, flux_ivar)``.

    The wavelength grid is assumed uniform (true for MaNGA LOG cubes only to
    first order; a proper trapezoidal integration is planned for later phase).
    Bad pixels (``spectrum.bad``) are excluded.
    """
    sel = (spectrum.spectral_axis >= wlo) & (spectrum.spectral_axis <= whi)
    flux = np.asarray(spectrum.flux.value)[sel]
    bad = np.zeros(flux.shape, dtype=bool)
    if spectrum.bad is not None:
        bad = np.asarray(spectrum.bad)[sel]
    ivar = None
    if spectrum.ivar is not None:
        ivar = np.asarray(spectrum.ivar.value)[sel]
    if ivar is None:
        ivar = np.where(bad, 0.0, 1.0 / (flux**2 + 1e-300))  # placeholder when no errors
    s, s_ivar = propagate_sum(flux, np.where(bad, 0.0, ivar))
    # flux per pixel integrated over uniform pixel width
    if len(flux) == 0:
        return Quantity(np.nan, spectrum.flux.unit * spectrum.spectral_axis.unit), Quantity(0.0, 1 / (spectrum.flux.unit * spectrum.spectral_axis.unit) ** 2)
    dw = np.abs(np.diff(spectrum.spectral_axis.value[sel]))[0] * spectrum.spectral_axis.unit
    flux_int = Quantity(s, spectrum.flux.unit) * dw
    ivar_int = Quantity(s_ivar, (spectrum.flux.unit * dw) ** -2)
    return flux_int, ivar_int


def to_rest_frame(spectrum: Spectrum, z: float) -> Spectrum:
    """Shift a spectrum to its rest frame (``wave_rest = wave_obs / (1+z)``)."""
    return Spectrum(
        flux=spectrum.flux,
        spectral_axis=spectrum.spectral_axis / (1.0 + z),
        uncertainty=spectrum.uncertainty,
        mask=spectrum.mask,
        meta=dict(spectrum.meta, redshift=z),
        provenance=spectrum.provenance,
        mask_defs=spectrum.mask_defs,
        redshift=z,
    )


def mask_wavelength_range(spectrum: Spectrum, wlo: Quantity, whi: Quantity) -> Spectrum:
    """Return a copy with ``[wlo, whi]`` flagged as bad."""
    sel = (spectrum.spectral_axis >= wlo) & (spectrum.spectral_axis <= whi)
    return spectrum.with_mask(np.asarray(sel))
