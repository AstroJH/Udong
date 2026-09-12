"""Air / vacuum wavelength conversions (survey frame handling).

Surveys disagree on the reference frame of their wavelength axes:

* MaNGA DRP WAVE is **vacuum** (the DAP line catalogue is vacuum too);
* MUSE pipeline cubes are **air** (``CTYPE3 = AWAV``), ~0.03% shorter.

A 0.028% shift is ~85 km/s at H-alpha, so comparing an air-frame MUSE
spectrum against vacuum rest wavelengths *without converting* would bias
every measured velocity by ~ -85 km/s.  These helpers make the conversion
explicit and central; callers that compare to vacuum rest lines (all
``udong.science.spectroscopy`` line catalogues are vacuum) should run
``spectrum.to_vacuum()`` first.

Formula: air-to-vacuum index correction (Morton 2000, ApJS 130, 403),
wavelength in Angstrom,

    n = 1 + 2.735182e-4 + 131.4182 / lam^2 + 2.76249e8 / lam^4

with ``vacuum = air * n``.  The inverse is applied by fixed-point iteration.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.units import Quantity

__all__ = ["air_to_vacuum", "vacuum_to_air"]


def _index(lam_aa: np.ndarray) -> np.ndarray:
    lam2 = lam_aa * lam_aa
    return 1.0 + 2.735182e-4 + 131.4182 / lam2 + 2.76249e8 / (lam2 * lam2)


def air_to_vacuum(wavelength: Quantity) -> Quantity:
    """Convert an **air** wavelength axis to **vacuum** (same unit)."""
    lam = Quantity(wavelength)
    unit = lam.unit
    lam_aa = lam.to_value(u.AA)
    return Quantity(lam_aa * _index(lam_aa), u.AA).to(unit)


def vacuum_to_air(wavelength: Quantity) -> Quantity:
    """Convert a **vacuum** wavelength axis to **air** (same unit).

    The refractive-index correction is inverted with two fixed-point
    iterations (error < 1e-4 A over the optical range).
    """
    lam = Quantity(wavelength)
    unit = lam.unit
    lam_aa = lam.to_value(u.AA)
    x = lam_aa  # guess: vacuum ~ air
    for _ in range(3):
        x = lam_aa / _index(x)
    return Quantity(x, u.AA).to(unit)
