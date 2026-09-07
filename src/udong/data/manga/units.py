"""MaNGA unit-string parsing (explicit mapping).

MaNGA uses a small fixed set of BUNIT strings.
An explicit mapping is more robust than trying to make astropy parse
survey-specific pseudo-units such as "spaxel" or "ang".
"""

from __future__ import annotations

from astropy import units as u

__all__ = ["manga_unit"]

# ------------------ units ------------------ #
_FLUX_UNIT = u.Unit("1e-17 erg/(s cm2 AA)")
_FLUX_SPAXEL_UNIT = u.Unit("1e-17 erg/(s cm2)")

_MANGA_UNITS: dict[str, u.UnitBase] = {
    "1e-17erg/s/cm^2/ang/spaxel": _FLUX_UNIT,
    "1e-17erg/s/cm^2/angstrom/spaxel": _FLUX_UNIT,
    "(1e-17erg/s/cm^2/ang/spaxel)^-2": _FLUX_UNIT ** -2,
    "1e-17erg/s/cm^2/spaxel": _FLUX_SPAXEL_UNIT,
    "(1e-17erg/s/cm^2/spaxel)^-2": _FLUX_SPAXEL_UNIT ** -2,
    "km/s": u.km / u.s,
    "(km/s)^-2": (u.km / u.s) ** -2,
    "ang": u.AA,
    "(ang)^-2": u.AA**-2,
    "arcsec": u.arcsec,
    "arcsec^2": u.arcsec**2,
    "degrees": u.deg,
    "mag": u.mag,
    "kpc/h": u.kpc,  # h absorbed; the h convention must be stated per analysis
    "": u.dimensionless_unscaled,
}


def _normalize_unit_string(bunit: str | None) -> str:
    return "".join( (bunit or "").split() )\
             .lower()\
             .replace("{", "")\
             .replace("}", "")


def manga_unit(bunit: str | None) -> u.UnitBase:
    """Parse a MaNGA BUNIT string into an astropy unit (explicit mapping)."""
    key = _normalize_unit_string(bunit)
    
    try:
        return _MANGA_UNITS[key]
    except KeyError:
        raise ValueError(f"unrecognized MaNGA BUNIT string: {bunit!r}") from None


