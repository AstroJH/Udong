"""MUSE (ESO Phase 3) unit-string parsing (explicit mapping).

MUSE pipeline BUNIT strings are standardized, e.g.
``10**(-20)erg.s**(-1).cm**(-2).angstrom**(-1)`` for DATA and
``10**(-40)erg**2.s**(-2).cm**(-4).angstrom**(-2)`` for STAT (variance).
As with MaNGA, an explicit mapping is more robust than making astropy parse
survey-specific pseudo-unit strings.
"""

from __future__ import annotations

from astropy import units as u

__all__ = ["muse_unit"]

_FLUX_UNIT = u.Unit("1e-20 erg/(s cm2 AA)")
_VARIANCE_UNIT = _FLUX_UNIT ** 2

_MUSE_UNITS: dict[str, u.UnitBase] = {
    "10**(-20)erg.s**(-1).cm**(-2).angstrom**(-1)": _FLUX_UNIT,
    "1e-20erg.s**(-1).cm**(-2).angstrom**(-1)": _FLUX_UNIT,
    "10**(-40)erg**2.s**(-2).cm**(-4).angstrom**(-2)": _VARIANCE_UNIT,
    "1e-40erg**2.s**(-2).cm**(-4).angstrom**(-2)": _VARIANCE_UNIT,
    "s": u.s,
    "arcsec": u.arcsec,
    "arcsec^2": u.arcsec ** 2,
    "deg": u.deg,
    "degree": u.deg,
    "": u.dimensionless_unscaled,
}


def _normalize(bunit: str | None) -> str:
    return "".join((bunit or "").split()).lower().replace("{", "").replace("}", "")


def muse_unit(bunit: str | None) -> u.UnitBase:
    """Parse a MUSE BUNIT string into an astropy unit (explicit mapping)."""
    key = _normalize(bunit)
    try:
        return _MUSE_UNITS[key]
    except KeyError:
        raise ValueError(f"unrecognized MUSE BUNIT string: {bunit!r}") from None
