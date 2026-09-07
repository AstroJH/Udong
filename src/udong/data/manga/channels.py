"""Channel maps for MaNGA DAP MAPS extensions.

The channel names and order were validated against a real DR17 MAPS file
(manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz) on 2026-08-30.
"""

from __future__ import annotations

__all__ = [
    "EMISSION_LINES",
    "SPECTRAL_INDICES",
    "resolve_emission_line",
    "emission_line_index",
    "spectral_index_unit",
]

# 35 channels, DR17/MPL-11 ordering
EMISSION_LINES: list[str] = [
    "OII-3727", "OII-3729", "H12-3751", "H11-3771", "Hthe-3798", "Heta-3836",
    "NeIII-3869", "HeI-3889", "Hzet-3890", "NeIII-3968", "Heps-3971",
    "Hdel-4102", "Hgam-4341", "HeII-4687", "Hb-4862", "OIII-4960",
    "OIII-5008", "NI-5199", "NI-5201", "HeI-5877", "OI-6302", "OI-6365",
    "NII-6549", "Ha-6564", "NII-6585", "SII-6718", "SII-6732", "HeI-7067",
    "ArIII-7137", "ArIII-7753", "Peta-9017", "SIII-9071", "Pzet-9231",
    "SIII-9533", "Peps-9548",
]

# 46 spectral indices, DR17/MPL-11 ordering; unit per index
SPECTRAL_INDICES: list[tuple[str, str | None]] = [
    ("CN1", "mag"), ("CN2", "mag"), ("Ca4227", "ang"), ("G4300", "ang"),
    ("Fe4383", "ang"), ("Ca4455", "ang"), ("Fe4531", "ang"), ("C24668", "ang"),
    ("Hb", "ang"), ("Fe5015", "ang"), ("Mg1", "mag"), ("Mg2", "mag"),
    ("Mgb", "ang"), ("Fe5270", "ang"), ("Fe5335", "ang"), ("Fe5406", "ang"),
    ("Fe5709", "ang"), ("Fe5782", "ang"), ("NaD", "ang"), ("TiO1", "mag"),
    ("TiO2", "mag"), ("HDeltaA", "ang"), ("HGammaA", "ang"), ("HDeltaF", "ang"),
    ("HGammaF", "ang"), ("CaHK", "ang"), ("CaII1", "ang"), ("CaII2", "ang"),
    ("CaII3", "ang"), ("Pa17", "ang"), ("Pa14", "ang"), ("Pa12", "ang"),
    ("MgICvD", "ang"), ("NaICvD", "ang"), ("MgIIR", "ang"), ("FeHCvD", "ang"),
    ("NaI", "ang"), ("bTiO", "mag"), ("aTiO", "mag"), ("CaH1", "mag"),
    ("CaH2", "mag"), ("NaISDSS", "ang"), ("TiO2SDSS", "mag"), ("D4000", None),
    ("Dn4000", None), ("TiOCvD", "ang"),
]

# Common aliases -> canonical DAP channel names (MaNGA-specific naming).
_ALIASES: dict[str, str] = {
    "ha": "Ha-6564", "halpha": "Ha-6564", "h_alpha": "Ha-6564",
    "hb": "Hb-4862", "hbeta": "Hb-4862", "h_beta": "Hb-4862",
    "oiii": "OIII-5008", "oiii5007": "OIII-5008", "oiii5008": "OIII-5008",
    "[oiii]5007": "OIII-5008", "oiii4960": "OIII-4960",
    "nii": "NII-6585", "nii6584": "NII-6585", "[nii]6584": "NII-6585",
    "nii6549": "NII-6549", "[nii]6549": "NII-6549",
    "sii6716": "SII-6718", "[sii]6716": "SII-6718", "sii6731": "SII-6732",
    "[sii]6731": "SII-6732", "sii": "SII-6718",
    "oii3727": "OII-3727", "oii3729": "OII-3729",
    "oii": "OII-3727", "neiii3869": "NeIII-3869", "heii4687": "HeII-4687",
    "oi": "OI-6302", "oi6300": "OI-6302", "[oi]6300": "OI-6302",
    "oi6364": "OI-6365", "[oi]6364": "OI-6365",
    "dn4000": "Dn4000", "d4000": "D4000",
}

_LINE_INDEX = {name: i for i, name in enumerate(EMISSION_LINES)}
_INDEX_UNITS = dict(SPECTRAL_INDICES)


def resolve_emission_line(name: str) -> str:
    """Resolve an alias to a canonical DAP emission-line channel name."""
    key = name.strip().lower().replace(" ", "")
    if key in _ALIASES:
        return _ALIASES[key]
    if name in _LINE_INDEX:
        return name
    raise KeyError(
        f"unknown emission line {name!r}; valid channels: {EMISSION_LINES}"
    )


def emission_line_index(name: str) -> int:
    """0-based channel index for an emission-line name/alias."""
    return _LINE_INDEX[resolve_emission_line(name)]


def spectral_index_unit(name: str) -> str | None:
    """Unit (string) of a spectral index, or None."""
    if name in _INDEX_UNITS:
        return _INDEX_UNITS[name]
    raise KeyError(f"unknown spectral index {name!r}")
