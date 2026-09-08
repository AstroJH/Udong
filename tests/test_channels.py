import pytest

from udong.data.manga.channels import (
    EMISSION_LINES,
    SPECTRAL_INDICES,
    emission_line_index,
    resolve_emission_line,
    spectral_index_unit,
)


def test_emission_line_length():
    assert len(EMISSION_LINES) == 35
    assert EMISSION_LINES.index("Ha-6564") == 23
    assert EMISSION_LINES.index("OIII-5008") == 16
    assert EMISSION_LINES.index("NII-6585") == 24


def test_spectral_indices():
    assert len(SPECTRAL_INDICES) == 46
    assert dict(SPECTRAL_INDICES)["Dn4000"] is None
    assert dict(SPECTRAL_INDICES)["Mgb"] == "ang"


def test_resolve_aliases():
    assert resolve_emission_line("Ha") == "Ha-6564"
    assert resolve_emission_line("Halpha") == "Ha-6564"
    assert resolve_emission_line("h_beta") == "Hb-4862"
    assert resolve_emission_line("[O III]5007") == "OIII-5008"
    assert resolve_emission_line("[NII]6584") == "NII-6585"
    assert resolve_emission_line("Ha-6564") == "Ha-6564"
    assert emission_line_index("Ha") == 23


def test_unknown_line_raises():
    with pytest.raises(KeyError):
        resolve_emission_line("CIV-1549")


def test_spectral_index_unit():
    assert spectral_index_unit("Dn4000") is None
    assert spectral_index_unit("Mgb") == "ang"
    with pytest.raises(KeyError):
        spectral_index_unit("bogus")
