"""Tests for the (fetched, never-vendored) MaNGA mask-bit registry.

The authoritative ``sdssMaskbits.par`` is maintained by SDSS and downloaded
into a local cache; these tests exercise the parser / loader with a small
inline *sample* of the file format (not the official file).
"""

import numpy as np
import pytest

from udong.core.mask import MaskDefs
from udong.data.manga.masks import (
    MASK_GROUPS,
    default_maskbits_file,
    ensure_maskbits,
    fetch_maskbits,
    get_mask_defs,
    load_mask_registries,
    mask_defs_from_par,
    mask_defs_or_none,
)

SAMPLE_PAR = """\
masktype MANGA_DRP2PIXMASK 32 "DRP 2-D pixel mask"
maskbits MANGA_DRP2PIXMASK 0 DONOTUSE "Do not use"
masktype MANGA_DRP3PIXMASK 32 "DRP 3-D pixel mask"
maskbits MANGA_DRP3PIXMASK 0 NOCOV "No coverage"
maskbits MANGA_DRP3PIXMASK 3 FORESTAR "Foreground star"
maskbits MANGA_DRP3PIXMASK 10 DONOTUSE "Do not use in analysis"
masktype MANGA_DAPPIXMASK 64 "DAP pixel mask"
maskbits MANGA_DAPPIXMASK 5 UNRELIABLE "Unreliable measurement"
maskbits MANGA_DAPPIXMASK 7 FITFAILED "Fit failed"
maskbits MANGA_DAPPIXMASK 10 MULTICOMP "Multiple kinematic components"
maskbits MANGA_DAPPIXMASK 30 DONOTUSE "Do not use"
masktype MANGA_DAPQUAL 64 "DAP quality bitmask"
maskbits MANGA_DAPQUAL 0 WARN "Warning"
masktype MANGA_DAPSPECMASK 64 "DAP spectral mask"
maskbits MANGA_DAPSPECMASK 0 IGNORED "Ignored"
masktype MANGA_DRP3QUAL 64 "DRP quality bitmask"
maskbits MANGA_DRP3QUAL 14 UNUSUAL "Unusual data"
maskbits MANGA_DRP3QUAL 30 CRITICAL "Critical failure"
masktype MANGA_TARGET1 64 "Targeting flag 1"
maskbits MANGA_TARGET1 0 NONE "No flag"
masktype MANGA_TARGET2 64 "Targeting flag 2"
maskbits MANGA_TARGET2 0 NONE "No flag"
masktype MANGA_TARGET3 64 "Targeting flag 3"
maskbits MANGA_TARGET3 0 NONE "No flag"
"""


def test_parser_reads_groups_and_bits():
    defs = mask_defs_from_par(SAMPLE_PAR)
    assert set(defs) == set(MASK_GROUPS)
    assert defs["MANGA_DRP3PIXMASK"].value("DONOTUSE") == 1 << 10
    assert defs["MANGA_DRP3PIXMASK"].value("NOCOV") == 1 << 0
    assert defs["MANGA_DAPPIXMASK"].value("DONOTUSE") == 1 << 30
    assert defs["MANGA_DAPPIXMASK"].value("UNRELIABLE") == 1 << 5
    assert defs["MANGA_DRP3QUAL"].value("UNUSUAL") == 1 << 14
    assert defs["MANGA_DRP3QUAL"].value("CRITICAL") == 1 << 30
    # all parsed registries are real MaskDefs objects with provenance source
    assert all(isinstance(md, MaskDefs) and md.source for md in defs.values())


def test_default_maskbits_file(monkeypatch):
    monkeypatch.setenv("UDONG_MASKPAR", "/tmp/custom/sdssMaskbits.par")
    assert default_maskbits_file() == __import__("pathlib").Path("/tmp/custom/sdssMaskbits.par")

    monkeypatch.delenv("UDONG_MASKPAR", raising=False)
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: __import__("pathlib").Path("/home/me"))
    assert default_maskbits_file() == Path("/home/me/Repository/udong_data/manga/sdssMaskbits.par")


def test_ensure_missing_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="sdssMaskbits.par"):
        ensure_maskbits(local=tmp_path / "sdssMaskbits.par", download=False)


def test_fetch_and_load_registries(tmp_path, monkeypatch):
    dest = tmp_path / "sdssMaskbits.par"

    def fake_download(url, local, **kwargs):
        assert url.startswith("https://")
        __import__("pathlib").Path(local).parent.mkdir(parents=True, exist_ok=True)
        __import__("pathlib").Path(local).write_text(SAMPLE_PAR)
        return local

    monkeypatch.setattr("udong.data.manga.masks.download_file", fake_download)
    fetched = fetch_maskbits(local=dest)
    assert fetched.exists() and fetched.read_text().startswith("masktype")
    regs = load_mask_registries(local=dest)
    assert regs["MANGA_DAPPIXMASK"].value("DONOTUSE") == 1 << 30


def test_get_mask_defs_and_missing_group(tmp_path):
    dest = tmp_path / "sdssMaskbits.par"
    dest.write_text(SAMPLE_PAR)
    md = get_mask_defs("MANGA_DRP3PIXMASK", local=dest, download=False)
    assert md.value("DONOTUSE") == 1 << 10
    with pytest.raises(KeyError):
        get_mask_defs("NOT_A_GROUP", local=dest, download=False)


def test_mask_defs_or_none_returns_none_when_absent(tmp_path):
    from udong.data.manga import masks as _masks

    _masks._warned.discard("MANGA_DRP3PIXMASK")
    with pytest.warns(UserWarning):
        assert mask_defs_or_none("MANGA_DRP3PIXMASK", local=tmp_path / "nope.par") is None
    # only warned once per process / group
    assert mask_defs_or_none("MANGA_DRP3PIXMASK", local=tmp_path / "nope.par") is None


def test_numpy_bit_consistency():
    defs = mask_defs_from_par(SAMPLE_PAR)
    arr = np.zeros(4, dtype="uint64")
    arr[1] = defs["MANGA_DAPPIXMASK"].value("DONOTUSE")
    assert defs["MANGA_DAPPIXMASK"].unmask(arr, "DONOTUSE").tolist() == [False, True, False, False]
