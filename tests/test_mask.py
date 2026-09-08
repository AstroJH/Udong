import numpy as np
import pytest

from udong.core.mask import Bit, MaskDefs


def test_maskdefs_basic():
    m = MaskDefs("TEST", [Bit("A", 0, "bit a"), Bit("B", 3, "bit b")], dtype="uint32")
    assert m.name == "TEST"
    assert m.value("A") == 1
    assert m.value("B") == 8
    assert "A" in m
    assert "C" not in m
    assert set(m.names()) == {"A", "B"}


def test_unmask_any_all():
    m = MaskDefs("TEST", [Bit("A", 0), Bit("B", 1)], dtype="uint32")
    masks = np.array([0, 1, 2, 3], dtype=np.uint32)
    assert m.unmask(masks, "A").tolist() == [False, True, False, True]
    assert m.unmask(masks, "B").tolist() == [False, False, True, True]
    assert m.unmask(masks).tolist() == [False, True, True, True]
    assert m.unmask(masks, "A", "B").tolist() == [False, True, True, True]
    assert m.unmask(masks, "A", "B", keep_all=True).tolist() == [False, False, False, True]
    assert m.good(masks, "A").tolist() == [True, False, True, False]


def test_unmask_unknown_bit_raises():
    m = MaskDefs("TEST", [Bit("A", 0)])
    with pytest.raises(KeyError):
        m.unmask(np.array([1]), "NOPE")


def test_serialization_roundtrip():
    m = MaskDefs("TEST", [Bit("A", 0, "desc")], dtype="uint64", source="src")
    m2 = MaskDefs.from_dict(m.to_dict())
    assert m2.name == m.name
    assert m2.value("A") == 1
    assert m2.source == "src"
    assert m2.dtype == np.dtype("uint64")
