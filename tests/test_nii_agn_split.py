import numpy as np

from udong.science.diagnostics.bpt import (
    AGN,
    COMPOSITE,
    LINER,
    SEYFERT,
    STAR_FORMING,
    cid_fernandes_2010_sl,
    classify_nii,
)


def test_cf10_line():
    assert np.isclose(cid_fernandes_2010_sl(0.0), 0.45)
    assert np.isclose(cid_fernandes_2010_sl(-0.2), 1.05 * (-0.2) + 0.45)


def test_classify_nii_default_unchanged():
    # x = -0.3: kauffmann03 = 0.61/(-0.35)+1.30 = -0.44 ; kewley01 = 0.61/(-0.77)+1.19 = 0.40
    x = np.array([-0.3, -0.3, -0.3])
    y = np.array([-0.6, 0.0, 1.0])
    c = classify_nii(x, y)
    assert c[0] == STAR_FORMING
    assert c[1] == COMPOSITE
    assert c[2] == AGN


def test_classify_nii_agn_split():
    # AGN wedge at x = 0.3: kewley01(0.3)=2.15 ; cf10(0.3)=0.765
    x = np.array([0.3, 0.3])
    y = np.array([1.0, 0.6])  # both above kewley01 -> AGN
    c = classify_nii(x, y, agn_split=cid_fernandes_2010_sl)
    assert c[0] == SEYFERT   # above cf10 line
    assert c[1] == LINER     # below cf10 line (but above kewley01)
