import numpy as np
from astropy import units as u

from udong.core.map import Map2D
from udong.science.diagnostics.bpt import (
    AGN,
    COMPOSITE,
    STAR_FORMING,
    UNCLASSIFIED,
    classify_nii,
    kauffmann_2003_nii,
    kewley_2001_nii,
    log10_flux_ratio,
    nii_bpt_classification,
)

# MaNGA-like flux maps: values in 1e-17 erg/s/cm2, IVAR as plain ndarray in
# (1e-17 erg/s/cm2)^-2.
_FLUX_UNIT = u.Unit("1e-17 erg/(s cm2)")


def make_flux_map(shape=(8, 8), fill=10.0, ivar=1e4):
    return Map2D(
        value=np.full(shape, fill) * _FLUX_UNIT,
        uncertainty=np.full(shape, ivar),
        mask=np.zeros(shape, dtype=bool),
    )


def test_classify_known_points():
    # boundaries: kauffmann(-0.5)=0.191 ; kauffmann(-0.3)=-0.443, kewley(-0.3)=0.398
    x = np.array([-0.5, -0.3, -0.2, np.nan])
    y = np.array([-0.5, 0.0, 1.0, -0.5])
    c = classify_nii(x, y)
    assert c[0] == STAR_FORMING  # below Kauffmann
    assert c[1] == COMPOSITE  # between Kauffmann and Kewley
    assert c[2] == AGN  # above Kewley
    assert c[3] == UNCLASSIFIED


def test_classify_no_composite():
    x = np.array([-0.3])
    y = np.array([0.0])
    c = classify_nii(x, y, allow_composite=False)
    assert c[0] == STAR_FORMING  # 0.0 < kewley(-0.3)=0.398


def test_boundary_functions():
    assert np.isclose(kauffmann_2003_nii(-0.5), 0.61 / (-0.55) + 1.30)
    assert np.isclose(kewley_2001_nii(-0.5), 0.61 / (-0.97) + 1.19)


def test_log10_flux_ratio_propagation():
    num = make_flux_map(fill=10.0)
    den = make_flux_map(fill=100.0)
    logr, sig = log10_flux_ratio(num, den)
    assert np.isclose(logr.value.value[0, 0], -1.0)
    rel2 = 1.0 / (1e4 * 10.0**2) + 1.0 / (1e4 * 100.0**2)
    expected_sigma_log = np.sqrt(rel2) / np.log(10.0)
    assert np.isclose(sig.value.value[0, 0], expected_sigma_log)
    assert logr.unit == u.dimensionless_unscaled


def test_full_bpt_pipeline():
    # left-to-right: log[NII]/Ha from -0.9 (SF) to -0.2 (AGN territory);
    # log[OIII]/Hb from -0.2 to +1.2.
    ny, nx = 6, 12
    ha = make_flux_map(shape=(ny, nx), fill=100.0)
    hb = make_flux_map(shape=(ny, nx), fill=30.0)
    xj = np.linspace(-0.9, -0.2, nx)  # log10([NII]/Ha)
    yj = np.linspace(-0.2, 1.2, nx)  # log10([OIII]/Hb)
    nii_val = 100.0 * 10 ** xj
    oiii_val = 30.0 * 10 ** yj
    nii = make_flux_map(shape=(ny, nx))
    oiii = make_flux_map(shape=(ny, nx))
    nii = Map2D(
        value=np.broadcast_to(nii_val, (ny, nx)) * _FLUX_UNIT,
        uncertainty=np.full((ny, nx), 1e4),
        mask=np.zeros((ny, nx), dtype=bool),
    )
    oiii = Map2D(
        value=np.broadcast_to(oiii_val, (ny, nx)) * _FLUX_UNIT,
        uncertainty=np.full((ny, nx), 1e4),
        mask=np.zeros((ny, nx), dtype=bool),
    )
    res = nii_bpt_classification(oiii, hb, nii, ha)
    assert res.class_map.shape == (ny, nx)
    ids = res.class_id
    # leftmost: logNII=-0.9, logOIII=-0.2 -> SF (kauffmann(-0.9)=0.66)
    assert (ids[:, 0] == STAR_FORMING).all()
    # rightmost: logNII=-0.2, logOIII=1.2 -> AGN (kewley(-0.2)=0.28)
    assert (ids[:, -1] == AGN).all()
    assert res.class_map.value.unit == u.dimensionless_unscaled
    fracs = res.class_fractions()
    assert set(fracs) <= {"star-forming", "composite", "AGN"}
