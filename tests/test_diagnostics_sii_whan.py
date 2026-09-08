import numpy as np
from astropy import units as u

from udong.core.map import Map2D
from udong.science.diagnostics.sii_bpt import (
    LINER,
    SEYFERT,
    STAR_FORMING,
    classify_sii,
    kewley_2001_sii,
    kewley_2006_sl,
    sii_bpt_classification,
    total_sii_flux,
)
from udong.science.diagnostics.whan import (
    AGN_STRONG,
    AGN_WEAK,
    RETIRED,
    classify_whan,
    whan_classification,
)

_FLUX = u.Unit("1e-17 erg/(s cm2)")


def flux_map(fill, shape=(8, 8)):
    return Map2D(value=np.full(shape, fill) * _FLUX,
                 uncertainty=np.full(shape, 1e4),
                 mask=np.zeros(shape, dtype=bool))


def test_sii_boundary_functions():
    assert np.isclose(kewley_2001_sii(-0.5), 0.72 / (-0.82) + 1.30)
    assert np.isclose(kewley_2006_sl(-0.5), 1.89 * (-0.5) + 0.76)


def test_classify_sii_points():
    x = np.array([-0.8, -0.5, np.nan])
    y = np.array([-0.5, 0.5, 0.0])
    c = classify_sii(x, y)
    # (-0.8,-0.5): kewley01(-0.8) = 0.66 -> below -> SF
    assert c[0] == STAR_FORMING
    # (-0.5,0.5): above kewley01(-0.5)=0.42 and kewley06(-0.5)=-0.185 -> Seyfert
    assert c[1] == SEYFERT
    assert c[2] == 0  # NaN -> unclassified


def test_classify_sii_liner():
    x = np.array([-0.3])
    # kewley01(-0.3)=0.139 (starburst); kewley06(-0.3)=0.193 (Seyfert/LINER)
    y_sey = np.array([0.4])  # above both -> Seyfert
    assert classify_sii(x, y_sey)[0] == SEYFERT
    y_lin = np.array([0.16])  # 0.139 < 0.16 < 0.193 -> LINER
    assert classify_sii(x, y_lin)[0] == LINER


def test_total_sii_flux():
    s1 = flux_map(30.0)
    s2 = flux_map(20.0)
    tot = total_sii_flux(s1, s2)
    assert np.isclose(tot.value.value[0, 0], 50.0)
    # independent sum: var adds -> ivar_sum = 1/(2 * 1e-4) = 5000
    assert np.isclose(tot.ivar.value[0, 0], 5000.0)


def test_sii_bpt_pipeline():
    # SF spaxels on the left half (low [OIII]), Seyfert on the right half
    ny, nx = 4, 8
    sii_m = Map2D(value=np.full((ny, nx), 50.0) * _FLUX,
                  uncertainty=np.full((ny, nx), 1e4),
                  mask=np.zeros((ny, nx), dtype=bool))
    ha_m = Map2D(value=np.full((ny, nx), 100.0) * _FLUX,
                 uncertainty=np.full((ny, nx), 1e4),
                 mask=np.zeros((ny, nx), dtype=bool))
    hb_m = Map2D(value=np.full((ny, nx), 30.0) * _FLUX,
                 uncertainty=np.full((ny, nx), 1e4),
                 mask=np.zeros((ny, nx), dtype=bool))
    oiii = np.full((ny, nx), 6.0)
    oiii[:, nx // 2:] = 60.0  # right half high [OIII]
    oiii_m = Map2D(value=oiii * _FLUX, uncertainty=np.full((ny, nx), 1e4),
                   mask=np.zeros((ny, nx), dtype=bool))
    res = sii_bpt_classification(sii_m, ha_m, oiii_m, hb_m)
    ids = res.class_id
    assert (ids[:, : nx // 2] == STAR_FORMING).all()
    assert (ids[:, nx // 2:] == SEYFERT).all()
    fracs = res.class_fractions()
    assert set(fracs) <= {"star-forming", "seyfert", "liner"}


def test_classify_whan_points():
    # log[NII]/Ha = -0.6 -> SF (any EW)
    c = classify_whan(np.array([-0.6]), np.array([0.0]))
    assert c[0] == STAR_FORMING
    # log[NII]/Ha=0.0: retired EW<3 -> log10(1)=0 < log10(3)=0.477 -> retired
    c = classify_whan(np.array([0.0]), np.array([0.0]))
    assert c[0] == RETIRED
    # EW=4.5 (log=0.653) -> weak AGN
    c = classify_whan(np.array([0.0]), np.array([np.log10(4.5)]))
    assert c[0] == AGN_WEAK
    # EW=10 (log=1.0) -> strong AGN
    c = classify_whan(np.array([0.0]), np.array([1.0]))
    assert c[0] == AGN_STRONG


def test_whan_pipeline_restframe():
    ny, nx = 4, 4
    nii = Map2D(value=np.full((ny, nx), 100.0) * _FLUX,
                uncertainty=np.full((ny, nx), 1e4), mask=np.zeros((ny, nx), dtype=bool))
    ha = Map2D(value=np.full((ny, nx), 100.0) * _FLUX,
               uncertainty=np.full((ny, nx), 1e4), mask=np.zeros((ny, nx), dtype=bool))
    # log[NII]/Ha = 0.0
    ew = Map2D(value=np.full((ny, nx), 4.0) * u.AA,
               uncertainty=np.full((ny, nx), 100.0) / u.AA**2,
               mask=np.zeros((ny, nx), dtype=bool))
    res_z0 = whan_classification(nii, ha, ew, z=0.0)
    assert (res_z0.class_id == AGN_WEAK).all()
    # z=1 -> rest EW = 2 A -> retired
    res_z1 = whan_classification(nii, ha, ew, z=1.0)
    assert (res_z1.class_id == RETIRED).all()
    assert res_z1.redshift == 1.0
