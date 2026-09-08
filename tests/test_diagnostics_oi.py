import numpy as np
from astropy import units as u

from udong.core.map import Map2D
from udong.science.diagnostics.oi_bpt import (
    LINER,
    SEYFERT,
    STAR_FORMING,
    classify_oi,
    kewley_2001_oi,
    kewley_2006_sl,
    oi_bpt_classification,
)

_FLUX = u.Unit("1e-17 erg/(s cm2)")


def flux_map(fill, shape=(8, 8)):
    return Map2D(value=np.full(shape, fill) * _FLUX,
                 uncertainty=np.full(shape, 1e4),
                 mask=np.zeros(shape, dtype=bool))


def test_oi_boundary_functions():
    # kewley01: y = 0.73/(x + 0.59) + 1.33
    assert np.isclose(kewley_2001_oi(-0.5), 0.73 / 0.09 + 1.33)
    # kewley06 Seyfert/LINER: y = 1.18 x + 1.30
    assert np.isclose(kewley_2006_sl(-0.5), 1.18 * (-0.5) + 1.30)


def test_classify_oi_points():
    # x = -0.8: kewley01(-0.8) = 0.73/(-0.21)+1.33 = -2.146
    #           kewley06(-0.8) = 1.18*(-0.8)+1.30 = 0.356
    x = np.array([-0.8, -0.8, -0.8, np.nan])
    y = np.array([-2.5, 0.0, 0.5, 0.0])
    c = classify_oi(x, y)
    assert c[0] == STAR_FORMING  # below kewley01
    assert c[1] == LINER         # above kewley01, below kewley06
    assert c[2] == SEYFERT       # above kewley06
    assert c[3] == 0             # NaN -> unclassified


def test_oi_bpt_pipeline():
    ny, nx = 4, 8
    ha_m = flux_map(100.0, (ny, nx))
    hb_m = flux_map(30.0, (ny, nx))
    oi_m = Map2D(value=np.full((ny, nx), 5.0) * _FLUX,
                 uncertainty=np.full((ny, nx), 1e4),
                 mask=np.zeros((ny, nx), dtype=bool))
    # log10(OI/Ha) = log10(0.05) = -1.301
    #   kewley01(-1.301) = 0.303 ; kewley06(-1.301) = -0.235
    # left half  log10(OIII/Hb) = log10(6/30) = -0.699 < 0.303 -> SF
    # right half log10(OIII/Hb) = log10(200/30) = 0.824 > 0.303 -> Seyfert
    oiii = np.full((ny, nx), 6.0)
    oiii[:, nx // 2:] = 200.0
    oiii_m = Map2D(value=oiii * _FLUX, uncertainty=np.full((ny, nx), 1e4),
                   mask=np.zeros((ny, nx), dtype=bool))
    res = oi_bpt_classification(oi_m, ha_m, oiii_m, hb_m)
    ids = res.class_id
    assert (ids[:, : nx // 2] == STAR_FORMING).all()
    assert (ids[:, nx // 2:] == SEYFERT).all()
    fracs = res.class_fractions()
    assert set(fracs) <= {"star-forming", "seyfert", "liner"}
