import numpy as np
from astropy import units as u

from udong.core.map import Map2D
from udong.science.spatial import (
    binned_profile,
    deprojected_radius_map,
    radial_profile,
    radius_over_re,
)


def test_radial_profile_median():
    # value = radius in arcsec -> profile should grow with radius
    yy, xx = np.indices((16, 16))
    r_pix = np.hypot(xx - 7.5, yy - 7.5)
    value = Map2D(value=(r_pix * 0.5) * u.arcsec,
                  mask=np.zeros((16, 16), dtype=bool))
    prof = radial_profile(value, center=(7.5, 7.5), bins=4, statistic="median")
    assert prof.radius.unit == u.arcsec
    assert prof.value.unit == u.arcsec
    assert len(prof.radius) == 4
    # median of radius in first bin (0..rmax/4) should be small
    assert prof.value.value[0] < prof.value.value[-1]
    assert (prof.n > 0).all()


def test_binned_profile_weighted_mean():
    r = np.ones((8, 8)) * 1.0 * u.arcsec
    rr = Map2D(value=r, mask=np.zeros((8, 8), dtype=bool))
    v = Map2D(value=np.ones((8, 8)) * 5.0 * u.km / u.s,
              uncertainty=np.full((8, 8), 100.0) / (u.km / u.s) ** 2,
              mask=np.zeros((8, 8), dtype=bool))
    prof = binned_profile(v, rr, bins=1, statistic="weighted_mean")
    assert np.isclose(prof.value.value[0], 5.0)
    assert prof.uncertainty is not None
    # 64 pixels with ivar=100 -> mean ivar=6400 -> sigma=1/sqrt(6400)=0.0125
    assert np.isclose(prof.uncertainty.value[0], 0.0125)


def test_deprojected_radius_map_geometry():
    """Thin-disk deprojection: R = sqrt(x'^2 + (y'/q)^2) with q = b/a."""
    import pytest

    shape = (21, 21)
    ref = Map2D(value=np.zeros(shape) * u.arcsec, mask=np.zeros(shape, dtype=bool))
    cx = cy = 10.0
    r = deprojected_radius_map(
        ref, center=(cx, cy), position_angle=0.0 * u.deg,
        axis_ratio=0.5, scale=1.0 * u.arcsec / u.pixel,
    )
    assert r.value.unit == u.arcsec
    i, j = int(cy), int(cx)  # y row / x column indexing
    # PA = 0 -> major axis along sky x': east offset unchanged
    assert np.isclose(r.value.value[i, j + 1], 1.0)   # x' = +1
    assert np.isclose(r.value.value[i, j - 1], 1.0)
    # minor axis (north, dy = +1 in row index) is stretched by 1/q = 2
    assert np.isclose(r.value.value[i + 1, j], 2.0)
    assert np.isclose(r.value.value[i - 1, j], 2.0)
    assert "R_cyl" in r.meta["quantity"]
    assert np.isclose(r.meta["axis_ratio_b_over_a"], 0.5)


def test_deprojected_radius_inclination_equals_axis_ratio():
    ref = Map2D(value=np.zeros((9, 9)) * u.arcsec, mask=np.zeros((9, 9), dtype=bool))
    kw = dict(center=(4.0, 4.0), position_angle=0.0 * u.deg,
              scale=1.0 * u.arcsec / u.pixel)
    r1 = deprojected_radius_map(ref, axis_ratio=0.5, **kw)          # q = 0.5
    r2 = deprojected_radius_map(ref, inclination=60.0 * u.deg, **kw)  # cos60 = 0.5
    assert np.allclose(r1.value.value, r2.value.value)
    assert np.isclose(r2.meta["inclination_deg"], 60.0)


def test_deprojected_radius_validation():
    import pytest

    ref = Map2D(value=np.zeros((5, 5)) * u.arcsec, mask=np.zeros((5, 5), dtype=bool))
    with pytest.raises(ValueError, match="exactly one"):
        deprojected_radius_map(ref, position_angle=0 * u.deg)  # none given
    with pytest.raises(ValueError, match="exactly one"):
        deprojected_radius_map(ref, position_angle=0 * u.deg,
                               inclination=30 * u.deg, axis_ratio=0.8)
    with pytest.raises(ValueError, match="inclination"):
        deprojected_radius_map(ref, position_angle=0 * u.deg, inclination=120 * u.deg)
    with pytest.raises(ValueError, match="axis_ratio"):
        deprojected_radius_map(ref, position_angle=0 * u.deg, axis_ratio=1.5)


def test_radius_over_re():
    r = deprojected_radius_map(
        Map2D(value=np.zeros((11, 11)) * u.arcsec, mask=np.zeros((11, 11), dtype=bool)),
        center=(5.0, 5.0), position_angle=0 * u.deg, axis_ratio=1.0,
        scale=1.0 * u.arcsec / u.pixel,
    )
    rre = radius_over_re(r, re=2.0 * u.arcsec)
    assert rre.value.unit == u.dimensionless_unscaled
    assert np.isclose(rre.value.value[5, 7], 1.0)   # R = 2 arcsec -> R/Re = 1
    assert "R / Re" in rre.meta["quantity"]


def test_binned_profile_with_dimensionless_radius():
    # radius map in units of R/Re (dimensionless)
    rvals = np.zeros((8, 8))
    rvals[:, :4] = 0.5   # inner half -> bin [0,1)
    rvals[:, 4:] = 1.5   # outer half -> bin [1,2)
    radius = Map2D(value=rvals * u.dimensionless_unscaled,
                   mask=np.zeros((8, 8), dtype=bool))
    value = Map2D(value=np.ones((8, 8)) * u.km / u.s,
                  mask=np.zeros((8, 8), dtype=bool))
    prof = binned_profile(value, radius, bins=np.array([0.0, 1.0, 2.0]) * u.dimensionless_unscaled)
    assert prof.radius.unit == u.dimensionless_unscaled
    assert len(prof.radius) == 2
    assert prof.n.tolist() == [32, 32]
    assert np.allclose(prof.value.value, 1.0)
