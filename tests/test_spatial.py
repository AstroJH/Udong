import numpy as np
from astropy import units as u

from udong.core.map import Map2D
from udong.science.spatial import binned_profile, radial_profile


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
