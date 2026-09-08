import numpy as np
from astropy import units as u

from udong.core.spectrum import Spectrum
from udong.science.spectroscopy import integrate_band, mask_wavelength_range, to_rest_frame


def make_spectrum():
    wave = np.arange(4000, 5001, 10) * u.AA
    flux = np.ones(101) * 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    ivar = np.full(101, 1e34) / (u.erg / (u.s * u.cm**2 * u.AA)) ** 2
    return Spectrum(flux=flux, spectral_axis=wave, uncertainty=ivar)


def test_integrate_band():
    s = make_spectrum()
    f, f_ivar = integrate_band(s, 4000 * u.AA, 4100 * u.AA)
    # 11 pixels x 10 AA x 1e-17 = 1.1e-15 erg/s/cm2
    assert np.isclose(f.to_value(u.erg / (u.s * u.cm**2)), 1.1e-15)
    assert np.isfinite(f_ivar.value)


def test_to_rest_frame():
    s = make_spectrum()
    sr = to_rest_frame(s, z=0.1)
    assert np.isclose(sr.spectral_axis[0].to_value(u.AA), 4000 / 1.1)


def test_mask_range():
    s = make_spectrum()
    s2 = mask_wavelength_range(s, 4000 * u.AA, 4050 * u.AA)
    assert s2.bad[0]
    assert not s2.bad[10]
