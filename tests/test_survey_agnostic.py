"""Survey-agnostic chain: identical science calls on MaNGA- and MUSE-like data.

Also locks in the air/vacuum frame contract: MaNGA readers tag spectra
``"vacuum"``, MUSE ``"air"``; vacuum-rest comparisons (the emission-line
catalogues) must run on vacuum wavelengths, so MUSE spectra get
``.to_vacuum()`` first -- otherwise every velocity is biased by ~ -85 km/s.
"""

import numpy as np
from astropy import units as u

from udong.core.spectrum import Spectrum
from udong.core.wavelength import vacuum_to_air
from udong.science.spectroscopy.moments import line_moments

REST_VAC = 5008.24 * u.AA          # [O III] 5007 vacuum rest
REST_AIR = vacuum_to_air(REST_VAC)  # ~ 5006.84 A
FLUX_UNIT = u.Unit("1e-20 erg/(s cm2 AA)")


def _gaussian(lam, peak, sigma=0.8, amp=1.0):
    return amp * np.exp(-0.5 * ((lam.to_value(u.AA) - peak) / sigma) ** 2)


def test_vacuum_and_air_moments_agree_after_conversion():
    z = 0.0619
    # vacuum-frame spectrum: physical line at vacuum rest * (1+z)
    peak_v = REST_VAC.to_value(u.AA) * (1 + z)
    wave_v = np.linspace(peak_v - 8.0, peak_v + 8.0, 401) * u.AA
    flux_v = _gaussian(wave_v, peak_v) * FLUX_UNIT
    _, m1_v, _ = line_moments(wave_v, flux_v, REST_VAC * (1 + z))
    assert abs(m1_v.to_value(u.km / u.s)) < 1.0

    # air-frame spectrum: the *same* physical line sits ~0.028% bluer in air
    peak_a = REST_AIR.to_value(u.AA) * (1 + z)
    wave_a = np.linspace(peak_a - 8.0, peak_a + 8.0, 401) * u.AA
    flux_a = _gaussian(wave_a, peak_a) * FLUX_UNIT

    # naive vacuum-rest comparison on the air spectrum -> ~ -85 km/s bias
    _, m1_bias, _ = line_moments(wave_a, flux_a, REST_VAC * (1 + z))
    assert abs(m1_bias.to_value(u.km / u.s) - (-83.0)) < 12.0

    # after Spectrum.to_vacuum() the bias vanishes
    spec_a = Spectrum(flux=flux_a, spectral_axis=wave_a, meta={"wavelength_frame": "air"})
    spec_v = spec_a.to_vacuum()
    _, m1_fixed, _ = line_moments(spec_v.spectral_axis, spec_v.flux, REST_VAC * (1 + z))
    assert abs(m1_fixed.to_value(u.km / u.s)) < 2.0


def test_readers_tag_frame_and_to_vacuum_noop(fake_cube_file, fake_muse_file):
    from udong.data.manga.reader import read_cube as read_manga
    from udong.data.muse.reader import read_cube as read_muse

    sm = read_manga(fake_cube_file).spectrum(1, 1)
    assert sm.wavelength_frame == "vacuum"
    assert sm.to_vacuum() is sm

    su = read_muse(fake_muse_file, dp_id="ADP.2024-04-30T18:20:44.624").spectrum(1, 1)
    assert su.wavelength_frame == "air"
    sv = su.to_vacuum()
    assert sv is not su and sv.wavelength_frame == "vacuum"
    assert sv.provenance is not None and sv.provenance.steps[-1].name == "air_to_vacuum"
