import numpy as np
import pytest
from astropy import units as u

from udong.core.mask import Bit, MaskDefs
from udong.core.provenance import Provenance
from udong.core.spectrum import Spectrum


def make_spectrum():
    wave = np.arange(4000, 5001, 10) * u.AA
    flux = np.ones(101) * 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    ivar = np.full(101, 1e34) / (u.erg / (u.s * u.cm**2 * u.AA)) ** 2
    return Spectrum(flux=flux, spectral_axis=wave, uncertainty=ivar)


def test_basic_properties():
    s = make_spectrum()
    assert len(s) == 101
    assert s.flux.unit == u.erg / (u.s * u.cm**2 * u.AA)
    assert s.ivar is not None
    assert np.allclose(s.sigma.value, 1e-17)
    assert s.mask is None
    assert s.bad is None


def test_mask_and_bad():
    s = make_spectrum()
    mdefs = MaskDefs("TEST", [Bit("DONOTUSE", 10)], dtype="uint64")
    raw = np.zeros(101, dtype=np.uint64)
    raw[5] = 1 << 10
    s2 = Spectrum(flux=s.flux, spectral_axis=s.spectral_axis, uncertainty=s.ivar,
                  mask=raw, mask_defs=mdefs)
    assert s2.bad[5]
    assert not s2.bad[6]
    s3 = s2.with_mask(np.zeros(101, dtype=bool))
    assert s3.bad[5]


def test_slice():
    s = make_spectrum()
    sub = s.slice(4200 * u.AA, 4300 * u.AA)
    assert len(sub) == 11  # 4200, 4210, ..., 4300
    assert sub.flux.unit == s.flux.unit
    assert sub.spectral_axis[0].value == pytest.approx(4200)


def test_provenance_attached():
    s = make_spectrum()
    prov = Provenance(product="test", source="x.fits")
    s2 = Spectrum(flux=s.flux, spectral_axis=s.spectral_axis, provenance=prov)
    assert s2.provenance.source == "x.fits"
