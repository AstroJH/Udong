"""Air/vacuum wavelength-frame helpers and Spectrum frame conversion."""

import numpy as np
import pytest
from astropy import units as u

from udong.core.spectrum import Spectrum
from udong.core.wavelength import air_to_vacuum, vacuum_to_air


def test_known_values():
    # H-alpha: air 6562.801 -> vacuum 6564.614 (Morton 2000)
    ha = air_to_vacuum(6562.801 * u.AA)
    assert np.isclose(ha.to_value(u.AA), 6564.61, atol=0.02)
    # [O III] 5007: air 5006.843 -> vacuum 5008.240
    oiii = air_to_vacuum(5006.843 * u.AA)
    assert np.isclose(oiii.to_value(u.AA), 5008.24, atol=0.02)
    # round-trip
    back = vacuum_to_air(ha)
    assert np.isclose(back.to_value(u.AA), 6562.801, atol=1e-3)
    # velocity-scale of the offset ~85 km/s at H-alpha
    dv = (ha - 6562.801 * u.AA) / (6562.801 * u.AA) * 299792.458 * (u.km / u.s)
    assert np.isclose(abs(dv.to_value(u.km / u.s)), 83.0, atol=5.0)


def _spectrum(axis, frame):
    return Spectrum(
        flux=np.ones(len(axis)) * u.Unit("1e-17 erg/(s cm2 AA)"),
        spectral_axis=axis,
        meta={"wavelength_frame": frame},
    )


def test_to_vacuum_and_to_air():
    air = np.linspace(6500.0, 6600.0, 50) * u.AA
    sp = _spectrum(air, "air")
    assert sp.wavelength_frame == "air"
    sv = sp.to_vacuum()
    assert sv is not sp
    assert sv.wavelength_frame == "vacuum"
    assert sv.meta["wavelength_frame"] == "vacuum"
    assert np.allclose(sv.spectral_axis.to_value(u.AA)[0],
                       air_to_vacuum(air).to_value(u.AA)[0])
    assert sv.provenance is None  # no provenance -> no step
    # round trip
    sa = sv.to_air()
    assert np.allclose(sa.spectral_axis.to_value(u.AA), air.to_value(u.AA), atol=1e-3)


def test_noop_when_frame_matches_or_unknown():
    vac = np.linspace(6500.0, 6600.0, 50) * u.AA
    spv = _spectrum(vac, "vacuum")
    assert spv.to_vacuum() is spv
    spu = Spectrum(flux=spv.flux, spectral_axis=vac, meta={})
    assert spu.wavelength_frame is None
    assert spu.to_vacuum() is spu


def test_provenance_step_recorded():
    from udong.core.provenance import Provenance

    prov = Provenance(product="t", source="s")
    air = np.linspace(6500.0, 6600.0, 50) * u.AA
    sp = Spectrum(flux=np.ones(50) * u.Unit("1e-17 erg/(s cm2 AA)"),
                  spectral_axis=air, meta={"wavelength_frame": "air"},
                  provenance=prov)
    sv = sp.to_vacuum()
    assert len(sv.provenance.steps) == 1
    assert sv.provenance.steps[0].name == "air_to_vacuum"
    # original provenance untouched (immutable chain)
    assert len(sp.provenance.steps) == 0
