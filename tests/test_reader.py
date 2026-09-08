import pytest
from astropy import units as u

from udong.data.manga.reader import manga_unit, read_cube, read_drpall, read_maps


def test_manga_unit():
    assert manga_unit("km/s") == u.km / u.s
    assert manga_unit("1E-17 erg/s/cm^2/ang/spaxel") == u.Unit("1e-17 erg/(s cm2 AA)")
    assert manga_unit("1E-17 erg/s/cm^2/ang/spaxel").is_equivalent(u.erg / (u.s * u.cm**2 * u.AA))
    assert manga_unit("(km/s)^{-2}") == (u.km / u.s) ** -2
    assert manga_unit("arcsec") == u.arcsec
    assert manga_unit("") == u.dimensionless_unscaled
    with pytest.raises(ValueError):
        manga_unit("bogus unit")


def test_read_cube(fake_cube_file):
    cube = read_cube(fake_cube_file)
    assert cube.shape == (50, 8, 8)
    assert cube.nwave == 50
    assert cube.plateifu == "8485-1901"
    assert cube.mangaid == "1-209232"
    assert cube.wavelength is not None and len(cube.wavelength) == 50
    assert cube.flux.unit == u.Unit("1e-17 erg/(s cm2 AA)")
    assert cube.wcs is not None and cube.wcs.pixel_n_dim == 3
    assert cube.lsf_pre is not None
    # DONOTUSE at (0,0,0)
    assert cube.mask[0, 0, 0] & (1 << 10)
    spec = cube.spectrum(0, 0)
    assert len(spec) == 50
    assert spec.bad[0]


def test_read_maps(fake_maps_file):
    maps = read_maps(fake_maps_file)
    assert maps.plateifu == "8485-1901"
    assert maps.daptype == "HYB10-MILESHC-MASTARSSP"
    assert "STELLAR_VEL" in maps
    vel = maps["STELLAR_VEL"]
    assert vel.value.unit == u.km / u.s
    assert vel.shape == (8, 8)
    assert vel.bad[0, 0]
    # emission lines via alias and canonical name
    ha = maps.emission_line_flux("Ha")
    assert ha.shape == (8, 8)
    assert ha.channel == "Ha-6564"
    assert ha.value.unit == u.Unit("1e-17 erg/(s cm2)")
    assert ha.value.unit.is_equivalent(u.erg / (u.s * u.cm**2))
    oiii = maps.emission_line_flux("OIII-5008")
    assert oiii.bad[2, 3]
    assert maps.emission_lines == ["Hb-4862", "OIII-5008", "Ha-6564"]
    # elliptical coordinates channel
    ell = maps.channel("SPX_ELLCOO", "R/Re")
    assert ell.shape == (8, 8)
    assert ell.value.unit == u.dimensionless_unscaled
    with pytest.raises(KeyError):
        maps.emission_line_flux("CIV-1549")


def test_read_drpall(fake_drpall_file):
    t = read_drpall(fake_drpall_file)
    assert len(t) == 2
    assert "PLATEIFU" in t.colnames
    assert t["PLATEIFU"][0] == "8485-1901"


def test_read_cube_truncated_gz_raises_clear_error(fake_cube_file, tmp_path):
    """A truncated .fits.gz must raise a descriptive IOError, not KeyError."""
    import gzip

    dest = tmp_path / "truncated-LOGCUBE.fits.gz"
    with open(fake_cube_file, "rb") as f:
        raw = gzip.compress(f.read())
    dest.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(IOError, match="truncated|corrupt"):
        read_cube(dest)
