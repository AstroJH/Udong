from astropy.io import fits

from udong.core.provenance import Provenance


def test_roundtrip_dict_yaml():
    p = Provenance(
        product="manga-cube",
        source="/data/cube.fits.gz",
        survey="MaNGA",
        release="dr17",
        checksums={"datasum": "123"},
        versions={"astropy": "8.0"},
    ).add_step("mask", {"bits": ["DONOTUSE"]}, version="0.1.0")
    d = p.to_dict()
    p2 = Provenance.from_dict(d)
    assert p2.source == p.source
    assert p2.steps[0].name == "mask"
    p3 = Provenance.from_yaml(p.to_yaml())
    assert p3.versions == p.versions


def test_write_fits_header():
    p = Provenance(product="manga-maps", source="x.fits", survey="MaNGA",
                   checksums={"datasum": "42"}).add_step("read")
    hdr = fits.Header()
    p.write_fits_header(hdr)
    assert hdr["UDONGPROD"] == "manga-maps"
    assert hdr["UDONGSUM"] == "42"
    assert hdr["UDONGNSTEP"] == 1
