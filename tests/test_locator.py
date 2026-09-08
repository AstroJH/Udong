import pytest

from udong.data.manga.locator import MangaPath, parse_plateifu


def test_parse_plateifu():
    assert parse_plateifu("8485-1901") == ("8485", "1901")
    assert parse_plateifu("7443-12703") == ("7443", "12703")
    with pytest.raises(ValueError):
        parse_plateifu("nonsense")


def test_paths(tmp_path):
    mp = MangaPath(tmp_path, release="dr17", drpver="v3_1_1", dapver="3.1.0")
    assert mp.cube_local("8485-1901") == (
        tmp_path / "dr17/manga/spectro/redux/v3_1_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz"
    )
    assert mp.maps_local("8485-1901", "HYB10-MILESHC-MASTARSSP") == (
        tmp_path / "dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/8485/1901/"
        "manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
    )
    assert mp.drpall_local() == tmp_path / "dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits"
    assert mp.cube_url("8485-1901") == (
        "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8485/stack/"
        "manga-8485-1901-LOGCUBE.fits.gz"
    )
