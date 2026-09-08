from pathlib import Path

import pytest

from udong.data.manga.dataset import MangaDataset, default_root


def _gzip_copy(src, dst):
    import gzip
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".gz":
        with open(src, "rb") as f, gzip.open(dst, "wb") as g:
            shutil.copyfileobj(f, g)
    else:
        shutil.copy(src, dst)


def _populate_root(tmp_path, fake_cube_file, fake_maps_file, fake_drpall_file):
    root = tmp_path / "manga"
    dst_cube = root / "dr17/manga/spectro/redux/v3_1_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz"
    dst_maps = root / (
        "dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/8485/1901/"
        "manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
    )
    dst_drpall = root / "dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits"
    _gzip_copy(fake_cube_file, dst_cube)
    _gzip_copy(fake_maps_file, dst_maps)
    _gzip_copy(fake_drpall_file, dst_drpall)
    return root


def test_load_cube_and_maps(tmp_path, fake_cube_file, fake_maps_file, fake_drpall_file):
    root = _populate_root(tmp_path, fake_cube_file, fake_maps_file, fake_drpall_file)
    manga = MangaDataset(root=root, allow_download=False)
    cube = manga.load_cube("8485-1901")
    assert cube.plateifu == "8485-1901"
    maps = manga.load_maps("8485-1901")
    assert "STELLAR_VEL" in maps
    drp = manga.drpall
    assert len(drp) == 2


def test_missing_file_raises(tmp_path, fake_cube_file):
    manga = MangaDataset(root=tmp_path / "empty", allow_download=False)
    with pytest.raises(FileNotFoundError):
        manga.load_cube("8485-1901")


def test_invalid_daptype():
    with pytest.raises(ValueError):
        MangaDataset(default_daptype="NOT-A-DAPTYPE")


def test_default_root_uses_repository(tmp_path, monkeypatch):
    monkeypatch.delenv("UDONG_MANGA_ROOT", raising=False)
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_root() == tmp_path / "Repository" / "manga"


def test_truncated_local_file_redownloaded(tmp_path, fake_cube_file, monkeypatch):
    """A truncated .gz file must be detected and re-fetched, not read as-is."""
    import gzip

    root = tmp_path / "manga"
    dst = root / "dr17/manga/spectro/redux/v3_1_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(fake_cube_file, "rb") as f:
        raw = gzip.compress(f.read())
    dst.write_bytes(raw[: len(raw) // 2])  # truncate

    calls = {}

    def fake_download(url, local, **kwargs):
        calls["url"] = url
        with open(fake_cube_file, "rb") as f:
            import gzip as _gz

            with _gz.open(local, "wb") as g:
                g.write(f.read())
        return local

    monkeypatch.setattr("udong.data.manga.dataset.download_file", fake_download)

    # Registry download must also stay hermetic: point it at a tmp cache file
    # and stub the network call.
    mask_file = tmp_path / "maskbits" / "sdssMaskbits.par"
    monkeypatch.setenv("UDONG_MASKPAR", str(mask_file))

    def fake_maskbits(url, local, **kwargs):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_text(
            'masktype MANGA_DRP3PIXMASK 32 "DRP3 pixel mask"\n'
            'maskbits MANGA_DRP3PIXMASK 10 DONOTUSE "do not use"\n'
        )
        return local

    monkeypatch.setattr("udong.data.manga.masks.download_file", fake_maskbits)
    manga = MangaDataset(root=root, allow_download=True)
    cube = manga.load_cube("8485-1901")
    assert cube.plateifu == "8485-1901"
    assert "data.sdss.org" in calls["url"]


def test_truncated_local_file_no_download_raises(tmp_path, fake_cube_file):
    import gzip

    root = tmp_path / "manga"
    dst = root / "dr17/manga/spectro/redux/v3_1_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(fake_cube_file, "rb") as f:
        raw = gzip.compress(f.read())
    dst.write_bytes(raw[: len(raw) // 2])

    manga = MangaDataset(root=root, allow_download=False)
    with pytest.raises(FileNotFoundError, match="truncated"):
        manga.load_cube("8485-1901")


def test_galaxies_near(tmp_path):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    manga = MangaDataset(root=tmp_path / "empty", allow_download=False)
    manga._drpall = Table(
        {
            "plateifu": ["8485-1901", "7443-12703", "11747-12703"],
            "ifura": [232.5447, 200.0, 130.0],
            "ifudec": [48.6902, 20.0, 29.8],
        }
    )
    center = SkyCoord(232.5447, 48.6902, unit="deg")
    sub = manga.galaxies_near(center, 30 * u.arcsec)
    assert len(sub) == 1
    assert sub["plateifu"][0] == "8485-1901"
    assert float(sub["sep_arcsec"][0]) < 1.0
    assert len(manga.galaxies_near(SkyCoord(0.0, 0.0, unit="deg"), 30 * u.arcsec)) == 0
