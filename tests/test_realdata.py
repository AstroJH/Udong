"""Real-data smoke tests (run with --run-realdata).

These verify the readers against files downloaded from the SDSS SAS.
Expected location: /tmp/udong_research/ (or set UDONG_REALDATA_DIR).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from udong.data.manga.reader import read_cube, read_maps
from udong.science.spatial import radial_profile

pytestmark = pytest.mark.realdata

DATA_DIR = Path(os.environ.get("UDONG_REALDATA_DIR", "/tmp/udong_research"))

REPO = Path.home() / "Repository" / "manga" / "dr17" / "manga" / "spectro"


def _first(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


MAPS_FILE = _first(
    REPO / "analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/8485/1901/"
    "manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz",
    DATA_DIR / "manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz",
)
CUBE_FILE = _first(
    REPO / "redux/v3_1_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz",
    DATA_DIR / "manga-8485-1901-LOGCUBE.fits.gz",
)


@pytest.mark.skipif(not MAPS_FILE.exists(), reason="real MAPS file not present")
def test_real_maps_read_and_radial_profile():
    maps = read_maps(MAPS_FILE)
    assert maps.plateifu == "8485-1901"
    assert maps.mangaid == "1-209232"
    assert maps.daptype == "HYB10-MILESHC-MASTARSSP"

    vel = maps["STELLAR_VEL"]
    assert vel.value.unit == u.km / u.s
    assert vel.shape == (34, 34)
    assert vel.bad.any()  # there should be masked spaxels

    ha = maps.emission_line_flux("Ha-6564")
    assert ha.shape == (34, 34)
    assert "Ha-6564" in maps.emission_lines
    assert ha.value.unit == 1e-17 * u.erg / (u.s * u.cm**2)

    # radial profile of Halpha flux (median over good spaxels)
    prof = radial_profile(ha, bins=6, statistic="median")
    assert len(prof.radius) == 6
    assert np.isfinite(prof.value.value).any()

    # elliptical coordinates provided by DAP
    ell = maps.channel("SPX_ELLCOO", "R/Re")
    assert ell.shape == (34, 34)


@pytest.mark.skipif(not CUBE_FILE.exists(), reason="real cube file not present")
def test_real_cube_read():
    cube = read_cube(CUBE_FILE)
    assert cube.plateifu == "8485-1901"
    assert cube.shape == (4563, 34, 34)
    assert cube.nwave == 4563
    assert cube.flux.unit == 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    spec = cube.spectrum(18, 18)
    assert len(spec) == 4563
