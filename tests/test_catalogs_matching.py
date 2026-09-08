import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from udong.catalogs.matching import (
    crossmatch,
    crossmatch_tables,
    deduplicate,
    match_to_catalog,
)


def test_crossmatch_basic():
    c1 = SkyCoord([0.0, 1.0, 2.0] * u.deg, [0.0, 0.0, 0.0] * u.deg)
    c2 = SkyCoord([0.001, 2.001, 5.0] * u.deg, [0.0, 0.0, 0.0] * u.deg)
    res = crossmatch(c1, c2, radius=5 * u.arcsec)
    assert len(res) == 2
    assert sorted(res.idx1.tolist()) == [0, 2]
    assert sorted(res.idx2.tolist()) == [0, 1]
    assert np.allclose(res.separation.to_value(u.arcsec), 3.6, atol=0.1)


def test_crossmatch_unique_dedup():
    # two sources within radius of the same catalog object -> keep closest
    c1 = SkyCoord([0.0, 0.0005, 3.0] * u.deg, [0.0, 0.0, 0.0] * u.deg)
    c2 = SkyCoord([0.0006, 3.0] * u.deg, [0.0, 0.0] * u.deg)
    res = crossmatch(c1, c2, radius=10 * u.arcsec)
    assert len(res) == 2
    assert res.n_duplicates_dropped == 1
    assert 0 not in res.idx1  # the farther duplicate (idx 0) was dropped


def test_crossmatch_none():
    c1 = SkyCoord([0.0] * u.deg, [0.0] * u.deg)
    c2 = SkyCoord([10.0] * u.deg, [10.0] * u.deg)
    res = crossmatch(c1, c2, radius=1 * u.arcsec)
    assert len(res) == 0


def test_match_to_catalog():
    c1 = SkyCoord([0.0, 10.0] * u.deg, [0.0, 0.0] * u.deg)
    c2 = SkyCoord([0.001] * u.deg, [0.0] * u.deg)
    idx, sep, ok = match_to_catalog(c1, c2, radius=5 * u.arcsec)
    assert idx[0] == 0 and ok[0]
    assert not ok[1]


def test_crossmatch_tables():
    t1 = Table({"id": [1, 2, 3], "ra": [0.0, 2.0, 5.0], "dec": [0.0, 0.0, 0.0]})
    t2 = Table({"name": ["a", "b"], "ra": [0.001, 2.001], "dec": [0.0, 0.0]})
    out = crossmatch_tables(t1, t2, "ra", "dec", "ra", "dec", radius=5 * u.arcsec)
    assert len(out) == 2
    assert "match_idx" in out.colnames
    assert "match_separation_arcsec" in out.colnames
    assert set(out["id"].tolist()) == {1, 2}


def test_deduplicate():
    c = SkyCoord([0.0, 0.0004, 2.0] * u.deg, [0.0, 0.0, 0.0] * u.deg)
    groups, n_neigh = deduplicate(c, radius=5 * u.arcsec)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert n_neigh[0] >= 1 and n_neigh[2] == 0
