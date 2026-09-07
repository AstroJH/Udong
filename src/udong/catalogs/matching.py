"""Sky-coordinate matching.

``udong.catalogs`` is the survey-agnostic *catalog layer*: it provides generic
cross-matching utilities (and, later, thin query wrappers for external
catalogs such as SDSS / WISE / FIRST).

This module implements nearest-neighbour cross-matching with an explicit
matching radius, separation output, and duplicate handling.  It is built on
``astropy.coordinates`` (``SkyCoord.match_to_catalog_sky`` /
``search_around_sky``) rather than reimplementing spherical geometry.

Example
-------
>>> from astropy.coordinates import SkyCoord
>>> from astropy import units as u
>>> c1 = SkyCoord([0.0, 1.0, 2.0]*u.deg, [0.0, 0.0, 0.0]*u.deg)
>>> c2 = SkyCoord([0.001, 2.001, 5.0]*u.deg, [0.0, 0.0, 0.0]*u.deg)
>>> res = crossmatch(c1, c2, radius=5*u.arcsec)
>>> len(res)  # two pairs
2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity

__all__ = ["CrossMatchResult", "match_to_catalog", "crossmatch", "deduplicate", "crossmatch_tables"]


@dataclass
class CrossMatchResult:
    """One-to-one nearest-neighbour cross-match between two coordinate sets."""

    idx1: np.ndarray  # indices into the first coordinate set
    idx2: np.ndarray  # indices into the second coordinate set
    separation: Quantity  # angular separation (same length as idx1/idx2)
    radius: Quantity  # matching radius used
    n_duplicates_dropped: int = 0

    def __len__(self) -> int:
        return len(self.idx1)

    def to_table(self) -> Table:
        t = Table()
        t["idx1"] = self.idx1
        t["idx2"] = self.idx2
        t["separation_arcsec"] = self.separation.to_value(u.arcsec)
        return t


def match_to_catalog(coords: SkyCoord, catalog: SkyCoord, radius: Quantity):
    """Nearest neighbour in ``catalog`` for every entry of ``coords``.

    Returns ``(idx, separation, in_radius)`` where ``idx[i]`` is the index of
    the nearest catalog object to ``coords[i]`` and ``separation[i]`` its
    angular distance.  ``in_radius`` flags entries within ``radius``.
    """
    radius = Quantity(radius)
    idx, sep2d, _ = coords.match_to_catalog_sky(catalog)
    sep = Quantity(sep2d)
    return idx, sep, sep < radius


def crossmatch(
    coord1: SkyCoord,
    coord2: SkyCoord,
    radius: Quantity,
    unique: bool = True,
) -> CrossMatchResult:
    """Nearest-neighbour cross-match of ``coord1`` against ``coord2``.

    Every object in ``coord1`` is matched to its nearest object in ``coord2``
    within ``radius``.  With ``unique=True`` (default), each ``coord2`` object
    is used at most once (ties resolved by keeping the closest match);
    duplicate pairs are reported via ``n_duplicates_dropped``.
    """
    radius = Quantity(radius).to(u.arcsec)
    idx, sep, ok = match_to_catalog(coord1, coord2, radius)
    i1 = np.nonzero(ok)[0]
    i2 = np.asarray(idx)[i1]
    seps = sep[i1]
    dropped = 0
    if unique and len(i1) > 0:
        # for each coord2 index, keep only the closest coord1 match
        order = np.argsort(seps, kind="stable")
        i1, i2, seps = i1[order], i2[order], seps[order]
        keep = np.ones(len(i1), dtype=bool)
        seen: set[int] = set()
        for k, j in enumerate(i2):
            j = int(j)
            if j in seen:
                keep[k] = False
                dropped += 1
            else:
                seen.add(j)
        i1, i2, seps = i1[keep], i2[keep], seps[keep]
    return CrossMatchResult(idx1=i1, idx2=i2, separation=seps, radius=radius, n_duplicates_dropped=dropped)


def deduplicate(coords: SkyCoord, radius: Quantity) -> tuple[np.ndarray, np.ndarray]:
    """Flag entries that have a neighbour within ``radius`` (excluding self).

    Returns ``(group_id, n_neighbors)`` where entries sharing a group lie
    within ``radius`` of at least one other member (connected components via
    union-find on the mutual-neighbour graph), and ``n_neighbors`` counts
    neighbours within ``radius`` for each entry.
    """
    radius = Quantity(radius).to(u.arcsec)
    n = len(coords)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # neighbours within radius (self-matches and mirrored pairs excluded)
    idx1, idx2, _, _ = coords.search_around_sky(coords, radius)
    n_neighbors = np.zeros(n, dtype=int)
    for a, b in zip(idx1, idx2, strict=True):
        a, b = int(a), int(b)
        if a < b:  # count each unordered pair once
            n_neighbors[a] += 1
            n_neighbors[b] += 1
            union(a, b)
    # relabel groups
    groups = {find(i): g for g, i in enumerate(sorted({find(i) for i in range(n)}))}
    group_id = np.array([groups[find(i)] for i in range(n)])
    return group_id, n_neighbors


def crossmatch_tables(
    table1: Table,
    table2: Table,
    ra1: str,
    dec1: str,
    ra2: str,
    dec2: str,
    radius: Quantity,
    *,
    unique: bool = True,
    prefix2: str = "match",
) -> Table:
    """Cross-match two astropy tables by sky coordinates.

    Returns a copy of ``table1`` restricted to matched rows with extra columns
    ``{prefix2}_idx`` (index into table2), ``{prefix2}_separation_arcsec``,
    and ``{prefix2}_n_duplicates``.
    """
    coord1 = SkyCoord(table1[ra1], table1[dec1], unit="deg")
    coord2 = SkyCoord(table2[ra2], table2[dec2], unit="deg")
    res = crossmatch(coord1, coord2, radius, unique=unique)
    out = table1[res.idx1].copy()
    out[f"{prefix2}_idx"] = res.idx2
    out[f"{prefix2}_separation_arcsec"] = res.separation.to_value(u.arcsec)
    out[f"{prefix2}_n_duplicates"] = np.full(len(out), res.n_duplicates_dropped)
    out.meta["match_radius_arcsec"] = res.radius.to_value(u.arcsec)
    out.meta["match_unique"] = unique
    out.meta["match_n_duplicates_dropped"] = res.n_duplicates_dropped
    return out
