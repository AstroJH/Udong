"""ESO Archive ObsCore (VO) queries.

ObsCore (`ivoa.obscore` at the ESO TAP endpoint) is the closest thing MUSE
has to a survey catalog: one row per *reduced product* (datacube / spectrum /
image), with `dp_id`, target name, coordinates, mode/pixel scale, wavelength
range, exposure, resolution and release date.  It is a remote, VO-standard
service, so this module lives in the catalog layer (`udong.catalogs`) rather
than in a survey data layer.

Example
-------
>>> from udong.catalogs.obscore import search_muse_products
>>> tab = search_muse_products(ra=14.3343, dec=-22.3822, radius=0.2)  # doctest: +SKIP
>>> tab["dp_id", "target_name", "s_pixel_scale"]                       # doctest: +SKIP
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Sequence
from typing import Any

from astropy import units as u
from astropy.table import Table
from astropy.units import Quantity

__all__ = ["query_obscore", "search_muse_products", "pixel_scale_mode", "EsoQueryError"]

TAP_ENDPOINT = "https://archive.eso.org/tap_obs/sync"
OBSCORE_TABLE = "ivoa.obscore"

# Curated default columns (a subset of the ObsCore standard + ESO extras).
DEFAULT_COLUMNS = (
    "dp_id", "instrument_name", "obs_collection", "proposal_id", "target_name",
    "s_ra", "s_dec", "s_fov", "s_pixel_scale", "s_resolution",
    "dataproduct_type", "dataproduct_subtype",
    "em_min", "em_max", "em_res_power", "em_xel",
    "t_exptime", "calib_level",
    "access_url", "access_estsize", "facility_name", "multi_ob", "n_obs",
    "publication_date", "obs_publisher_did",
)

#: Physical units for well-known ObsCore columns (metadata rarely carries them).
_COLUMN_UNITS = {
    "s_ra": "deg", "s_dec": "deg", "s_fov": "deg",
    "s_pixel_scale": "arcsec", "s_resolution": "arcsec",
    "t_exptime": "s", "em_min": "m", "em_max": "m",
    "em_res_power": "", "em_xel": "", "access_estsize": "kbyte",
}

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


class EsoQueryError(RuntimeError):
    """Raised when an ESO TAP query fails (network, HTTP or ADQL error)."""


def _quote(value: str) -> str:
    """Single-quote a literal for ADQL, doubling embedded quotes."""
    return "'" + str(value).replace("'", "''") + "'"


def _build_where(
    target: str | None,
    ra: float | Quantity | None,
    dec: float | Quantity | None,
    radius: float | Quantity | None,
    instrument: str | None,
    dataproduct_type: str | None,
    dataproduct_subtype: str | None,
) -> str:
    where: list[str] = []
    if instrument:
        # ESO TAP does not support UPPER()/LOWER() in ADQL; match plain LIKE.
        where.append(f"instrument_name LIKE '%{instrument.upper()}%'")
    if dataproduct_type:
        where.append(f"dataproduct_type = {_quote(dataproduct_type)}")
    if dataproduct_subtype:
        where.append(f"dataproduct_subtype = {_quote(dataproduct_subtype)}")
    if target:
        # case-sensitive substring on the PI-provided name; aliases differ
        # (e.g. "Ton S 180" vs "HE 0054-2239") so prefer coordinate search.
        where.append(f"target_name LIKE {_quote('%' + target + '%')}")

    coords = [ra, dec, radius]
    if any(c is not None for c in coords):
        if any(c is None for c in coords):
            raise ValueError("ra, dec and radius must be given together")
        ra_d = ra.to_value(u.deg) if isinstance(ra, Quantity) else float(ra)
        dec_d = dec.to_value(u.deg) if isinstance(dec, Quantity) else float(dec)
        if isinstance(radius, Quantity):
            r_d = radius.to_value(u.deg)
        elif isinstance(radius, (int, float)):
            r_d = float(radius)
        else:
            raise TypeError("radius must be a Quantity or degrees")
        where.append(
            "CONTAINS(POINT('ICRS', s_ra, s_dec), "
            f"CIRCLE('ICRS', {ra_d:.8f}, {dec_d:.8f}, {r_d:.8f})) = 1"
        )
    return " WHERE " + " AND ".join(where) if where else ""


def query_obscore(
    *,
    target: str | None = None,
    ra: float | Quantity | None = None,
    dec: float | Quantity | None = None,
    radius: float | Quantity | None = None,
    instrument: str | None = None,
    dataproduct_type: str | None = None,
    dataproduct_subtype: str | None = None,
    columns: Sequence[str] | None = None,
    top: int | None = None,
    endpoint: str | None = None,
    timeout: float = 120.0,
) -> Table:
    """Query the ESO ObsCore table and return rows as an ``astropy`` Table.

    Parameters
    ----------
    target
        Case-sensitive substring match on the (PI-provided) target name.
        Note ObsCore names are free text and aliases differ (e.g. ``Ton S 180``
        vs ``HE 0054-2239``); prefer ``ra/dec/radius`` when coordinates are known.
    ra, dec, radius
        Cone search centre (deg or Quantity) and radius (deg/arcmin/... as a
        Quantity, or degrees as a float).  All three must be given together.
    instrument, dataproduct_type, dataproduct_subtype
        Extra filters (e.g. ``instrument="MUSE"``, ``dataproduct_type="cube"``).
    columns
        ObsCore columns to select (must be valid identifiers).
    top
        ``TOP n`` row limit for the ADQL query.
    endpoint, timeout
        Override the TAP endpoint / network timeout.
    """
    if columns is None:
        columns = DEFAULT_COLUMNS
    cols = list(columns)
    for c in cols:
        if _IDENT_RE.match(c) is None:
            raise ValueError(f"invalid ObsCore column name: {c!r}")
    select = f"SELECT TOP {int(top)} " if top else "SELECT "
    adql = (
        f"{select}{', '.join(cols)} FROM {OBSCORE_TABLE}"
        + _build_where(target, ra, dec, radius, instrument, dataproduct_type, dataproduct_subtype)
    )
    return _run_query(adql, endpoint=endpoint, timeout=timeout)


def search_muse_products(
    *,
    target: str | None = None,
    ra: float | Quantity | None = None,
    dec: float | Quantity | None = None,
    radius: float | Quantity | None = None,
    dataproduct_type: str | None = "cube",
    **kwargs: Any,
) -> Table:
    """Convenience: query ESO ObsCore for MUSE reduced products (cubes by default)."""
    return query_obscore(
        target=target,
        ra=ra,
        dec=dec,
        radius=radius,
        instrument="MUSE",
        dataproduct_type=dataproduct_type,
        **kwargs,
    )


def pixel_scale_mode(pixel_scale_arcsec: float | None) -> str:
    """Guess MUSE observing mode from the pixel scale (``s_pixel_scale``).

    * ``"WFM"`` -- ~0.2 arcsec (wide field mode);
    * ``"NFM"`` -- ~0.025 arcsec (narrow field mode / AO);
    * ``"unknown"`` otherwise.
    """
    if pixel_scale_arcsec is None:
        return "unknown"
    if 0.15 <= pixel_scale_arcsec <= 0.25:
        return "WFM"
    if 0.02 <= pixel_scale_arcsec <= 0.04:
        return "NFM"
    return "unknown"


# --------------------------------------------------------------------------- #
def _get_text(url: str, timeout: float = 120.0) -> str:
    """HTTP GET via easycat's shared client, wrapping errors for the TAP API."""
    from udong.data.net import http_client

    try:
        with http_client(timeout=timeout, retries=2) as client:
            return client.get_text(url)
    except Exception as exc:  # noqa: BLE001 - normalise as ESO query failure
        # easycat raises structured HttpError (status/url/body_snippet); fall
        # back to the raw response for anything else.
        response = getattr(exc, "response", None)
        status = getattr(exc, "status", None)
        if status is None:
            status = getattr(response, "status_code", None)
        body = getattr(exc, "body_snippet", None)
        if not body and response is not None:
            text = getattr(response, "text", "") or ""
            body = " ".join(text.split())[:500] if text else None
        prefix = f"HTTP {status}: " if status else ""
        suffix = f" -- {body}" if body else ""
        raise EsoQueryError(f"ESO TAP request failed: {prefix}{exc}{suffix}") from exc


def _run_query(
    adql: str,
    endpoint: str | None,
    timeout: float,
) -> Table:
    params = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql, "FORMAT": "json"}
    )
    url = f"{endpoint or TAP_ENDPOINT}?{params}"
    try:
        payload = json.loads(_get_text(url, timeout=timeout))
    except json.JSONDecodeError as exc:
        raise EsoQueryError(f"ESO TAP returned non-JSON body: {exc}") from exc

    if payload.get("error"):
        raise EsoQueryError(f"ESO TAP error: {payload['error']}")
    meta = payload.get("metadata", [])
    names = [m.get("name") for m in meta]
    rows = payload.get("data", [])
    table = Table(rows=rows, names=names)

    # attach physical units where we know them
    for col in table.colnames:
        unit = _COLUMN_UNITS.get(col)
        if unit is not None:
            try:
                table[col].unit = u.Unit(unit)
            except (ValueError, TypeError):
                pass
    return table
