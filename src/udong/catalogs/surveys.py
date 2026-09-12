"""External-survey catalog access built on easycat Archives.

Design (ADR-008, Phase B):

* :data:`SURVEYS` describes the supported surveys (names, aliases, intent);
* per-survey implementations are provided by small adapter modules
  (``udong.catalogs.wise`` / ``.ztf`` / ``.desi`` / ...) and registered with
  :func:`register`;
* :func:`fetch_one` and :func:`fetch_cone` return a :class:`SurveyProduct`
  (downloaded file + astropy Table when the product is tabular + metadata);
* :func:`cone_search` is the convenience layer: it performs a cone search and
  returns the resulting :class:`~astropy.table.Table` (raising for products
  that are not tabular, e.g. DESI spectra/images).

All downloads are delegated to ``easycat.download`` Archives; the on-disk
location is always chosen by the caller (``dest=`` / ``store_dir=``), never by
this package.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u
from astropy.table import Table
from astropy.units import Quantity

__all__ = [
    "SurveySpec",
    "coerce_product_data",
    "SurveyProduct",
    "SURVEYS",
    "available_surveys",
    "resolve",
    "register",
    "fetch_one",
    "fetch_cone",
    "cone_search",
    "read_product_table",
    "SurveyNotImplemented",
]


@dataclass(frozen=True)
class SurveySpec:
    """Static description of a supported external catalog."""

    name: str
    description: str
    access: str
    aliases: tuple[str, ...] = ()
    note: str = ""


SURVEYS: dict[str, SurveySpec] = {
    spec.name: spec
    for spec in [
        SurveySpec(
            name="sdss",
            description="SDSS photometry/spectroscopy of galaxies & AGN hosts.",
            access="SDSS SkyServer CAS/SQL (DR17) or VizieR photometric table (to verify)",
            aliases=("sdss-dr17",),
            note="Spectral products need the science archive server, not VizieR.",
        ),
        SurveySpec(
            name="manga",
            description="MaNGA summary catalog (drpall): coordinates, redshift, NSA properties.",
            access="local drpall via MangaDataset / SDSSArchive(mode='manga')",
            aliases=("drpall", "manga-dr17"),
            note="Use udong.data.manga for data products; this registry is for "
                 "cross-matching on the drpall summary table.",
        ),
        SurveySpec(
            name="wise",
            description="WISE/AllWISE W1-W4 photometry for AGN SEDs and colors.",
            access="easycat WISEArchive (IRSA Gator)",
            aliases=("allwise", "wise-allwise"),
        ),
        SurveySpec(
            name="ztf",
            description="ZTF optical light curves / variability (time domain).",
            access="easycat ZTFArchive (IRSA ZTF LC API)",
            note="Returns a per-object light-curve table (CSV).",
        ),
        SurveySpec(
            name="galex",
            description="GALEX NUV/FUV photometry (star formation, AGN UV excess).",
            access="VizieR TAP GALEX GR6/7 (II/312, table id to verify)",
            aliases=("galext",),
        ),
        SurveySpec(
            name="gaia",
            description="Gaia astrometry + photometry (parallax, proper motion, variability).",
            access="Gaia TAP (https://gea.esac.esa.int/tap-server/tap/sync)",
            aliases=("gaia-dr3", "gaiaedr3"),
            note="Gaia has its own TAP service; coordinate epoch 2016.0.",
        ),
        SurveySpec(
            name="first",
            description="FIRST 20 cm radio (morphology, radio-loud AGN).",
            access="VizieR TAP FIRST (VIII/92, table id to verify)",
        ),
        SurveySpec(
            name="nvss",
            description="NVSS 1.4 GHz radio (arcmin-scale flux densities).",
            access="VizieR TAP NVSS (VIII/65, table id to verify)",
        ),
        SurveySpec(
            name="desi",
            description="DESI Legacy Surveys photometry, spectra and cutouts.",
            access="easycat DESIArchive (photometry / spectra / image)",
            aliases=("legacy-surveys", "ls-dr10"),
            note="mode='photometry' returns a Table; 'spectra'/'image' return files.",
        ),
    ]
}

# Aliases -> canonical name.
_ALIAS: dict[str, str] = {}
for _spec in SURVEYS.values():
    _ALIAS[_spec.name] = _spec.name
    for _a in _spec.aliases:
        _ALIAS[_a] = _spec.name

#: Per-survey product implementations (registered by adapter modules).
_IMPLS: dict[str, Callable[..., "SurveyProduct"]] = {}

#: Adapter modules imported lazily on first use.
_AUTO_IMPORT = {
    "wise": "udong.catalogs.wise",
    "ztf": "udong.catalogs.ztf",
    "desi": "udong.catalogs.desi",
}


@dataclass
class SurveyProduct:
    """Result of a single-target/catalog-archive download.

    ``path`` is the downloaded file (``None`` when the source has no product
    or ``download=False``); ``table`` is the parsed astropy Table for tabular
    products (WISE photometry, ZTF light curves, DESI photometry), else None.
    ``meta`` carries provenance-friendly fields (url, dest, size, checksum,
    product, version, ...) as provided by easycat's ``ItemResult.meta``.
    """

    survey: str
    obj_id: str
    path: Path | None = None
    table: Table | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""


class SurveyNotImplemented(NotImplementedError):
    """Raised when a survey has no registered implementation yet."""


def available_surveys() -> list[str]:
    """Canonical survey names currently described in the registry."""
    return sorted(SURVEYS)


def resolve(name: str) -> SurveySpec:
    """Resolve a survey name/alias to its :class:`SurveySpec`."""
    key = str(name).strip().lower()
    if key not in _ALIAS:
        raise KeyError(f"unknown survey {name!r}; available: {available_surveys()}")
    return SURVEYS[_ALIAS[key]]


def register(name: str):
    """Decorator registering a product implementation for a survey."""

    canonical = _ALIAS.get(name.lower(), name.lower())
    if canonical not in SURVEYS:
        raise KeyError(f"cannot register unknown survey {name!r}")

    def deco(fn: Callable[..., SurveyProduct]) -> Callable[..., SurveyProduct]:
        _IMPLS[canonical] = fn
        return fn

    return deco


def _impl(name: str) -> Callable[..., SurveyProduct]:
    spec = resolve(name)
    if spec.name not in _IMPLS and spec.name in _AUTO_IMPORT:
        importlib.import_module(_AUTO_IMPORT[spec.name])  # registers the adapter
    fn = _IMPLS.get(spec.name)
    if fn is None:
        raise SurveyNotImplemented(
            f"survey {spec.name!r} is a skeleton entry only (not implemented yet).\n"
            f"  planned access: {spec.access}\n"
            f"  register an implementation with udong.catalogs.surveys.register('{spec.name}')"
        )
    return fn


def read_product_table(path: Path | str) -> Table:
    """Read a downloaded product file into an astropy Table.

    Handles FITS tables (WISE/AllWISE, DESI photometry) and CSV light curves
    (ZTF); raises for file types that have no table representation.
    """
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        return Table.read(path, format="ascii.csv")
    if name.endswith((".fits", ".fit", ".fits.gz")):
        try:
            return Table.read(path)          # first table HDU
        except Exception:
            return Table.read(path, hdu=1)
    raise TypeError(f"no tabular representation for {path.name!r}")


def coerce_product_data(data: Any) -> tuple[Path | None, Table | None]:
    """Normalise easycat ``ItemResult.data`` into ``(path, table)``.

    Archive implementations may return a file path, an astropy Table or a
    pandas DataFrame (query-oriented archives keep the result in memory).
    """
    if data is None:
        return None, None
    if isinstance(data, Table):
        return None, data
    if hasattr(data, "columns") and hasattr(data, "to_dict"):  # pandas DataFrame
        return None, Table.from_pandas(data)
    path = Path(data)
    table = None
    try:
        table = read_product_table(path)
    except TypeError:
        table = None
    return path, table


def fetch_one(survey: str, target: Any, **kwargs: Any) -> SurveyProduct:
    """Download one source from a survey Archive (single-target, no checkpoint).

    ``target`` may be an identifier string, a mapping/Series/one-row DataFrame
    (see easycat ``SurveyArchive.fetch_one``).  Extra keyword arguments are
    forwarded to the adapter (``dest``, ``store_dir``, ``download``,
    ``progress``, ``validate``, survey-specific options, ...).
    """
    return _impl(survey)(target, **kwargs)


def _radius_arcsec(radius: float | Quantity) -> float:
    if isinstance(radius, Quantity):
        return float(radius.to_value(u.arcsec))
    return float(radius) * 3600.0  # bare float = degrees (obscore convention)


def fetch_cone(
    survey: str,
    ra: float | Quantity,
    dec: float | Quantity,
    radius: float | Quantity,
    *,
    obj_id: str = "target",
    **kwargs: Any,
) -> SurveyProduct:
    """Download the product for a sky position (RA/Dec/radius cone).

    Creates a one-row target mapping (``obj_id``/``raj2000``/``dej2000``) and
    delegates to the survey adapter; ``radius`` is converted to arcseconds
    (a bare float is interpreted as degrees, matching the Obscore client).
    """
    target = {
        "obj_id": str(obj_id),
        "raj2000": float(Quantity(ra).to_value(u.deg)) if isinstance(ra, Quantity) else float(ra),
        "dej2000": float(Quantity(dec).to_value(u.deg)) if isinstance(dec, Quantity) else float(dec),
    }
    kwargs.setdefault("radius_arcsec", _radius_arcsec(radius))
    return _impl(survey)(target, **kwargs)


def cone_search(
    survey: str,
    ra: float | Quantity,
    dec: float | Quantity,
    radius: float | Quantity,
    top: int | None = None,
    **kwargs: Any,
) -> Table:
    """Cone-search a survey and return the resulting astropy Table.

    This is the tabular convenience layer (WISE photometry, ZTF light curves,
    DESI photometry).  For file-only products (DESI spectra/images) use
    :func:`fetch_one`/:func:`fetch_cone` and inspect ``SurveyProduct.path``.
    """
    if top is not None:
        if resolve(survey).name != "ztf":
            raise ValueError(f"top is only supported for ztf, not {survey!r}")
        kwargs.setdefault("max_objects", top)
    product = fetch_cone(survey, ra, dec, radius, **kwargs)
    if product.success is False:
        raise RuntimeError(f"{survey} fetch failed: {product.error}")
    if product.table is None:
        raise TypeError(
            f"{survey}: product has no tabular representation "
            f"(path={product.path}); use fetch_one()/fetch_cone() instead"
        )
    table = product.table
    table.meta.update(product.meta)
    return table
