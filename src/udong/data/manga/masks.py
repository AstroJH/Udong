"""MaNGA mask-bit definitions.

Design
------
The authoritative definitions live in the SDSS ``idlutils`` repository as
``sdssMaskbits.par`` (the same file the SDSS Marvin package downloads at
install time) -- SDSS maintains them, **not** this repository.  Udong
therefore does not ship a copy under ``src/``:

* the file is cached on first use at :func:`default_maskbits_file`
  (default ``~/Repository/udong_data/manga/sdssMaskbits.par``, override the
  location with the ``UDONG_MASKPAR`` environment variable);
* :func:`fetch_maskbits` downloads the official file on demand;
  :func:`ensure_maskbits` auto-downloads when the file is missing and
  ``download=True``;
* high-level readers (:meth:`~udong.data.manga.dataset.MangaDataset.load_cube`
  / ``load_maps``) auto-download through ``MangaDataset.allow_download``;
  the lower-level ``reader.read_*`` functions degrade gracefully to
  ``mask_defs=None`` (any non-zero mask bit counts as bad) when the file is
  not available and downloading is not requested.

Notes
-----
* ``MANGA_DRP3PIXMASK`` is the per-spaxel mask of DRP data cubes (stored as a
  3-D mask cube in LOGCUBE files).
* ``MANGA_DAPPIXMASK`` is the per-spaxel mask of DAP MAPS quantities (stored
  as 32-bit integers in the FITS files even though the definition allows 64
  bits; the highest defined bit is ``DONOTUSE = 2**30``).
* ``MANGA_DRP3QUAL`` / ``MANGA_DAPQUAL`` are global (per-cube) quality flags.
* ``MANGA_DAPSPECMASK`` applies to DAP *model cube* pixels.
  The mangadap 4.2.0 docs define two additional bits (ELFAILED=8,
  NOMODEL=9) so re-validate against a model-cube file before use.
"""

from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache
from pathlib import Path

from udong.core.mask import Bit, MaskDefs
from udong.data.manga.downloader import download_file

__all__ = [
    "MASKBITS_URL",
    "MASK_GROUPS",
    "default_maskbits_file",
    "ensure_maskbits",
    "fetch_maskbits",
    "get_mask_defs",
    "load_mask_registries",
    "mask_defs_from_par",
    "mask_defs_or_none",
]

# Official upstream (SDSS idlutils SVN; the same URL SDSS Marvin fetches).
MASKBITS_URL = (
    "https://svn.sdss.org/public/repo/sdss/idlutils/trunk/"
    "data/sdss/sdssMaskbits.par"
)

_MASKTYPE_RE = re.compile(r'masktype\s+(\S+)\s+(\d+)\s+"(.*)"')
_MASKBITS_RE = re.compile(r'maskbits\s+(\S+)\s+(\d+)\s+(\S+)\s+"(.*)"')

# Group name -> numpy dtype (bit positions come from the upstream file).
_MANGA_GROUPS: dict[str, str] = {
    "MANGA_DRP2PIXMASK": "uint32",
    "MANGA_DRP3PIXMASK": "uint32",
    "MANGA_DAPPIXMASK": "uint64",
    "MANGA_DAPQUAL": "uint64",
    "MANGA_DAPSPECMASK": "uint64",
    "MANGA_DRP3QUAL": "uint64",
    "MANGA_TARGET1": "uint64",
    "MANGA_TARGET2": "uint64",
    "MANGA_TARGET3": "uint64",
}

MASK_GROUPS: tuple[str, ...] = tuple(_MANGA_GROUPS)


# --------------------------------------------------------------------------- #
# Locating / fetching the registry file
# --------------------------------------------------------------------------- #
def default_maskbits_file() -> Path:
    """Local cache path of ``sdssMaskbits.par``.

    Uses the ``UDONG_MASKPAR`` environment variable when set, otherwise
    ``~/Repository/udong_data/manga/sdssMaskbits.par`` (same external data
    cache as the E-MILES SSP models, outside the source tree).
    """
    env = os.environ.get("UDONG_MASKPAR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Repository" / "udong_data" / "manga" / "sdssMaskbits.par"


def fetch_maskbits(url: str | None = None, local: Path | str | None = None) -> Path:
    """Download the official ``sdssMaskbits.par`` into the local cache.

    Parameters
    ----------
    url
        Override the upstream URL (defaults to :data:`MASKBITS_URL`).
    local
        Destination path (defaults to :func:`default_maskbits_file`).

    Returns
    -------
    Path
        The downloaded file.
    """
    local = Path(local) if local is not None else default_maskbits_file()
    return download_file(url or MASKBITS_URL, local, verify_gzip=False)


def ensure_maskbits(
    local: Path | str | None = None,
    url: str | None = None,
    *,
    download: bool = True,
) -> Path:
    """Return an existing ``sdssMaskbits.par``, fetching it if necessary.

    Parameters
    ----------
    local
        Cache path to look at (defaults to :func:`default_maskbits_file`).
    url
        Upstream URL used when downloading.
    download
        When the file is missing and this is ``True``, download it;
        otherwise raise ``FileNotFoundError`` with instructions.
    """
    local = Path(local) if local is not None else default_maskbits_file()
    if local.exists() and local.stat().st_size > 0:
        return local
    if not download:
        raise FileNotFoundError(
            f"sdssMaskbits.par not found at {local}\n"
            "  Run udong.data.manga.fetch_maskbits() (or use a MangaDataset "
            f"with allow_download=True) to fetch it from:\n  {url or MASKBITS_URL}"
        )
    return fetch_maskbits(url=url, local=local)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def mask_defs_from_par(text: str) -> dict[str, MaskDefs]:
    """Parse all MaNGA mask groups from ``sdssMaskbits.par`` text.

    Parameters
    ----------
    text
        Content of the upstream ``sdssMaskbits.par`` file.

    Returns
    -------
    dict
        ``{group_name: MaskDefs}`` for the groups listed in ``MASK_GROUPS``
        that are defined in the file.  The ``MaskDefs.source`` records the
        upstream URL and a sha256 of the file content.
    """
    import hashlib

    groups: dict[str, list[Bit]] = {name: [] for name in _MANGA_GROUPS}
    current: str | None = None
    for line in text.splitlines():
        m = _MASKTYPE_RE.match(line)
        if m:
            current = m.group(1)
            continue
        if current is None or current not in _MANGA_GROUPS:
            continue
        m = _MASKBITS_RE.match(line)
        if m and m.group(1) == current:
            groups[current].append(
                Bit(name=m.group(3), position=int(m.group(2)), description=m.group(4))
            )
    digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]
    source = f"{MASKBITS_URL} (sha256:{digest})"
    out: dict[str, MaskDefs] = {}
    for gname, dtype in _MANGA_GROUPS.items():
        if groups[gname]:
            out[gname] = MaskDefs(gname, groups[gname], dtype=dtype, source=source)
    return out


@lru_cache(maxsize=8)
def _parsed_at(path: str) -> dict[str, MaskDefs]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return mask_defs_from_par(text)


def load_mask_registries(
    local: Path | str | None = None,
    *,
    download: bool = True,
) -> dict[str, MaskDefs]:
    """Load (and cache) all MaNGA mask registries from the local par file.

    See :func:`ensure_maskbits` for the download behaviour.
    """
    path = ensure_maskbits(local=local, download=download)
    return _parsed_at(str(path.resolve()))


def get_mask_defs(
    name: str,
    *,
    local: Path | str | None = None,
    download: bool = True,
) -> MaskDefs:
    """Return the :class:`~udong.core.mask.MaskDefs` for one MaNGA group.

    Parameters
    ----------
    name
        One of ``MASK_GROUPS`` (e.g. ``"MANGA_DRP3PIXMASK"``).
    local
        Cache path of the par file (see :func:`ensure_maskbits`).
    download
        Download the file when missing (see :func:`ensure_maskbits`).

    Raises
    ------
    KeyError
        If ``name`` is not one of the known MaNGA groups, or the downloaded
        file does not define it.
    """
    if name not in _MANGA_GROUPS:
        raise KeyError(f"unknown MaNGA mask group {name!r}; expected one of {MASK_GROUPS}")
    regs = load_mask_registries(local=local, download=download)
    try:
        return regs[name]
    except KeyError as e:  # file downloaded but group missing
        raise KeyError(
            f"mask group {name!r} not found in the sdssMaskbits.par file at "
            f"{Path(local).resolve() if local else default_maskbits_file().resolve()}"
        ) from e


_warned: set[str] = set()


def mask_defs_or_none(name: str, *, local: Path | str | None = None) -> MaskDefs | None:
    """Best-effort registry lookup that never downloads or raises.

    Returns ``None`` (with a one-time warning) when the par file is not
    available; in that case containers keep their raw masks and ``bad`` is
    defined as *any non-zero* mask bit.  Used by the low-level
    ``reader.read_*`` defaults.
    """
    try:
        return get_mask_defs(name, local=local, download=False)
    except (FileNotFoundError, KeyError):
        if name not in _warned:
            _warned.add(name)
            warnings.warn(
                f"sdssMaskbits.par not available: mask group {name!r} will not be "
                "attached (raw masks kept; 'bad' = any non-zero bit). "
                "Run udong.data.manga.fetch_maskbits() to download the official "
                "definitions.",
                stacklevel=2,
            )
        return None
