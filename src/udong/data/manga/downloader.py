"""Resumable, verified MaNGA downloads with a Udong user prompt.

Transport, atomic ``.part`` handling, resume, size/gzip verification and the
progress bar are provided by ``easycat.download.HttpClient.download_file``;
this module adds only the MaNGA-facing API and the ``[udong] downloading …``
text prompt.
"""

from __future__ import annotations

import gzip
import sys
from collections.abc import Callable
from pathlib import Path

from udong.data.net import http_client, progress_enabled

__all__ = ["download_file", "gzip_ok"]


def gzip_ok(path: Path, chunk: int = 1 << 20) -> bool:
    """True when ``path`` is a readable gzip stream (full decompress test)."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(chunk):
                pass
        return True
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def download_file(
    url: str,
    dest: Path | str,
    retries: int = 5,
    timeout: int = 30,
    chunk: int = 1 << 16,
    verify_gzip: bool = True,
    progress: Callable[[int, int], None] | bool | None = None,
) -> Path:
    """Download ``url`` to ``dest`` (atomic, resumable, verified).

    ``.gz`` destinations are validated as gzip; the byte count is always
    checked against the server's declared size.  A ``[udong] downloading …``
    line is printed unless ``progress=False``.
    """
    dest = Path(dest)
    quiet = progress is False
    if not quiet:
        sys.stderr.write(f"[udong] downloading {Path(url).name or url}\n")
        sys.stderr.write(f"         -> {dest}\n")
    if progress is False:
        show = False
    elif callable(progress):
        show = progress
    else:
        show = progress_enabled()

    client = http_client(timeout=timeout, retries=max(1, retries))
    try:
        client.download_file(
            url,
            dest,
            chunk_size=chunk,
            resume="auto",
            verify_size=True,
            trust_existing=False,
            validate="gzip" if (verify_gzip and dest.suffix == ".gz") else None,
            cleanup_partial=False,
            progress=show,
        )
    finally:
        client.close()
    return dest
