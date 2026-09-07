"""Resumable, verifiable downloads for MaNGA SAS files.

The SDSS SAS is frequently slow or flaky; this module implements a minimal
HTTP range-resume downloader with retries and an optional integrity check.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

__all__ = ["download_file", "gzip_ok", "sha256_file"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def gzip_ok(path: Path, chunk: int = 1 << 20) -> bool:
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
    progress: Callable[[int, int], None] | None = None,
    http_11: bool = True,
) -> Path:
    """Download ``url`` to ``dest``, resuming partial files and retrying.

    Returns the destination path.  ``verify_gzip`` runs a full gzip integrity
    check for ``.gz`` files after a successful download.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", 200)
                mode = "ab" if existing and code == 206 else "wb"
                if code != 206:
                    existing = 0
                total = existing + int(resp.headers.get("Content-Length") or 0)
                got = existing
                with open(dest, mode) as f:
                    while True:
                        b = resp.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        if progress is not None:
                            progress(got, total)
            if dest.suffix == ".gz" and verify_gzip and not gzip_ok(dest):
                raise OSError(f"gzip integrity check failed for {dest.name}")
            return dest
        except Exception:  # noqa: BLE001 - retry on any network error
            if attempt == retries:
                raise
            import time

            time.sleep(min(2 ** attempt, 30))
            
    raise RuntimeError("unreachable")  # pragma: no cover
