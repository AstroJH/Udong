"""Resumable, verifiable downloads for MaNGA SAS files with visible feedback.

The SDSS SAS is frequently slow or flaky; this module implements a minimal
HTTP range-resume downloader with retries and an optional integrity check.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

__all__ = ["download_file", "gzip_ok", "sha256_file"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file (streamed, low memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def gzip_ok(path: Path, chunk: int = 1 << 20) -> bool:
    """True when ``path`` is a readable gzip stream (full decompress test)."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(chunk):
                pass
        return True
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def _progress_enabled() -> bool:
    """Show a progress bar for interactive runs (terminal or Jupyter).

    Set ``UDONG_NO_PROGRESS=1`` to disable the bar (the text prompt is kept
    unless a caller explicitly passes ``progress=False``).
    """
    if os.environ.get("UDONG_NO_PROGRESS") in ("1", "true", "True"):
        return False
    try:
        if "ipykernel" in sys.modules or "ipykernel_launcher" in sys.modules:
            return True
        return bool(sys.stderr.isatty())
    except Exception:  # pragma: no cover - defensive
        return False


def download_file(
    url: str,
    dest: Path | str,
    retries: int = 5,
    timeout: int = 30,
    chunk: int = 1 << 16,
    verify_gzip: bool = True,
    progress: Callable[[int, int], None] | bool | None = None,
    http_11: bool = True,
) -> Path:
    """Download ``url`` to ``dest``, resuming partial files and retrying.

    User feedback (unless ``progress=False``):

    * a text line naming the remote file and the local destination, printed
      to ``stderr`` before the transfer starts;
    * a byte progress bar for interactive terminals and Jupyter notebooks.

    Parameters
    ----------
    url
        Remote URL.
    dest
        Destination path (created parent directories as needed).
    retries, timeout, chunk, http_11
        Low-level transfer controls (see source).
    verify_gzip
        Run a full gzip integrity check for ``.gz`` files after download.
    progress
        * ``None`` -- default behaviour (text prompt + progress bar).
        * ``False`` -- completely silent.
        * ``Callable[[int, int], None]`` -- custom reporter, called with
          ``(bytes_written, total_bytes)`` after every chunk.

    Returns
    -------
    Path
        The destination path.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    quiet = progress is False
    custom_cb = callable(progress)
    show_bar = not quiet and (progress is None) and _progress_enabled()

    printed = False
    bar: tqdm | None = None

    def report(got: int, total: int) -> None:
        nonlocal printed, bar
        if custom_cb:
            progress(got, total)  # type: ignore[misc]
            return
        if quiet:
            return
        if not printed:
            printed = True
            name = Path(url).name or url
            sys.stderr.write(f"[udong] downloading {name}\n")
            sys.stderr.write(f"         -> {dest}\n")
        if show_bar:
            if bar is None:
                bar = tqdm(
                    total=total if total and total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    leave=False,
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            if total and bar.total != total:
                bar.total = total
            bar.n = got
            bar.refresh()

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
                        report(got, total)
            if bar is not None:
                bar.close()
            if dest.suffix == ".gz" and verify_gzip and not gzip_ok(dest):
                raise OSError(f"gzip integrity check failed for {dest.name}")
            return dest
        except Exception:  # noqa: BLE001 - retry on any network error
            if bar is not None:
                bar.close()
                bar = None
            if attempt == retries:
                raise
            import time

            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError("unreachable")  # pragma: no cover
