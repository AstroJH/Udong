"""HTTP access for the data layer, delegated to ``easycat``.

Udong is tightly bound to :mod:`easycat.download`: session reuse, retries,
rate limiting, resumable/atomic downloads, verification and structured
errors all come from ``easycat.download.HttpClient``.  This module is the
single import point (and applies Udong's User-Agent).
"""

from __future__ import annotations

import os
import sys
from typing import Any

from easycat.download import HttpClient

__all__ = ["UDONG_USER_AGENT", "http_client", "progress_enabled"]

#: Identifies Udong (and the toolkit it builds on) to archive servers.
UDONG_USER_AGENT = "udong/0.1 (+easycat)"


def http_client(**kwargs: Any) -> HttpClient:
    """Return a new ``easycat.download.HttpClient`` with ``kwargs`` applied."""
    kwargs.setdefault("user_agent", UDONG_USER_AGENT)
    return HttpClient(**kwargs)


def progress_enabled() -> bool:
    """Whether an interactive byte-progress bar should be shown.

    Set ``UDONG_NO_PROGRESS=1`` to disable; the text prompt printed by the
    downloaders is kept unless a caller passes ``progress=False`` explicitly.
    """
    if os.environ.get("UDONG_NO_PROGRESS") in ("1", "true", "True"):
        return False
    try:
        if "ipykernel" in sys.modules or "ipykernel_launcher" in sys.modules:
            return True
        return bool(sys.stderr.isatty())
    except Exception:  # pragma: no cover - defensive
        return False
