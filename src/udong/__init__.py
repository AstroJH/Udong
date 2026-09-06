"""Udong: long-term scientific computing infrastructure for SDSS-IV MaNGA.

The package is organized as:

- ``udong.core``: survey-agnostic core abstractions (Spectrum, Map2D, Cube,
  masks, uncertainty, provenance, coordinates).  No MaNGA-specific code.
- ``udong.data``: data-access layer.  ``udong.data.manga`` contains everything
  that is specific to MaNGA products.
- ``udong.science``: science analysis modules that only use ``udong.core``.
- ``udong.viz``: thin plotting wrappers (matplotlib).
"""

from __future__ import annotations

__version__ = "0.1.0"
