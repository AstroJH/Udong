"""Data-access layer.

``udong.data.base`` defines the survey-agnostic dataset interface (Protocol);
``udong.data.manga`` contains everything specific to SDSS MaNGA products.
"""

from __future__ import annotations

from udong.data.base import IFUDataset

__all__ = ["IFUDataset"]
