"""Provenance records.

Design notes
------------
Every science product should be traceable to its original data, processing
steps, parameters, and software versions.  A :class:`Provenance` record is a
lightweight, serializable description of that chain:

* ``source``: the original file (path or URL),
* ``checksums``: e.g. FITS ``DATASUM``/``CHECKSUM`` and a sha256 of the file,
* ``survey``/``release``/``drpver``/``dapver``: survey-level versioning,
* ``versions``: software versions used to produce/read the product,
* ``steps``: an ordered list of :class:`ProcessingStep` entries.

Serialization is intentionally simple (dict / YAML) so provenance can be
attached to FITS headers and side-car YAML files.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

__all__ = ["ProcessingStep", "Provenance"]


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ProcessingStep:
    """A single, immutable processing step applied to some data.

    Attributes
    ----------
    name
        Short step name, e.g. ``"fit_continuum"`` or ``"sigma_clip"``.
    params
        Free-form parameters of the step (use simple JSON/YAML-able values so
        the record stays serializable).
    version
        Version of the code/algorithm that performed this step.
    input_checksums
        Checksums of the inputs consumed by this step (optional).
    timestamp
        UTC ISO timestamp at which the step was created.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    input_checksums: tuple[str, ...] = ()
    timestamp: str = field(default_factory=_now)


@dataclass
class Provenance:
    """Provenance of a data product.

    Attributes
    ----------
    product
        Type/name of the product the record describes (e.g. ``"MaNGA MAPS
        stellar velocity"``).
    source
        Original local file path.
    url
        Original remote URL (if downloaded).
    survey, release, drpver, dapver
        Survey-level version identifiers (e.g. ``"MaNGA"``, ``"DR17"``,
        ``"v3_1_1"``, ``"3.1.0"``).
    checksums
        File/integrity checksums keyed by name (e.g. ``"sha256"``,
        ``"datasum"``).
    versions
        Software/library versions used to produce/read the product.
    steps
        Ordered list of :class:`ProcessingStep`.
    meta
        Any additional free-form metadata.
    created_at
        UTC ISO timestamp of record creation.
    """

    product: str
    source: str | None = None
    url: str | None = None
    survey: str | None = None
    release: str | None = None
    drpver: str | None = None
    dapver: str | None = None
    checksums: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    steps: list[ProcessingStep] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    # ------------------------------------------------------------------ #
    def add_step(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        version: str | None = None,
        input_checksums: tuple[str, ...] = (),
    ) -> Provenance:
        """Append a processing step and return ``self`` (chainable)."""
        self.steps.append(
            ProcessingStep(
                name=name,
                params=dict(params or {}),
                version=version,
                input_checksums=tuple(input_checksums),
            )
        )
        return self

    def copy(self) -> Provenance:
        """Return a deep (value) copy of this record."""
        return Provenance.from_dict(self.to_dict())

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain (JSON/YAML-able) dict."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Serialize to a YAML string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Provenance:
        """Re-build a record from the dict of :meth:`to_dict`."""
        d = dict(d)
        steps = d.pop("steps", [])
        obj = cls(**d)
        for s in steps:
            obj.steps.append(ProcessingStep(**s))
        return obj

    @classmethod
    def from_yaml(cls, text: str) -> Provenance:
        """Re-build a record from a YAML string produced by :meth:`to_yaml`."""
        return cls.from_dict(yaml.safe_load(text))

    # ------------------------------------------------------------------ #
    def write_fits_header(self, header: Any, prefix: str = "UDONG") -> Any:
        """Write key provenance fields into a FITS header, in place.

        Parameters
        ----------
        header
            An ``astropy.io.fits`` header (or any mapping supporting
            ``header[key] = (value, comment)``).
        prefix
            Header-keyword prefix (must be <= 8 chars to stay FITS-legal without
            HIERARCH cards; default ``"UDONG"``).

        Notes
        -----
        Only the first nine processing steps are written (as ``{prefix}S0`` ...
        ``{prefix}S8``) and step names are truncated to 68 chars, to keep the header
        compact; for full records use YAML sidecar files.
        """
        header[f"{prefix}PROD"] = (self.product, "Udong product type")
        if self.source is not None:
            header[f"{prefix}SRC"] = (self.source, "Udong source file")
        if self.survey is not None:
            header[f"{prefix}SURV"] = (self.survey, "Udong survey")
        if self.release is not None:
            header[f"{prefix}REL"] = (self.release, "Udong data release")
        if self.checksums:
            header[f"{prefix}SUM"] = (
                self.checksums.get("datasum", ""),
                "Udong DATASUM of source FITS data",
            )
        header[f"{prefix}NSTEP"] = (len(self.steps), "Udong processing steps")
        for i, step in enumerate(self.steps[:9]):
            header[f"{prefix}S{i}"] = (step.name[:68], "Udong processing step")
        return header
