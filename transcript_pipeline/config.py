"""Credential-free, immutable configuration for one pipeline run."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TITLE = "Recorded session"
DEFAULT_TIMEZONE = "UTC"
DEFAULT_MODEL_NAME = "large-v3-turbo"
_MAX_TITLE_LENGTH = 200
_SLUG_INVALID = re.compile(r"[^a-z0-9]+")


def _optional_positive_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def slugify(value: str, *, fallback: str = "recording", limit: int = 48) -> str:
    """Return a lowercase ASCII slug that is safe inside a directory name."""

    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_INVALID.sub("-", ascii_text.casefold()).strip("-")[:limit].strip("-")
    return slug or fallback


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Effective settings for a run. No field may ever carry a credential."""

    title: str = DEFAULT_TITLE
    timezone: str = DEFAULT_TIMEZONE
    model_name: str = DEFAULT_MODEL_NAME
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be non-empty text")
        title = " ".join(self.title.split())
        if len(title) > _MAX_TITLE_LENGTH:
            raise ValueError(f"title must not exceed {_MAX_TITLE_LENGTH} characters")

        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError("timezone must be a non-empty IANA name")
        timezone = self.timezone.strip()
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {timezone}") from exc

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be non-empty text")

        language = self.language
        if language is not None:
            if not isinstance(language, str):
                raise TypeError("language must be a language code, 'auto', or None")
            language = language.strip() or None
            if language is not None and language.lower() == "auto":
                language = None

        minimum = _optional_positive_int("min_speakers", self.min_speakers)
        maximum = _optional_positive_int("max_speakers", self.max_speakers)
        exact = _optional_positive_int("num_speakers", self.num_speakers)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("min_speakers cannot exceed max_speakers")
        if exact is not None and (
            (minimum is not None and exact < minimum)
            or (maximum is not None and exact > maximum)
        ):
            raise ValueError("num_speakers must lie within min_speakers..max_speakers")

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "timezone", timezone)
        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(self, "language", language)

    @classmethod
    def from_overrides(cls, **overrides: object) -> PipelineConfig:
        return cls(**overrides)  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PipelineConfig:
        """Rebuild a configuration persisted in a manifest, ignoring unknown keys."""

        if not isinstance(value, Mapping):
            raise TypeError("configuration must be a mapping")
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value[key] for key in allowed if key in value})

    @property
    def slug(self) -> str:
        return slugify(self.title)

    def transcription_settings(self) -> Any:
        """Return transcription settings without importing the ML stack."""

        from .transcription import TranscriptionSettings

        return TranscriptionSettings(
            model_name=self.model_name,
            language=self.language,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            num_speakers=self.num_speakers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "timezone": self.timezone,
            "model_name": self.model_name,
            "language": self.language,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "num_speakers": self.num_speakers,
        }


__all__ = ["DEFAULT_TIMEZONE", "DEFAULT_TITLE", "PipelineConfig", "slugify"]
