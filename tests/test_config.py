from __future__ import annotations

import dataclasses
import json

import pytest

from transcript_pipeline.config import PipelineConfig, slugify


def test_defaults_are_generic_and_serializable() -> None:
    config = PipelineConfig()

    assert config.to_dict() == {
        "title": "Recorded session",
        "timezone": "UTC",
        "model_name": "large-v3-turbo",
        "language": None,
        "min_speakers": None,
        "max_speakers": None,
        "num_speakers": None,
    }
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()
    assert config.slug == "recorded-session"


def test_overrides_are_normalized_and_immutable() -> None:
    config = PipelineConfig.from_overrides(
        title="  Quarterly   review  ",
        timezone="Europe/Berlin",
        language="AUTO",
        min_speakers=2,
        max_speakers=4,
    )

    assert config.title == "Quarterly review"
    assert config.language is None
    assert config.slug == "quarterly-review"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.title = "changed"  # type: ignore[misc]


def test_round_trip_from_manifest_mapping_ignores_unknown_keys() -> None:
    original = PipelineConfig(title="Team sync", num_speakers=3)
    restored = PipelineConfig.from_mapping({**original.to_dict(), "legacy_field": True})

    assert restored == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": ""},
        {"title": "x" * 201},
        {"timezone": "Mars/Olympus_Mons"},
        {"model_name": " "},
        {"language": 5},
        {"min_speakers": 0},
        {"max_speakers": True},
        {"min_speakers": 4, "max_speakers": 2},
        {"num_speakers": 5, "max_speakers": 3},
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        PipelineConfig.from_overrides(**overrides)


def test_config_carries_no_credential_fields() -> None:
    names = {field.name for field in dataclasses.fields(PipelineConfig)}

    assert not any(
        marker in name for name in names for marker in ("token", "secret", "password", "key")
    )


def test_transcription_settings_mirror_config_without_importing_ml_stack() -> None:
    import sys

    settings = PipelineConfig(language="de", min_speakers=2).transcription_settings()

    assert settings.language == "de"
    assert settings.min_speakers == 2
    assert settings.hf_token_env == "HF_TOKEN"
    assert "torch" not in sys.modules
    assert "whisperx" not in sys.modules


@pytest.mark.parametrize(
    ("value", "expected"),
    [("Team Sync #4 (draft)", "team-sync-4-draft"), ("***", "recording"), ("Ünïcode talk", "unicode-talk")],
)
def test_slugify_is_filesystem_safe(value: str, expected: str) -> None:
    assert slugify(value) == expected
