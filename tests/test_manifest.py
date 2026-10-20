import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.manifest import (
    Manifest,
    ManifestError,
    ManifestPathError,
    ManifestTransitionError,
    RunManifest,
)
from transcript_pipeline.models import StageStatus, from_utc_iso, to_utc_iso
from transcript_pipeline.redaction import REDACTED


def manifest_at(run_dir: Path, **kwargs: object) -> Manifest:
    return Manifest(run_dir / ".work" / "manifest.json", run_dir=run_dir, **kwargs)


def test_utc_helpers_require_and_round_trip_utc() -> None:
    timestamp = datetime(2026, 7, 17, 18, 30, 45, 123456, tzinfo=UTC)

    encoded = to_utc_iso(timestamp)

    assert encoded == "2026-07-17T18:30:45.123456Z"
    assert from_utc_iso(encoded) == timestamp
    with pytest.raises(ValueError, match="timezone-aware"):
        to_utc_iso(datetime(2026, 7, 17, 18, 30))


def test_stage_statuses_are_stable_string_values() -> None:
    assert {status.value for status in StageStatus} == {
        "pending",
        "running",
        "completed",
        "failed",
        "blocked",
    }


def test_completed_stage_is_persisted_and_skipped(tmp_path: Path) -> None:
    path = tmp_path / ".work" / "manifest.json"
    manifest = Manifest(path, run_dir=tmp_path)

    manifest.start("media")
    manifest.complete("media", artifacts={"flac": "audio.flac"})

    assert manifest.should_run("media") is False
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["stages"]["media"]["status"] == "completed"
    assert saved["stages"]["media"]["artifacts"] == {
        "flac": "audio.flac"
    }
    assert saved["artifacts"]["flac"]["path"] == "audio.flac"


def test_completed_stage_requires_explicit_force_to_restart(tmp_path: Path) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    manifest.complete("media")

    with pytest.raises(ManifestTransitionError):
        manifest.start("media")
    assert manifest.should_run("media", force=True) is True

    manifest.start("media", force=True)

    assert manifest.stage("media")["status"] == "running"
    assert manifest.stage("media")["attempts"] == 2


def test_blocked_stage_pauses_until_an_explicit_forced_rerun(tmp_path: Path) -> None:
    manifest = manifest_at(tmp_path)
    blocker_id = manifest.block(
        "review",
        code="manual_review",
        message="confirm speaker name",
        details={"candidate": "user-4"},
    )
    blocked_document = manifest.path.read_text(encoding="utf-8")

    assert manifest.should_run("review") is False
    assert RunManifest(manifest.path, run_dir=tmp_path).should_run("review") is False
    with pytest.raises(ManifestTransitionError, match="force"):
        manifest.start("review")
    assert manifest.path.read_text(encoding="utf-8") == blocked_document

    manifest.start("review", force=True)

    saved = json.loads(manifest.path.read_text(encoding="utf-8"))
    assert saved["stages"]["review"]["status"] == "running"
    assert saved["stages"]["review"]["blocker_ids"] == []
    blocker = next(item for item in saved["blockers"] if item["id"] == blocker_id)
    assert blocker["resolved_at"].endswith("Z")
    assert blocker["resolution"] == "forced_rerun"


def test_failed_and_interrupted_running_stages_can_resume(tmp_path: Path) -> None:
    path = tmp_path / ".work" / "manifest.json"
    manifest = Manifest(path, run_dir=tmp_path)
    manifest.start("frames")
    manifest.fail("frames", RuntimeError("rate limit after page 4"))

    assert manifest.should_run("frames") is True
    manifest.start("frames")
    assert manifest.stage("frames")["attempts"] == 2

    reloaded = RunManifest(path, run_dir=tmp_path)
    assert reloaded.should_run("frames") is True
    reloaded.start("frames")
    assert reloaded.stage("frames")["attempts"] == 3


def test_illegal_transitions_do_not_change_persisted_state(tmp_path: Path) -> None:
    manifest = manifest_at(tmp_path)
    original = manifest.path.read_text(encoding="utf-8")

    with pytest.raises(ManifestTransitionError):
        manifest.complete("media")
    with pytest.raises(ManifestTransitionError):
        manifest.fail("media", "not running")
    with pytest.raises(ValueError, match="stage name"):
        manifest.start("../media")

    assert manifest.path.read_text(encoding="utf-8") == original


def test_artifact_paths_are_relative_and_cannot_escape_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = manifest_at(run_dir)
    manifest.start("media")

    with pytest.raises(ManifestPathError):
        manifest.complete("media", artifacts={"flac": "../outside.flac"})
    with pytest.raises(ManifestPathError):
        manifest.complete(
            "media", artifacts={"flac": str(tmp_path / "outside.flac")}
        )

    inside = run_dir / ".work" / "segments.json"
    manifest.complete("media", artifacts={"segments": inside})
    assert manifest.stage("media")["artifacts"]["segments"] == (
        ".work/segments.json"
    )


def test_interrupted_atomic_write_preserves_old_manifest_and_cleans_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    manifest.complete("media")
    old_bytes = manifest.path.read_bytes()
    old_document = json.loads(old_bytes)

    def interrupted_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        raise OSError("simulated interruption before replace")

    monkeypatch.setattr("transcript_pipeline.manifest.os.replace", interrupted_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        manifest.start("transcription")

    assert manifest.path.read_bytes() == old_bytes
    assert json.loads(manifest.path.read_text(encoding="utf-8")) == old_document
    assert manifest.stage("transcription")["status"] == "pending"
    assert not list(manifest.path.parent.glob(f".{manifest.path.name}.*.tmp"))


def test_manifest_records_safe_provenance_and_redacts_failures(tmp_path: Path) -> None:
    secret = "fake-api-secret"
    config = PipelineConfig.from_overrides(title="Synthetic session")
    manifest = manifest_at(
        tmp_path,
        input_path="recording.mp4",
        input_hash="a" * 64,
        media_metadata={"duration_seconds": 300.5, "has_audio": True},
        effective_config=config,
        inferred_event={"start_utc": "2026-07-17T18:00:00Z"},
        tool_versions={"ffmpeg": "7.1", "whisperx_model": "large-v3"},
    )
    manifest.start("frames")
    manifest.fail(
        "frames",
        RuntimeError(
            f"GET /models failed with Authorization: Bearer {secret} after page 2"
        ),
    )
    manifest.block(
        "review",
        code="speaker_manual_review",
        message="read-only projection unavailable",
        details={"retryable": True},
    )
    manifest.record_manual_decision(
        "role-map-1", "Moderator role ID confirmed", actor="operator"
    )

    saved_text = manifest.path.read_text(encoding="utf-8")
    saved = json.loads(saved_text)

    assert secret not in saved_text
    assert "GET /models failed" in saved_text
    assert saved["input"] == {
        "path": str((tmp_path / "recording.mp4").resolve()),
        "sha256": "a" * 64,
        "media": {"duration_seconds": 300.5, "has_audio": True},
    }
    assert saved["effective_config"]["title"] == "Synthetic session"
    assert saved["inferred_event"]["start_utc"] == "2026-07-17T18:00:00Z"
    assert saved["tool_versions"]["ffmpeg"] == "7.1"
    assert saved["stages"]["frames"]["error"]["type"] == "RuntimeError"
    assert saved["stages"]["review"]["status"] == "blocked"
    assert saved["blockers"][0]["code"] == "speaker_manual_review"
    assert saved["manual_decisions"][0]["id"] == "role-map-1"


@pytest.mark.parametrize("sensitive_key", ["password", "credential", "client_secret"])
def test_manifest_failure_redacts_structured_exception_args_on_disk(
    tmp_path: Path, sensitive_key: str
) -> None:
    secret = f"manifest-unique-{sensitive_key}-secret-2741"
    manifest = manifest_at(tmp_path)
    manifest.start("media")

    manifest.fail(
        "media",
        RuntimeError({sensitive_key: secret, "context": "page 4"}),
    )

    raw_manifest = manifest.path.read_text(encoding="utf-8")
    saved_error = json.loads(raw_manifest)["stages"]["media"]["error"]
    assert secret not in raw_manifest
    assert saved_error["type"] == "RuntimeError"
    assert sensitive_key in saved_error["message"]
    assert REDACTED in saved_error["message"]
    assert "page 4" in saved_error["message"]
    assert json.loads(json.dumps(saved_error)) == saved_error


def test_complete_records_hashes_and_validation(tmp_path: Path) -> None:
    manifest = manifest_at(tmp_path)
    digest = "b" * 64
    manifest.start("media")

    manifest.complete(
        "media",
        artifacts={"flac": "audio.flac"},
        artifact_hashes={"flac": digest},
        validation={"flac": {"decodable": True}},
    )

    saved = json.loads(manifest.path.read_text(encoding="utf-8"))
    assert saved["stages"]["media"]["artifact_hashes"]["flac"] == digest
    assert saved["artifacts"]["flac"] == {
        "path": "audio.flac",
        "sha256": digest,
        "validation": {"decodable": True},
        "stage": "media",
    }


def test_forced_rerun_removes_only_artifacts_owned_by_that_stage(
    tmp_path: Path,
) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    manifest.complete(
        "media",
        artifacts={"old_flac": "old.flac", "media_shared": "media-shared.json"},
    )
    manifest.start("frames")
    manifest.complete(
        "frames",
        artifacts={"frames": ".work/frames.json", "shared": ".work/shared.json"},
    )

    manifest.start("media", force=True)

    assert manifest.stage("media")["artifacts"] == {}
    assert manifest.data["artifacts"] == {
        "frames": {"path": ".work/frames.json", "stage": "frames"},
        "shared": {"path": ".work/shared.json", "stage": "frames"},
    }

    manifest.fail("media", "transcoder unavailable")
    assert "old_flac" not in manifest.data["artifacts"]
    assert manifest.data["artifacts"]["shared"]["stage"] == "frames"


def test_forced_rerun_completion_with_fewer_artifacts_leaves_no_stale_records(
    tmp_path: Path,
) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    manifest.complete(
        "media",
        artifacts={"flac": "old.flac", "segments": ".work/old-segments.json"},
    )

    manifest.start("media", force=True)
    manifest.complete("media", artifacts={"flac": "new.flac"})

    assert manifest.stage("media")["artifacts"] == {"flac": "new.flac"}
    assert manifest.data["artifacts"] == {
        "flac": {"path": "new.flac", "stage": "media"}
    }


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "accessToken",
        "clientSecret",
        "apiKey",
        "credentials",
        "tokens",
        "secrets",
        "passwords",
        "apiKeys",
        "clientSecrets",
        "accessTokens",
    ],
)
def test_manifest_redacts_singular_and_plural_secrets_before_disk_persistence(
    tmp_path: Path, sensitive_key: str
) -> None:
    secret = f"manifest-{sensitive_key}-quality-gate-secret"

    manifest = manifest_at(
        tmp_path,
        tool_versions={sensitive_key: secret, "context": "page 4"},
    )
    raw_manifest = manifest.path.read_text(encoding="utf-8")
    saved = json.loads(raw_manifest)

    assert secret not in raw_manifest
    assert saved["tool_versions"][sensitive_key] == REDACTED
    assert saved["tool_versions"]["context"] == "page 4"


def _valid_manifest_document(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest = manifest_at(tmp_path)
    return manifest.path, json.loads(manifest.path.read_text(encoding="utf-8"))


def test_loading_an_empty_manifest_object_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / ".work" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="missing required"):
        Manifest(path, run_dir=tmp_path)


@pytest.mark.parametrize(
    "missing_field",
    ["schema_version", "input", "effective_config", "stages"],
)
def test_loading_manifest_missing_required_fields_is_rejected(
    tmp_path: Path, missing_field: str
) -> None:
    path, document = _valid_manifest_document(tmp_path)
    document.pop(missing_field)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="missing required"):
        Manifest(path, run_dir=tmp_path)


@pytest.mark.parametrize("schema_version", [0, 2, "1", True])
def test_loading_incompatible_manifest_schema_is_rejected(
    tmp_path: Path, schema_version: object
) -> None:
    path, document = _valid_manifest_document(tmp_path)
    document["schema_version"] = schema_version
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="schema_version"):
        Manifest(path, run_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("created_at", 17),
        ("updated_at", None),
        ("input", []),
        ("effective_config", []),
        ("inferred_event", []),
        ("tool_versions", []),
        ("stages", []),
        ("artifacts", []),
        ("blockers", {}),
        ("manual_decisions", {}),
    ],
)
def test_loading_malformed_manifest_field_types_is_rejected(
    tmp_path: Path, field: str, malformed: object
) -> None:
    path, document = _valid_manifest_document(tmp_path)
    document[field] = malformed
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match=field):
        Manifest(path, run_dir=tmp_path)


def test_loading_malformed_stage_field_types_is_rejected(tmp_path: Path) -> None:
    path, document = _valid_manifest_document(tmp_path)
    manifest = Manifest(path, run_dir=tmp_path)
    manifest.start("media")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["stages"]["media"]["attempts"] = "one"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="attempts"):
        Manifest(path, run_dir=tmp_path)


def test_loading_rejects_an_unresolved_orphan_blocker(tmp_path: Path) -> None:
    manifest = manifest_at(tmp_path)
    manifest.block("review", code="manual_review", message="confirm speaker")
    document = json.loads(manifest.path.read_text(encoding="utf-8"))
    document["stages"]["review"].update(
        status="running",
        attempts=1,
        started_at=document["updated_at"],
        finished_at=None,
        blocker_ids=[],
    )
    manifest.path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="unresolved blocker"):
        Manifest(manifest.path, run_dir=tmp_path)


def test_loading_preserves_unknown_extensible_fields(tmp_path: Path) -> None:
    path, document = _valid_manifest_document(tmp_path)
    document["future_provenance"] = {"schema": 2, "note": "preserve me"}
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = Manifest(path, run_dir=tmp_path)

    assert loaded.data["future_provenance"] == {
        "schema": 2,
        "note": "preserve me",
    }


def test_non_finite_write_fails_closed_without_changing_manifest(
    tmp_path: Path,
) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    previous = manifest.path.read_bytes()

    with pytest.raises(ValueError, match="non-finite"):
        manifest.complete(
            "media",
            validation={"probe": {"duration_seconds": math.inf}},
        )

    assert manifest.path.read_bytes() == previous
    assert manifest.stage("media")["status"] == "running"
    assert not list(manifest.path.parent.glob(f".{manifest.path.name}.*.tmp"))


def test_loading_non_finite_json_is_rejected(tmp_path: Path) -> None:
    path, document = _valid_manifest_document(tmp_path)
    document["input"]["media"]["duration_seconds"] = math.nan
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="non-finite"):
        Manifest(path, run_dir=tmp_path)


def test_interleaved_manifest_instances_do_not_lose_stage_updates(
    tmp_path: Path,
) -> None:
    first = manifest_at(tmp_path)
    second = Manifest(first.path, run_dir=tmp_path)

    first.start("media")
    second.start("frames")
    first.complete("media", artifacts={"flac": "audio.flac"})
    second.complete(
        "frames", artifacts={"messages": ".work/visual_evidence.json"}
    )

    saved = Manifest(first.path, run_dir=tmp_path).data
    assert saved["stages"]["media"]["status"] == "completed"
    assert saved["stages"]["frames"]["status"] == "completed"
    assert set(saved["artifacts"]) == {"flac", "messages"}
    assert first.path.with_name(f"{first.path.name}.lock").exists()


def test_external_absolute_input_path_is_allowed_but_artifacts_remain_contained(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "external" / "session.mp4"
    manifest = manifest_at(
        run_dir,
        input_path=source,
        input_hash="a" * 64,
        effective_config=PipelineConfig(),
    )

    assert manifest.data["input"]["path"] == str(source.resolve())
    manifest.start("media")
    with pytest.raises(ManifestPathError):
        manifest.complete("media", artifacts={"copy": source})


def test_existing_manifest_rejects_input_path_hash_and_config_mismatches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recordings" / "session.mp4"
    config = PipelineConfig()
    manifest = manifest_at(
        tmp_path,
        input_path=source,
        input_hash="a" * 64,
        effective_config=config,
    )

    normalized_equivalent = source.parent / ".." / "recordings" / source.name
    Manifest(
        manifest.path,
        run_dir=tmp_path,
        input_path=normalized_equivalent,
        input_hash="a" * 64,
        effective_config=config,
    )
    with pytest.raises(ManifestError, match="input path"):
        Manifest(
            manifest.path,
            run_dir=tmp_path,
            input_path=tmp_path / "different.mp4",
        )
    with pytest.raises(ManifestError, match="input hash"):
        Manifest(manifest.path, run_dir=tmp_path, input_hash="b" * 64)
    with pytest.raises(ManifestError, match="effective config"):
        Manifest(
            manifest.path,
            run_dir=tmp_path,
            effective_config=PipelineConfig(title="Different title"),
        )


def test_metadata_updates_merge_latest_state_and_redact_before_disk(
    tmp_path: Path,
) -> None:
    secret = "metadata-access-token-secret"
    first = manifest_at(tmp_path)
    second = Manifest(first.path, run_dir=tmp_path)

    first.update_metadata(
        media_metadata={"duration_seconds": 300.5, "has_audio": True},
        tool_versions={"ffmpeg": "7.1", "accessTokens": secret},
    )
    second.update_metadata(
        inferred_event={"start_utc": "2026-07-17T18:00:00Z"},
        model_versions={"whisperx": "large-v3"},
    )

    raw = first.path.read_text(encoding="utf-8")
    saved = json.loads(raw)
    assert secret not in raw
    assert saved["input"]["media"]["duration_seconds"] == 300.5
    assert saved["tool_versions"] == {
        "ffmpeg": "7.1",
        "accessTokens": REDACTED,
    }
    assert saved["inferred_event"]["start_utc"] == "2026-07-17T18:00:00Z"
    assert saved["model_versions"] == {"whisperx": "large-v3"}


def test_metadata_uri_and_structured_diagnostics_are_safe_on_disk(
    tmp_path: Path,
) -> None:
    uri_secret = "uri-password-quality-secret"
    diagnostic_secret = 'diagnostic-password#with"quote'
    manifest = manifest_at(tmp_path)

    manifest.update_metadata(
        tool_versions={
            "endpoint": (
                f"postgresql://operator:{uri_secret}@db.example.test/pipeline"
                f"?password={uri_secret}"
            ),
            "diagnostic": json.dumps(
                {"passwords": diagnostic_secret, "context": "page 4"}
            ),
        }
    )

    raw = manifest.path.read_text(encoding="utf-8")
    saved = json.loads(raw)["tool_versions"]
    assert uri_secret not in raw
    assert diagnostic_secret not in raw
    assert "postgresql://[REDACTED]@db.example.test/pipeline" in saved["endpoint"]
    assert json.loads(saved["diagnostic"]) == {
        "context": "page 4",
        "passwords": REDACTED,
    }


def test_cross_stage_artifact_name_overwrite_is_rejected(tmp_path: Path) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    manifest.complete("media", artifacts={"result": "audio.flac"})
    manifest.start("frames")
    previous = manifest.path.read_bytes()

    with pytest.raises(ManifestTransitionError, match="artifact.*result"):
        manifest.complete(
            "frames", artifacts={"result": ".work/visual_evidence.json"}
        )

    assert manifest.path.read_bytes() == previous
    assert manifest.stage("frames")["status"] == "running"


@pytest.mark.parametrize(
    "corruption",
    ["missing_global", "missing_stage", "wrong_owner", "wrong_path"],
)
def test_loading_rejects_inconsistent_artifact_ownership(
    tmp_path: Path, corruption: str
) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    manifest.complete("media", artifacts={"flac": "audio.flac"})
    document = json.loads(manifest.path.read_text(encoding="utf-8"))
    if corruption == "missing_global":
        del document["artifacts"]["flac"]
    elif corruption == "missing_stage":
        del document["stages"]["media"]["artifacts"]["flac"]
    elif corruption == "wrong_owner":
        document["artifacts"]["flac"]["stage"] = "frames"
    else:
        document["artifacts"]["flac"]["path"] = "different.flac"
    manifest.path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="artifact"):
        Manifest(manifest.path, run_dir=tmp_path)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("root_timestamp", "created_at"),
        ("completed_attempts", "attempts"),
        ("completed_started", "started_at"),
        ("completed_finished", "finished_at"),
        ("completed_error", "completed"),
        ("running_started", "started_at"),
        ("failed_error", "failed"),
        ("pending_attempts", "pending"),
    ],
)
def test_loading_rejects_invalid_timestamps_and_stage_lifecycle(
    tmp_path: Path, corruption: str, message: str
) -> None:
    manifest = manifest_at(tmp_path)
    manifest.start("media")
    manifest.complete("media")
    document = json.loads(manifest.path.read_text(encoding="utf-8"))
    stage = document["stages"]["media"]
    if corruption == "root_timestamp":
        document["created_at"] = "2026-07-17T18:00:00+02:00"
    elif corruption == "completed_attempts":
        stage["attempts"] = 0
    elif corruption == "completed_started":
        stage["started_at"] = None
    elif corruption == "completed_finished":
        stage["finished_at"] = None
    elif corruption == "completed_error":
        stage["error"] = {"type": "Error", "message": "stale"}
    elif corruption == "running_started":
        stage.update(status="running", started_at=None, finished_at=None)
    elif corruption == "failed_error":
        stage["status"] = "failed"
    else:
        stage.update(
            status="pending", attempts=1, started_at=None, finished_at=None
        )
    manifest.path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match=message):
        Manifest(manifest.path, run_dir=tmp_path)


def test_atomic_replace_fsyncs_the_containing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = manifest_at(tmp_path)
    real_open = os.open
    real_fsync = os.fsync
    directory_fds: set[int] = set()
    directory_fsyncs: list[int] = []

    def tracking_open(path: os.PathLike[str] | str, flags: int, *args: object) -> int:
        fd = real_open(path, flags, *args)
        if Path(path) == manifest.path.parent and flags & os.O_DIRECTORY:
            directory_fds.add(fd)
        return fd

    def tracking_fsync(fd: int) -> None:
        if fd in directory_fds:
            directory_fsyncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("transcript_pipeline.manifest.os.open", tracking_open)
    monkeypatch.setattr("transcript_pipeline.manifest.os.fsync", tracking_fsync)

    manifest.start("media")

    assert directory_fsyncs
