from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeStages, load_fixture, prepare_run, requires_ffmpeg, run_cli

pytestmark = pytest.mark.usefixtures("flac_check")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codes(report: Any) -> set[str]:
    return {issue.code for issue in report.issues}


def _evidence_run(tmp_path: Path, evidence_files: Any) -> Path:
    run_dir = prepare_run(tmp_path, FakeStages())
    speaker_path, qa_path = evidence_files()
    code, _payload, _ = run_cli(
        ["validate", "--run", str(run_dir), "--phase", "evidence",
         "--speaker-evidence", str(speaker_path), "--qa-evidence", str(qa_path)]
    )
    assert code == 0
    return run_dir


def _final_run(tmp_path: Path, evidence_files: Any) -> Path:
    run_dir = _evidence_run(tmp_path, evidence_files)
    code, _payload, _ = run_cli(["render", "--run", str(run_dir)])
    assert code == 0
    return run_dir


def test_evidence_phase_requires_both_documents(tmp_path: Path) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = prepare_run(tmp_path, FakeStages())

    report = validate_run(run_dir, phase="evidence")

    assert report.valid is False
    missing = {issue.path for issue in report.issues if issue.code == "required_evidence_missing"}
    assert missing == {".work/speaker_evidence.json", ".work/qa_evidence.json"}


def test_evidence_phase_accepts_complete_documents(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)

    report = validate_run(run_dir, phase="evidence")

    assert report.valid is True, report.to_dict()


def test_evidence_issues_are_reported_with_file_and_json_path(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    qa_path = run_dir / ".work" / "qa_evidence.json"
    qa = json.loads(qa_path.read_text())
    qa["topics"][0]["questions"][0]["asked_sources"][0]["quote"] = "how fast is it"
    qa_path.write_text(json.dumps(qa))

    report = validate_run(run_dir, phase="evidence")

    assert report.valid is False
    assert any(
        issue.code == "quote_not_found"
        and issue.path == ".work/qa_evidence.json:$.topics[0].questions[0].asked_sources[0].quote"
        for issue in report.issues
    )
    assert "hash_mismatch" in _codes(report)


def test_unregistered_visual_evidence_cannot_be_cited(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    (run_dir / ".work" / "visual_evidence.json").write_text(
        json.dumps({"sources": [{"frame_id": "frame_000003000", "timestamp_seconds": 3.0, "status": "ok"}]})
    )
    qa_path = run_dir / ".work" / "qa_evidence.json"
    qa = json.loads(qa_path.read_text())
    qa["topics"][1]["questions"][0]["asked_sources"].append({"visual_frame_id": "frame_000003000"})
    qa_path.write_text(json.dumps(qa))

    report = validate_run(run_dir, phase="evidence")

    assert {"source_unregistered", "unknown_source"} <= _codes(report)


def test_manifest_escape_hash_drift_and_secret_are_reported_without_echo(
    tmp_path: Path, evidence_files: Any
) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    manifest_path = run_dir / ".work" / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["artifacts"]["unsafe"] = {"path": "../escape.txt", "sha256": "b" * 64, "stage": "media"}
    manifest_path.write_text(json.dumps(document))
    secret = "known-super-secret-2817"
    (run_dir / ".work" / "operator.log").write_text(f"Authorization: Bearer {secret}\n")

    report = validate_run(run_dir, phase="evidence", known_secrets=(secret,))
    rendered = json.dumps(report.to_dict())

    assert report.valid is False
    assert {"manifest_invalid", "unsafe_path", "secret_detected"} <= _codes(report)
    assert secret not in rendered
    assert any(issue.path == ".work/operator.log" for issue in report.issues)


def test_changed_recording_is_detected(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    (tmp_path / "synthetic-session.mp4").write_bytes(b"a different recording")

    assert "hash_mismatch" in _codes(validate_run(run_dir, phase="evidence"))


def test_hf_token_from_the_environment_is_scanned_for_but_never_echoed(
    tmp_path: Path, evidence_files: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    token = "plain-value-without-any-prefix-5150"
    monkeypatch.setenv("HF_TOKEN", token)
    (run_dir / ".work" / "debug.txt").write_text(f"model download used {token}\n")

    report = validate_run(run_dir, phase="evidence")

    assert "secret_detected" in _codes(report)
    assert token not in json.dumps(report.to_dict())


@pytest.mark.parametrize(
    ("relative_name", "payload", "known"),
    [
        ("operator.cfg", b"API_TOKEN=fake-secret-cfg-4815", True),
        ("settings.toml", b'password = "fake-secret-toml-5926"', False),
        ("credentials", b"client_secret=fake-secret-extensionless-6037", False),
        ("trace.bin", b"\x00\x01Authorization: Bearer fake-secret-binary-7148\xff", False),
        ("service.log", b"hf_" + b"A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7", False),
        ("debug_fake-secret-filename-9360.log", b"ordinary diagnostics", True),
    ],
)
def test_secret_scan_covers_every_managed_file_without_echoing_values_or_paths(
    tmp_path: Path, evidence_files: Any, relative_name: str, payload: bytes, known: bool
) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    secrets = (
        "fake-secret-cfg-4815",
        "fake-secret-toml-5926",
        "fake-secret-extensionless-6037",
        "fake-secret-binary-7148",
        "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7",
        "fake-secret-filename-9360",
    )
    secret = next(value for value in secrets if value in relative_name or value.encode() in payload)
    (run_dir / relative_name).write_bytes(payload)

    report = validate_run(run_dir, phase="evidence", known_secrets=(secret,) if known else ())

    assert report.valid is False
    assert "secret_detected" in _codes(report)
    assert secret not in json.dumps(report.to_dict())


def test_secret_scan_does_not_treat_arbitrary_binary_as_a_secret(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    (run_dir / "noise.bin").write_bytes(bytes(range(256)) * 4)

    assert "secret_detected" not in _codes(validate_run(run_dir, phase="evidence"))


def test_secret_scan_detects_pattern_split_across_read_chunks(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    secret = "boundary-secret-must-not-escape-4139"
    prefix = b"Authorization: Bearer "
    with (run_dir / "large.bin").open("wb") as output:
        output.write(b"\x00" * (1024 * 1024 - len(prefix) + 3))
        output.write(prefix + secret.encode())
        output.seek(9 * 1024 * 1024)
        output.write(b"\x00")

    report = validate_run(run_dir, phase="evidence")

    assert "secret_detected" in _codes(report)
    assert secret not in json.dumps(report.to_dict())


def test_secret_scan_ignores_url_like_bytes_inside_known_media(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    (run_dir / "frame.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"https://random-user:random-pass@example.invalid"
    )

    assert "secret_detected" not in _codes(validate_run(run_dir, phase="evidence"))


@pytest.mark.parametrize(
    ("filename", "prefix", "credential"),
    [
        ("frame.png", b"\x89PNG\r\n\x1a\n", b"API_TOKEN=leaked-value"),
        ("extra.flac", b"fLaC", b"Authorization: Bearer leaked-value"),
        ("photo.jpg", b"\xff\xd8\xff", b"hf_ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"),
    ],
)
def test_secret_scan_still_detects_explicit_credentials_in_media_headers(
    tmp_path: Path, evidence_files: Any, filename: str, prefix: bytes, credential: bytes
) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    (run_dir / filename).write_bytes(prefix + b"\x00\x01" + credential)

    assert "secret_detected" in _codes(validate_run(run_dir, phase="evidence"))


def test_symlinks_inside_a_run_are_rejected(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    (run_dir / "outside-link").symlink_to(tmp_path)

    assert "unsafe_path" in _codes(validate_run(run_dir, phase="evidence"))


def test_final_phase_accepts_a_completed_run(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _final_run(tmp_path, evidence_files)

    report = validate_run(run_dir, phase="all")

    assert report.valid is True, report.to_dict()


def test_final_phase_requires_registered_outputs(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _final_run(tmp_path, evidence_files)
    (run_dir / "qa.md").unlink()
    (run_dir / "notes.md").write_text("extra file\n")

    report = validate_run(run_dir, phase="final")

    assert report.valid is False
    assert {"required_artifact_missing", "artifact_missing"} <= _codes(report)


def test_ready_review_is_rejected_while_a_blocker_exists(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.manifest import Manifest
    from transcript_pipeline.validate import validate_run

    run_dir = _final_run(tmp_path, evidence_files)
    manifest = Manifest(run_dir / ".work" / "manifest.json", run_dir=run_dir)
    manifest.start("event")
    manifest.block("event", code="ambiguous_event_time", message="needs a decision")

    report = validate_run(run_dir, phase="final")

    assert {"ready_with_blockers", "manifest_blocker"} <= _codes(report)


def test_not_ready_review_requires_a_blocker(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _final_run(tmp_path, evidence_files)
    review = run_dir / "review.md"
    lines = review.read_text().splitlines()
    lines[1] = "**Status:** NOT READY — edited by hand."
    review.write_text("\n".join(lines) + "\n")

    assert {"not_ready_without_blocker", "hash_mismatch"} <= _codes(validate_run(run_dir, phase="final"))


def test_final_phase_recomputes_review_items_from_sources(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _final_run(tmp_path, evidence_files)
    path = run_dir / ".work" / "speaker_evidence.json"
    speaker = load_fixture("speaker-evidence.json")
    speaker["speakers"][1].update(status="manual_review", display_name="Bo Sample", contradictions=["x"])
    path.write_text(json.dumps(speaker))

    report = validate_run(run_dir, phase="final")

    assert {"speaker_manual_review", "hash_mismatch"} <= _codes(report)


def test_event_window_must_cover_the_recording(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _final_run(tmp_path, evidence_files)
    manifest_path = run_dir / ".work" / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["inferred_event"] = {
        "status": "resolved",
        "start_utc": "2026-03-14T09:30:00Z",
        "end_utc": "2026-03-14T09:40:00Z",
        "local_date": "2026-03-15",
    }
    manifest_path.write_text(json.dumps(document))

    report = validate_run(run_dir, phase="final")

    assert {"event_duration_mismatch", "run_date_mismatch"} <= _codes(report)
    assert not next(issue for issue in report.issues if issue.code == "run_date_mismatch").blocking


@requires_ffmpeg
def test_corrupt_or_wrong_length_flac_is_rejected(tmp_path: Path, evidence_files: Any) -> None:
    from conftest import make_silent_flac
    from transcript_pipeline.validate import validate_run

    run_dir = _final_run(tmp_path, evidence_files)
    flac = run_dir / "audio.flac"
    make_silent_flac(flac, duration=20.0)
    assert {"duration_mismatch", "hash_mismatch"} <= _codes(validate_run(run_dir, phase="final"))

    flac.write_bytes(b"fLaC" + b"\x00" * 64)
    assert "flac_invalid" in _codes(validate_run(run_dir, phase="final"))


def test_missing_run_is_reported(tmp_path: Path) -> None:
    from transcript_pipeline.validate import validate_run

    report = validate_run(tmp_path / "nope", phase="final")

    assert report.valid is False
    assert _codes(report) == {"run_missing"}


def test_build_render_bundle_rejects_unregistered_transcript_edits(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import build_render_bundle

    run_dir = _evidence_run(tmp_path, evidence_files)
    (run_dir / ".work" / "transcript_edits.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "diarized_sha256": _sha(run_dir / ".work" / "diarized_segments.json"),
                "edits": [{"segment_id": "seg-000001", "finalized_text": "unreviewed"}],
            }
        )
    )

    with pytest.raises(ValueError, match="not registered"):
        build_render_bundle(run_dir)


def test_secret_scan_does_not_flag_hyphenated_identifiers_that_end_in_token(
    tmp_path: Path, evidence_files: Any
) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir = _evidence_run(tmp_path, evidence_files)
    (run_dir / "notes.json").write_text(json.dumps({"slide-token": "intro", "q-token": "q-mobile"}))

    assert "secret_detected" not in _codes(validate_run(run_dir, phase="evidence"))
