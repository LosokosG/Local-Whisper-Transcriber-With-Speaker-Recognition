from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeStages, load_fixture, prepare_run, run_cli, write_json

pytestmark = pytest.mark.usefixtures("flac_check")


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / ".work" / "manifest.json").read_text(encoding="utf-8"))


def _stage(run_dir: Path, name: str) -> str | None:
    return _manifest(run_dir)["stages"].get(name, {}).get("status")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(run_dir: Path) -> dict[str, Any]:
    code, payload, _ = run_cli(["status", "--run", str(run_dir)])
    assert code == 0 and payload is not None
    return payload


def _ingest(run_dir: Path, evidence_files: Any, speaker: Any = None, qa: Any = None, *extra: str) -> tuple[int, dict[str, Any] | None]:
    speaker_path, qa_path = evidence_files(speaker, qa)
    code, payload, _ = run_cli(
        [
            "validate", "--run", str(run_dir), "--phase", "evidence",
            "--speaker-evidence", str(speaker_path), "--qa-evidence", str(qa_path), *extra,
        ]
    )
    return code, payload


def _ready_run(tmp_path: Path, evidence_files: Any, **stage_options: Any) -> tuple[Path, FakeStages]:
    stages = FakeStages(**stage_options)
    run_dir = prepare_run(tmp_path, stages)
    code, payload = _ingest(run_dir, evidence_files)
    assert code == 0, payload
    return run_dir, stages


def test_help_is_lazy_and_exposes_exact_public_commands() -> None:
    script = """
import json, sys
from transcript_pipeline.cli import main
try:
    main(['--help'])
except SystemExit as error:
    code = error.code
heavy = ('torch', 'whisperx', 'pyannote', 'faster_whisper')
loaded = [name for name in sys.modules if name.split('.')[0] in heavy]
print(json.dumps({'code': code, 'loaded': loaded}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    payload = json.loads(completed.stdout.splitlines()[-1])
    help_text = "\n".join(completed.stdout.splitlines()[:-1])
    assert payload == {"code": 0, "loaded": []}
    for command in ("prepare", "infer-event", "extract-frames", "validate", "render", "status", "resume"):
        assert command in help_text
    assert "token" not in help_text.lower()


def test_usage_errors_have_stable_code_two() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "transcript_pipeline", "prepare"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parents[1],
    )
    assert completed.returncode == 2


@pytest.mark.parametrize("flag", ["--hf-token", "--token", "--api-key"])
def test_parser_rejects_credential_flags(flag: str) -> None:
    from transcript_pipeline.cli import build_parser

    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["prepare", "recording.mp4", flag, "value"])
    assert raised.value.code == 2


def test_prepare_names_run_by_embedded_local_date_and_records_provenance(tmp_path: Path) -> None:
    stages = FakeStages()
    run_dir = prepare_run(tmp_path, stages, "--timezone", "Pacific/Auckland")

    assert run_dir.name == "2026-03-14_synthetic-session"
    assert (run_dir / "audio.flac").is_file()
    manifest = _manifest(run_dir)
    assert manifest["effective_config"]["title"] == "Synthetic session"
    assert manifest["effective_config"]["timezone"] == "Pacific/Auckland"
    assert manifest["model_versions"]["asr_model"] == "large-v3-turbo"
    assert set(manifest["tool_versions"]) == {"ffmpeg", "ffprobe"}
    assert manifest["stages"]["media"]["status"] == "completed"
    assert manifest["artifacts"]["diarized_segments"]["sha256"] == _sha(
        run_dir / ".work" / "diarized_segments.json"
    )
    assert stages.calls == ["probe", "extract", "transcribe"]


def test_prepare_without_creation_time_uses_an_undated_hash_name(tmp_path: Path) -> None:
    run_dir = prepare_run(tmp_path, FakeStages(creation_time=None))

    assert run_dir.name.startswith("undated_")
    assert run_dir.name.endswith("_synthetic-session")


def test_missing_diarization_keeps_flac_and_raw_transcript_but_blocks(tmp_path: Path) -> None:
    diarized = load_fixture("diarized.json")
    diarized["diarization_available"] = False
    diarized["speaker_recognition"] = {"status": "blocked", "code": "hf_token_unavailable", "reason": "x"}
    for segment in diarized["segments"]:
        segment["speaker"] = None
    stages = FakeStages(diarized=diarized)
    recording = tmp_path / "synthetic-session.mp4"
    recording.write_bytes(b"synthetic")

    code, payload, _ = run_cli(
        ["prepare", str(recording), "--output-root", str(tmp_path / "out"), "--title", "Synthetic session"],
        stages.adapters(),
    )

    assert code == 3
    run_dir = Path(payload["run"])  # type: ignore[index]
    assert (run_dir / "audio.flac").is_file()
    assert (run_dir / ".work" / "diarized_segments.json").is_file()
    status = _status(run_dir)
    assert status["next_action"] == "manual:transcription"
    assert status["blockers"][0]["code"] == "hf_token_unavailable"


def test_status_is_light_and_points_at_evidence_after_prepare(tmp_path: Path) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())

    status = _status(run_dir)

    assert status["stages"] == {"media": "completed", "transcription": "completed"}
    assert status["next_action"] == "validate"


def test_status_rejects_nonexistent_run_without_creating_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing-run"

    code, _payload, stderr = run_cli(["status", "--run", str(missing)])

    assert code == 2
    assert json.loads(stderr)["error"] == "usage"
    assert not missing.exists()


def test_resume_skips_completed_prepare_and_waits_for_evidence(tmp_path: Path) -> None:
    stages = FakeStages()
    run_dir = prepare_run(tmp_path, stages)
    calls_before = list(stages.calls)

    code, payload, _ = run_cli(["resume", "--run", str(run_dir)], stages.adapters())

    assert code == 3
    assert payload["next_action"] == "validate"  # type: ignore[index]
    assert stages.calls == calls_before


def test_stage_failure_has_stable_code_four_and_no_message_leak(tmp_path: Path) -> None:
    stages = FakeStages()

    def broken(_audio: Path, _output: Path) -> dict[str, Any]:
        raise RuntimeError("CUDA out of memory at /home/someone/private/path token=abc")

    recording = tmp_path / "rec.mp4"
    recording.write_bytes(b"x")
    code, _payload, stderr = run_cli(
        ["prepare", str(recording), "--output-root", str(tmp_path / "out")],
        stages.adapters(transcribe=broken),
    )

    assert code == 4
    assert json.loads(stderr) == {"error": "stage_failed", "type": "RuntimeError"}
    run_dir = next((tmp_path / "out").iterdir())
    manifest = _manifest(run_dir)
    assert manifest["stages"]["transcription"]["status"] == "failed"
    assert "abc" not in json.dumps(manifest)


def test_infer_event_blocks_on_a_single_signal_and_accepts_an_explicit_override(tmp_path: Path) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())

    code, payload, _ = run_cli(["infer-event", "--run", str(run_dir)])
    assert code == 3
    assert payload["next_action"] == "manual:event"  # type: ignore[index]
    resolution = json.loads((run_dir / ".work" / "event_resolution.json").read_text())
    assert resolution["blocker"] == "ambiguous_event_time"

    code, _payload, _ = run_cli(
        ["infer-event", "--run", str(run_dir), "--force-stage", "--event-start", "2026-03-14T10:30:00+01:00"]
    )
    assert code == 0
    event = _manifest(run_dir)["inferred_event"]
    assert event["start_utc"] == "2026-03-14T09:30:00Z"
    assert event["resolution_basis"] == "authoritative_user_override"
    assert _status(run_dir)["blockers"] == []


def test_infer_event_accepts_corroborating_evidence_file(tmp_path: Path) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())
    evidence = write_json(
        tmp_path / "event.json",
        {
            "schema_version": 1,
            "recording_duration_seconds": 56.0,
            "signals": [
                {
                    "id": "media", "kind": "media_creation_time", "candidate_start_utc": "2026-03-14T09:30:00Z",
                    "source": "container", "detail": "creation_time tag", "strength": 0.9,
                    "independent_group": "container", "authoritative": False,
                },
                {
                    "id": "slide-clock", "kind": "visible_ui", "candidate_start_utc": "2026-03-14T09:30:40Z",
                    "source": "frame_000003000", "detail": "clock on the first slide", "strength": 0.8,
                    "independent_group": "screen", "authoritative": False,
                },
            ],
        },
    )

    code, _payload, _ = run_cli(["infer-event", "--run", str(run_dir), "--evidence", str(evidence)])

    assert code == 0
    assert _manifest(run_dir)["inferred_event"]["local_date"] == "2026-03-14"


def test_extract_frames_requires_video(tmp_path: Path) -> None:
    run_dir = prepare_run(tmp_path, FakeStages(has_video=False))

    code, _payload, stderr = run_cli(["extract-frames", "--run", str(run_dir)])

    assert code == 2
    assert "video" in json.loads(stderr)["message"]
    assert _stage(run_dir, "frames") is None


def _fake_frames(run_dir: Path) -> Any:
    def frames(_video: Path, _diarized: Any, events: Any, work: Path) -> dict[str, Any]:
        directory = work / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        return {
            "schema_version": 1,
            "duration_seconds": 56.0,
            "sources": [
                {
                    "frame_id": "frame_000003000",
                    "timestamp_seconds": 3.0,
                    "status": "ok",
                    "raw_path": str(directory / "frame_000003000.png"),
                    "reasons": ["turn_midpoint"],
                    "source_refs": [],
                    "priority": 85,
                }
            ],
            "contact_sheets": [],
            "presence_observations": [],
            "presence": {"intervals": [], "isolated": [], "departures": []},
            "errors": [],
            "events_seen": len(events),
        }

    (run_dir / ".work" / "frames").mkdir(parents=True, exist_ok=True)
    (run_dir / ".work" / "frames" / "frame_000003000.png").write_bytes(b"png")
    return frames


def test_extract_frames_registers_visual_evidence_and_binds_observations(tmp_path: Path) -> None:
    stages = FakeStages(has_video=True)
    run_dir = prepare_run(tmp_path, stages)
    adapters = stages.adapters(extract_frames=_fake_frames(run_dir))
    observation = {
        "id": "obs-1",
        "frame_id": "frame_000003000",
        "timestamp_seconds": 3.0,
        "region": "speakers",
        "evidence_type": "speaking_indicator",
        "confidence": 0.9,
        "exact_visible_name": "Ada Example",
    }

    code, _payload, _ = run_cli(["extract-frames", "--run", str(run_dir)], adapters)
    assert code == 0
    assert _stage(run_dir, "frames") == "completed"

    bad = write_json(tmp_path / "bad-observations.json", [{**observation, "timestamp_seconds": 9.0}])
    code, _payload, _ = run_cli(["extract-frames", "--run", str(run_dir), "--observations", str(bad)], adapters)
    assert code == 4
    assert _stage(run_dir, "frames") == "failed"

    good = write_json(tmp_path / "observations.json", {"observations": [observation]})
    code, _payload, _ = run_cli(["extract-frames", "--run", str(run_dir), "--observations", str(good)], adapters)
    assert code == 0
    visual = json.loads((run_dir / ".work" / "visual_evidence.json").read_text())
    assert visual["presence_observations"][0]["id"] == "obs-1"
    assert _manifest(run_dir)["artifacts"]["visual_evidence"]["sha256"] == _sha(
        run_dir / ".work" / "visual_evidence.json"
    )


@pytest.mark.parametrize("managed_relative", ("frames", "frames/ocr", "frames/contact_sheets"))
def test_extract_frames_rejects_symlinked_managed_tree_before_adapter_or_write(
    tmp_path: Path, managed_relative: str
) -> None:
    stages = FakeStages(has_video=True)
    run_dir = prepare_run(tmp_path, stages)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("untouched\n", encoding="utf-8")
    managed = run_dir / ".work" / managed_relative
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.symlink_to(external, target_is_directory=True)
    calls: list[str] = []

    def frames(*_args: Any) -> dict[str, Any]:
        calls.append("called")
        return {}

    code, _payload, _ = run_cli(["extract-frames", "--run", str(run_dir)], stages.adapters(extract_frames=frames))

    assert code == 4
    assert calls == []
    assert list(external.iterdir()) == [sentinel]


def test_default_frame_stage_compacts_subsegments_before_scheduling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames
    from transcript_pipeline.cli import _default_frame_stage

    captured: list[dict[str, object]] = []

    def select(duration: float, segments: Any, events: Any, periodic_interval: float) -> list[dict[str, object]]:
        assert (duration, events, periodic_interval) == (100, [], 60)
        captured.extend(segments)
        return [{"timestamp_seconds": 0.0}]

    monkeypatch.setattr(frames, "select_frame_targets", select)
    monkeypatch.setattr(frames, "extract_frames", lambda *_a, **_k: [{"status": "error", "raw_path": None}])
    monkeypatch.setattr(frames, "create_contact_sheets", lambda *_a, **_k: [])
    monkeypatch.setattr(frames, "build_visual_evidence", lambda **kwargs: {"duration_seconds": kwargs["duration_seconds"]})
    segments = [
        {"id": f"seg-{index:06d}", "start": index / 10, "end": (index + 1) / 10, "speaker": "SPEAKER_00"}
        for index in range(1_000)
    ]

    result = _default_frame_stage(tmp_path / "talk.mp4", {"duration_seconds": 100, "segments": segments}, [], tmp_path)

    assert result == {"duration_seconds": 100.0}
    assert len(captured) == 1
    assert captured[0]["source_refs"] == ["seg-000000", "seg-000999"]


def test_default_frame_stage_ocr_is_limited_to_representative_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames
    from transcript_pipeline.cli import _default_frame_stage

    targets = [
        {"timestamp_seconds": 10.0, "priority": 70, "reasons": ["turn_midpoint_minus_5s"]},
        {"timestamp_seconds": 15.0, "priority": 85, "reasons": ["turn_midpoint"]},
        {"timestamp_seconds": 20.0, "priority": 10, "reasons": ["periodic_baseline"]},
    ]
    ocr_inputs: list[str] = []
    regions_seen: list[Any] = []

    monkeypatch.setattr(frames, "select_frame_targets", lambda *_a, **_k: targets)
    monkeypatch.setattr(
        frames,
        "extract_frames",
        lambda *_a, **_k: [
            {**target, "status": "ok", "raw_path": str(tmp_path / f"frame-{index}.png")}
            for index, target in enumerate(targets)
        ],
    )

    def crop(path: str, *_args: Any, regions: Any = None, **_kwargs: Any) -> list[dict[str, str]]:
        ocr_inputs.append(path)
        regions_seen.append(regions)
        return [{"path": path}]

    monkeypatch.setattr(frames, "create_ocr_crops", crop)
    monkeypatch.setattr(frames, "run_tesseract_ocr", lambda crops: crops)
    monkeypatch.setattr(frames, "create_contact_sheets", lambda *_a, **_k: [])
    monkeypatch.setattr(frames, "build_visual_evidence", lambda **kwargs: {"records": kwargs["frame_records"]})

    result = _default_frame_stage(
        tmp_path / "talk.mp4", {"duration_seconds": 30, "segments": []}, [], tmp_path, {"slides": (0, 0, 1, 1)}
    )

    assert [Path(path).name for path in ocr_inputs] == ["frame-1.png", "frame-2.png"]
    assert regions_seen == [{"slides": (0, 0, 1, 1)}] * 2
    assert "ocr" not in result["records"][0]
    assert "ocr" in result["records"][1]


def test_valid_evidence_completes_the_stage_with_source_hashes(tmp_path: Path, evidence_files: Any) -> None:
    run_dir, _stages = _ready_run(tmp_path, evidence_files)

    manifest = _manifest(run_dir)
    evidence = manifest["stages"]["evidence"]
    assert evidence["status"] == "completed"
    assert evidence["validation"]["evidence"]["source_hashes"] == {
        "diarized_segments": _sha(run_dir / ".work" / "diarized_segments.json")
    }
    assert manifest["artifacts"]["qa_evidence"]["stage"] == "evidence"
    assert _status(run_dir)["next_action"] == "render"


def test_invalid_evidence_is_rejected_without_touching_reviewed_files(tmp_path: Path, evidence_files: Any) -> None:
    run_dir, _stages = _ready_run(tmp_path, evidence_files)
    before = {name: _sha(run_dir / ".work" / name) for name in ("speaker_evidence.json", "qa_evidence.json")}
    qa = load_fixture("qa-evidence.json")
    qa["topics"][0]["questions"][0]["claims"][0]["sources"][0]["quote"] = "about 45 minutes"
    speaker = load_fixture("speaker-evidence.json")
    speaker["speakers"][1]["display_name"] = "Changed"

    code, payload = _ingest(run_dir, evidence_files, speaker, qa)

    assert code == 5
    assert payload["valid"] is False  # type: ignore[index]
    message = payload["issues"][0]["message"]  # type: ignore[index]
    assert "quote_not_found" in message and "unidentified" in message
    assert {name: _sha(run_dir / ".work" / name) for name in before} == before
    assert _stage(run_dir, "evidence") == "completed"


def test_evidence_ingest_needs_both_lanes(tmp_path: Path, evidence_files: Any) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())
    _speaker_path, qa_path = evidence_files()

    code, payload, _ = run_cli(["validate", "--run", str(run_dir), "--phase", "evidence", "--qa-evidence", str(qa_path)])

    assert code == 5
    assert "both speaker and Q&A evidence" in payload["issues"][0]["message"]  # type: ignore[index]
    assert not (run_dir / ".work" / "qa_evidence.json").exists()


def test_evidence_ingest_rejects_symlinked_input(tmp_path: Path, evidence_files: Any) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())
    speaker_path, qa_path = evidence_files()
    link = tmp_path / "linked-qa.json"
    link.symlink_to(qa_path)

    code, _payload, _ = run_cli(
        ["validate", "--run", str(run_dir), "--phase", "evidence",
         "--speaker-evidence", str(speaker_path), "--qa-evidence", str(link)]
    )

    assert code == 5
    assert not (run_dir / ".work" / "qa_evidence.json").exists()


def test_render_refuses_unvalidated_evidence(tmp_path: Path, evidence_files: Any) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())
    speaker_path, qa_path = evidence_files()
    (run_dir / ".work" / "speaker_evidence.json").write_bytes(speaker_path.read_bytes())
    (run_dir / ".work" / "qa_evidence.json").write_bytes(qa_path.read_bytes())

    code, payload, _ = run_cli(["render", "--run", str(run_dir)])

    assert code == 3
    assert payload["next_action"] == "validate"  # type: ignore[index]
    assert _status(run_dir)["blockers"][0]["code"] == "evidence_not_validated"
    assert not (run_dir / "transcript.md").exists()

    # Validating the evidence clears the way; the render blocker is downstream.
    speaker_path, qa_path = evidence_files()
    code, _payload, _ = run_cli(
        ["validate", "--run", str(run_dir), "--phase", "evidence",
         "--speaker-evidence", str(speaker_path), "--qa-evidence", str(qa_path)]
    )
    assert code == 0
    assert run_cli(["render", "--run", str(run_dir)])[0] == 0


def test_full_run_renders_validates_and_finalizes(tmp_path: Path, evidence_files: Any) -> None:
    run_dir, _stages = _ready_run(tmp_path, evidence_files)

    code, payload, _ = run_cli(["render", "--run", str(run_dir)])

    assert code == 0, payload
    assert payload["next_action"] == "complete"  # type: ignore[index]
    manifest = _manifest(run_dir)
    assert manifest["stages"]["review"]["status"] == "completed"
    for name in ("transcript", "qa", "review"):
        assert manifest["artifacts"][name]["sha256"] == _sha(run_dir / f"{name}.md")
    review = (run_dir / "review.md").read_text(encoding="utf-8")
    assert review.splitlines()[1].startswith("**Status:** READY")
    assert "about 40 minutes" in (run_dir / "qa.md").read_text(encoding="utf-8")

    code, report, _ = run_cli(["validate", "--run", str(run_dir), "--phase", "final"])
    assert code == 0, report

    # A second render with unchanged inputs is a no-op.
    before = _sha(run_dir / "review.md")
    code, _payload, _ = run_cli(["render", "--run", str(run_dir)])
    assert code == 0
    assert _sha(run_dir / "review.md") == before


def test_manual_review_speaker_blocks_then_resume_finalizes_after_a_decision(
    tmp_path: Path, evidence_files: Any
) -> None:
    stages = FakeStages()
    run_dir = prepare_run(tmp_path, stages)
    speaker = load_fixture("speaker-evidence.json")
    speaker["speakers"][1].update(
        status="manual_review", display_name="Bo Sample", contradictions=["a slide shows a different name"]
    )
    assert _ingest(run_dir, evidence_files, speaker)[0] == 0

    code, payload, _ = run_cli(["render", "--run", str(run_dir)])

    assert code == 3
    assert payload["next_action"] == "resolve_review"  # type: ignore[index]
    review = (run_dir / "review.md").read_text(encoding="utf-8")
    assert review.splitlines()[1].startswith("**Status:** NOT READY")
    assert "Bo Sample" not in (run_dir / "transcript.md").read_text(encoding="utf-8")
    code, report, _ = run_cli(["validate", "--run", str(run_dir), "--phase", "final"])
    assert code == 5
    assert "manifest_blocker" in {issue["code"] for issue in report["issues"]}  # type: ignore[index]

    # The operator decides the speaker stays unidentified.
    assert _ingest(run_dir, evidence_files, load_fixture("speaker-evidence.json"))[0] == 0
    code, payload, _ = run_cli(["resume", "--run", str(run_dir)], stages.adapters())
    assert code == 0, payload
    assert _status(run_dir)["next_action"] == "complete"
    assert (run_dir / "review.md").read_text(encoding="utf-8").splitlines()[1].startswith("**Status:** READY")


def test_validator_exception_demotes_review_and_blocks(tmp_path: Path, evidence_files: Any) -> None:
    from transcript_pipeline.validate import validate_run

    run_dir, stages = _ready_run(tmp_path, evidence_files)

    def flaky(directory: Path, phase: str) -> Any:
        if phase == "final":
            raise RuntimeError("validator crashed")
        return validate_run(directory, phase=phase)  # type: ignore[arg-type]

    code, _payload, _ = run_cli(["render", "--run", str(run_dir)], stages.adapters(validate=flaky))

    assert code == 3
    assert _stage(run_dir, "review") == "blocked"
    review = (run_dir / "review.md").read_text(encoding="utf-8")
    assert review.splitlines()[1].startswith("**Status:** NOT READY")
    assert "validator_exception" in review
    assert _manifest(run_dir)["artifacts"]["review"]["sha256"] == _sha(run_dir / "review.md")


def test_transcript_edits_overlay_public_text_and_keep_canonical_source(tmp_path: Path, evidence_files: Any) -> None:
    run_dir, _stages = _ready_run(tmp_path, evidence_files)
    canonical = run_dir / ".work" / "diarized_segments.json"
    canonical_hash = _sha(canonical)
    edits = write_json(
        tmp_path / "edits.json",
        {
            "schema_version": 2,
            "diarized_sha256": canonical_hash,
            "edits": [{"segment_id": "seg-000004", "finalized_text": "For a ten gigabyte archive it usually finishes in about 40 minutes on one worker. [figure confirmed]"}],
        },
    )

    code, _payload, _ = run_cli(
        ["validate", "--run", str(run_dir), "--phase", "evidence", "--transcript-edits", str(edits)]
    )
    assert code == 0
    code, _payload, _ = run_cli(["render", "--run", str(run_dir)])

    assert code == 0
    assert "[figure confirmed]" in (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert _sha(canonical) == canonical_hash


def test_transcript_edits_for_unknown_segments_are_rejected(tmp_path: Path, evidence_files: Any) -> None:
    run_dir, _stages = _ready_run(tmp_path, evidence_files)
    edits = write_json(
        tmp_path / "edits.json",
        {
            "schema_version": 2,
            "diarized_sha256": _sha(run_dir / ".work" / "diarized_segments.json"),
            "edits": [{"segment_id": "seg-000099", "finalized_text": "invented"}],
        },
    )

    code, payload, _ = run_cli(
        ["validate", "--run", str(run_dir), "--phase", "evidence", "--transcript-edits", str(edits)]
    )

    assert code == 5
    assert "unknown segment" in payload["issues"][0]["message"]  # type: ignore[index]
    assert not (run_dir / ".work" / "transcript_edits.json").exists()


def test_changed_transcript_requires_explicit_reingest_of_both_lanes(tmp_path: Path, evidence_files: Any) -> None:
    run_dir, stages = _ready_run(tmp_path, evidence_files)
    changed = copy.deepcopy(stages.diarized)
    changed["segments"][7]["text"] = "Thanks everyone, see you next month."
    stages.diarized = changed
    recording = tmp_path / "synthetic-session.mp4"
    code, _payload, _ = run_cli(
        ["prepare", str(recording), "--output-root", str(tmp_path / "outputs"), "--title", "Synthetic session", "--force-stage"],
        stages.adapters(),
    )
    assert code == 0

    code, payload, _ = run_cli(["validate", "--run", str(run_dir), "--phase", "evidence"])
    assert code == 5
    assert payload["issues"][0]["code"] == "reviewed_evidence_stale"  # type: ignore[index]
    assert _status(run_dir)["next_action"] == "manual:evidence"

    code, _payload = _ingest(run_dir, evidence_files)
    assert code == 0
    assert _status(run_dir)["next_action"] == "render"


def test_hand_edited_evidence_after_render_forces_revalidation(tmp_path: Path, evidence_files: Any) -> None:
    run_dir, _stages = _ready_run(tmp_path, evidence_files)
    assert run_cli(["render", "--run", str(run_dir)])[0] == 0

    qa_path = run_dir / ".work" / "qa_evidence.json"
    qa = json.loads(qa_path.read_text())
    qa["topics"][0]["title"] = "Import pipeline"
    qa_path.write_text(json.dumps(qa))

    assert _status(run_dir)["next_action"] == "validate"
    code, _payload, _ = run_cli(["render", "--run", str(run_dir)])
    assert code == 3
    assert _stage(run_dir, "review") == "blocked"
    assert "Import pipeline" not in (run_dir / "qa.md").read_text(encoding="utf-8")
    code, report, _ = run_cli(["validate", "--run", str(run_dir), "--phase", "final"])
    assert code == 5
    assert "ready_with_blockers" in {issue["code"] for issue in report["issues"]}  # type: ignore[index]

    _speaker_path, new_qa = evidence_files(None, qa)
    code, _payload, _ = run_cli(
        ["validate", "--run", str(run_dir), "--phase", "evidence", "--speaker-evidence", str(_speaker_path), "--qa-evidence", str(new_qa)]
    )
    assert code == 0
    assert _status(run_dir)["next_action"] == "render"
    assert run_cli(["render", "--run", str(run_dir)])[0] == 0
    assert "## Import pipeline" in (run_dir / "qa.md").read_text(encoding="utf-8")


def test_resume_drives_a_ready_run_to_completion_without_repeating_prepare(
    tmp_path: Path, evidence_files: Any
) -> None:
    run_dir, stages = _ready_run(tmp_path, evidence_files)
    calls = list(stages.calls)

    code, payload, _ = run_cli(["resume", "--run", str(run_dir)], stages.adapters())

    assert code == 0, payload
    assert stages.calls == calls
    assert _status(run_dir)["next_action"] == "complete"


def test_operator_can_confirm_a_speaker_name_and_the_decision_is_audited(
    tmp_path: Path, evidence_files: Any
) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())
    speaker = load_fixture("speaker-evidence.json")
    speaker["speakers"][1].update(
        status="identified",
        confidence="high",
        display_name="Bo Sample",
        output_label="Bo Sample",
        binding_evidence=[{"type": "operator_confirmation", "sources": [{"segment_id": "seg-000003"}]}],
    )

    code, payload = _ingest(run_dir, evidence_files, speaker)
    assert code == 5
    assert "binding_not_definitive" in payload["issues"][0]["message"]  # type: ignore[index]

    code, payload = _ingest(run_dir, evidence_files, speaker, None, "--confirm-speaker", "SPEAKER_01", "Bo Sample")
    assert code == 0, payload
    decisions = _manifest(run_dir)["manual_decisions"]
    assert decisions[-1]["details"] == {"speaker_label": "SPEAKER_01", "display_name": "Bo Sample"}

    assert run_cli(["render", "--run", str(run_dir)])[0] == 0
    assert "**[00:00:12] Bo Sample:**" in (run_dir / "transcript.md").read_text(encoding="utf-8")
    review = (run_dir / "review.md").read_text(encoding="utf-8")
    assert "## Operator decisions" in review and "Bo Sample" in review


def test_confirm_speaker_rejects_malformed_labels(tmp_path: Path, evidence_files: Any) -> None:
    run_dir = prepare_run(tmp_path, FakeStages())

    code, _payload = _ingest(run_dir, evidence_files, None, None, "--confirm-speaker", "guest", "Bo Sample")

    assert code == 2
    assert _manifest(run_dir)["manual_decisions"] == []


def test_render_recovers_a_blocked_review_once_evidence_is_valid_again(
    tmp_path: Path, evidence_files: Any
) -> None:
    run_dir, _stages = _ready_run(tmp_path, evidence_files)
    assert run_cli(["render", "--run", str(run_dir)])[0] == 0
    qa_path = run_dir / ".work" / "qa_evidence.json"
    original = qa_path.read_bytes()
    qa_path.write_bytes(original.replace(b"Roadmap", b"Plans"))
    assert run_cli(["render", "--run", str(run_dir)])[0] == 3
    assert _stage(run_dir, "review") == "blocked"

    # Restore the reviewed file byte-for-byte and validate it explicitly.
    qa_path.write_bytes(original)
    assert _ingest(run_dir, evidence_files)[0] == 0
    code, payload, _ = run_cli(["render", "--run", str(run_dir)])

    assert code == 0, payload
    assert _stage(run_dir, "review") == "completed"
