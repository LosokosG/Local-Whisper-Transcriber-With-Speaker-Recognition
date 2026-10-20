"""Resumable stdlib CLI for the recording-to-reviewed-transcript pipeline.

Exit codes are part of the public interface: 0 success, 2 usage, 3 a manual
blocker, 4 stage failure, and 5 validation failure.

Stages and their artifacts (all paths relative to the run directory):

    media          audio.flac                       (prepare)
    transcription  .work/diarized_segments.json     (prepare)
    event          .work/event_resolution.json      (infer-event, optional)
    frames         .work/visual_evidence.json       (extract-frames, optional)
    evidence       .work/speaker_evidence.json,     (validate --phase evidence)
                   .work/qa_evidence.json
    render         transcript.md, qa.md, review.md  (render)
    review         final validation of the outputs  (render)

Every completed stage records SHA-256 hashes of what it produced and, where
relevant, of what it consumed. A later stage refuses to trust an earlier
artifact whose hash no longer matches, so editing a file by hand forces the
affected stages to run again instead of silently mixing old and new data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

SUCCESS_EXIT = 0
USAGE_EXIT = 2
BLOCKED_EXIT = 3
STAGE_FAILURE_EXIT = 4
VALIDATION_FAILURE_EXIT = 5

PROG = "transcript-pipeline"
FLAC_NAME = "audio.flac"
_STAGE_ORDER = (
    "media",
    "transcription",
    "event",
    "frames",
    "evidence",
    "render",
    "review",
)
_OPTIONAL_STAGES = frozenset({"event", "frames"})
_STAGE_COMMANDS = {
    "media": "prepare",
    "transcription": "prepare",
    "event": "infer-event",
    "frames": "extract-frames",
    "evidence": "validate",
    "render": "render",
    "review": "render",
}
_PUBLIC_OUTPUTS = {
    "transcript.md": "transcript",
    "qa.md": "qa",
    "review.md": "review",
}
_DIARIZED = ("transcription", "diarized_segments", ".work/diarized_segments.json")
_VISUAL = ("frames", "visual_evidence", ".work/visual_evidence.json")
_RENDER_FILE_INPUTS = {
    "diarized_segments": _DIARIZED,
    "speaker_evidence": ("evidence", "speaker_evidence", ".work/speaker_evidence.json"),
    "qa_evidence": ("evidence", "qa_evidence", ".work/qa_evidence.json"),
    "transcript_edits": ("evidence", "transcript_edits", ".work/transcript_edits.json"),
    "visual_evidence": _VISUAL,
}
_RENDER_OPTIONAL_INPUTS = frozenset({"transcript_edits", "visual_evidence"})
_EVIDENCE_LANES = ("speaker_evidence", "qa_evidence")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(USAGE_EXIT, f"{self.prog}: error: invalid arguments\n")


@dataclass(slots=True)
class StageAdapters:
    """Optional light-weight stage seams used by tests and offline callers."""

    probe_media: Callable[[Path], Mapping[str, Any]] | None = None
    extract_media: Callable[[Path, Path], Mapping[str, Any]] | None = None
    transcribe: Callable[[Path, Path], Mapping[str, Any]] | None = None
    extract_frames: Callable[
        [Path, Mapping[str, Any], Sequence[Mapping[str, Any]], Path],
        Mapping[str, Any],
    ] | None = None
    render: Callable[[Path, Mapping[str, Any]], Mapping[str, Path]] | None = None
    validate: Callable[[Path, str], Any] | None = None


class _UsageProblem(ValueError):
    """A precondition failure whose message is safe to print verbatim."""


class _EvidenceIngestValidationError(ValueError):
    """A candidate reviewed bundle is invalid and must not be committed."""


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    _write_atomic(
        path,
        json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(
            source,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")
            ),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_value_hash(value: Any) -> str:
    serialized = json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _stage_artifact_current(
    run_dir: Path,
    manifest: Any,
    *,
    stage: str,
    name: str,
    relative_path: str,
) -> bool:
    """Return whether one completed stage still owns its exact hashed artifact."""

    stage_record = manifest.stage(stage)
    if stage_record.get("status") != "completed":
        return False
    expected = stage_record.get("artifact_hashes", {}).get(name)
    if stage_record.get("artifacts", {}).get(name) != relative_path or not isinstance(expected, str):
        return False
    owned = manifest.data.get("artifacts", {}).get(name)
    if (
        not isinstance(owned, Mapping)
        or owned.get("stage") != stage
        or owned.get("path") != relative_path
        or owned.get("sha256") != expected
    ):
        return False
    path = run_dir / relative_path
    try:
        return path.is_file() and _sha256(path) == expected
    except OSError:
        return False


def _render_input_hashes(run_dir: Path, manifest: Any) -> dict[str, str] | None:
    """Snapshot every manifest-bound value consumed by the renderer.

    Optional inputs (transcript edits, visual evidence) may be absent, but an
    optional file that exists without a current hash binding makes the whole
    snapshot invalid: it must never silently drop out of a render.
    """

    result: dict[str, str] = {}
    for dependency, (stage, artifact, relative_path) in _RENDER_FILE_INPUTS.items():
        if _stage_artifact_current(
            run_dir, manifest, stage=stage, name=artifact, relative_path=relative_path
        ):
            result[dependency] = _sha256(run_dir / relative_path)
            continue
        if dependency not in _RENDER_OPTIONAL_INPUTS:
            return None
        if dependency == "transcript_edits" and (run_dir / relative_path).exists():
            return None
        result[dependency] = "unavailable"
    data = manifest.data
    result["inferred_event"] = _stable_value_hash(data.get("inferred_event", {}))
    result["effective_config"] = _stable_value_hash(data.get("effective_config", {}))
    return result


def _render_inputs_current(run_dir: Path, manifest: Any) -> bool:
    current = _render_input_hashes(run_dir, manifest)
    recorded = manifest.stage("render").get("validation", {}).get("input_hashes")
    return current is not None and isinstance(recorded, Mapping) and dict(recorded) == current


def _render_outputs_current(run_dir: Path, manifest: Any) -> bool:
    render = manifest.stage("render")
    if render.get("status") != "completed":
        return False
    paths = render.get("artifacts", {})
    if not isinstance(paths, Mapping) or not set(_PUBLIC_OUTPUTS.values()) <= set(paths):
        return False
    return all(
        _stage_artifact_current(
            run_dir, manifest, stage="render", name=name, relative_path=str(paths[name])
        )
        for name in _PUBLIC_OUTPUTS.values()
    )


def _render_refresh_action(run_dir: Path, manifest: Any) -> str | None:
    """Return the first deterministic repair required by a completed render."""

    if manifest.stage("render").get("status") != "completed":
        return None
    if not _evidence_current(run_dir, manifest):
        return "validate"
    if not _render_outputs_current(run_dir, manifest) or not _render_inputs_current(
        run_dir, manifest
    ):
        return "render"
    return None


def _probe(path: Path, adapters: StageAdapters) -> Mapping[str, Any]:
    if adapters.probe_media is not None:
        return adapters.probe_media(path)
    from .media import probe_media

    return probe_media(path)


def _extract(source: Path, destination: Path, adapters: StageAdapters) -> Mapping[str, Any]:
    if adapters.extract_media is not None:
        return adapters.extract_media(source, destination)
    from .media import extract_canonical_flac

    return extract_canonical_flac(source, destination)


def _transcribe(
    audio: Path,
    output: Path,
    effective_config: Mapping[str, Any],
    adapters: StageAdapters,
) -> Mapping[str, Any]:
    if adapters.transcribe is not None:
        return adapters.transcribe(audio, output)
    from .config import PipelineConfig
    from .transcription import transcribe_to_file

    settings = PipelineConfig.from_mapping(effective_config).transcription_settings()
    return transcribe_to_file(audio, output, settings)


def _render(run_dir: Path, bundle: Mapping[str, Any], adapters: StageAdapters) -> Mapping[str, Path]:
    if adapters.render is not None:
        return adapters.render(run_dir, bundle)
    from .render import render_run_artifacts

    return render_run_artifacts(run_dir, bundle)


def _validate(run_dir: Path, phase: str, adapters: StageAdapters) -> Any:
    if adapters.validate is not None:
        return adapters.validate(run_dir, phase)
    from .validate import validate_run

    return validate_run(run_dir, phase=phase)  # type: ignore[arg-type]


def _tool_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    first = (completed.stdout or completed.stderr).splitlines()
    return first[0][:160] if first else "unavailable"


def _parse_iso(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _run_name(metadata: Mapping[str, Any], config: Any, digest: str) -> str:
    creation = metadata.get("creation_time")
    if isinstance(creation, str):
        try:
            local_date = _parse_iso(creation).astimezone(ZoneInfo(config.timezone)).date()
            return f"{local_date.isoformat()}_{config.slug}"
        except (ValueError, TypeError):
            pass
    return f"undated_{digest[:8]}_{config.slug}"


def _default_title(recording: Path) -> str:
    title = " ".join(re.sub(r"[_\-.]+", " ", recording.stem).split())
    return title[:200] or "Recorded session"


def _config_from_args(args: argparse.Namespace) -> Any:
    from .config import PipelineConfig

    overrides: dict[str, Any] = {}
    for argument, field in (
        ("title", "title"),
        ("timezone", "timezone"),
        ("model", "model_name"),
        ("language", "language"),
        ("min_speakers", "min_speakers"),
        ("max_speakers", "max_speakers"),
        ("num_speakers", "num_speakers"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field] = value
    if "title" not in overrides:
        overrides["title"] = _default_title(args.recording)
    return PipelineConfig.from_overrides(**overrides)


def _manifest(run_dir: Path) -> Any:
    from .manifest import Manifest

    return Manifest(run_dir / ".work" / "manifest.json", run_dir=run_dir)


def _existing_run_manifest(run_dir: Path) -> Any:
    """Open an existing run without following mutable `.work` symlinks."""

    if not run_dir.is_dir():
        raise _UsageProblem("run directory does not exist")
    work = run_dir / ".work"
    if work.is_symlink() or not work.is_dir():
        raise _UsageProblem("run .work must be a real contained directory")
    if work.resolve(strict=True) != run_dir / ".work":
        raise _UsageProblem("run .work resolves outside its expected location")
    manifest_path = work / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise _UsageProblem("run manifest does not exist")
    return _manifest(run_dir)


def _block(manifest: Any, stage: str, code: str, message: str) -> None:
    status = manifest.stage(stage)["status"]
    if status == "blocked":
        return
    if status == "completed":
        manifest.start(stage, force=True)
    manifest.block(stage, code=code, message=message)


def _prepare_run(
    recording: Path,
    run_dir: Path,
    effective_config: Mapping[str, Any],
    adapters: StageAdapters,
    *,
    metadata: Mapping[str, Any] | None = None,
    force: bool = False,
) -> bool:
    from .manifest import Manifest

    if not recording.is_file():
        raise _UsageProblem("recording does not exist")
    source_hash = _sha256(recording)
    probed = dict(metadata or _probe(recording, adapters))
    if not probed.get("has_audio"):
        raise _UsageProblem("recording has no audio stream")
    (run_dir / ".work").mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        run_dir / ".work" / "manifest.json",
        run_dir=run_dir,
        input_path=recording,
        input_hash=source_hash,
        media_metadata=probed,
        effective_config=effective_config,
    )
    flac = run_dir / FLAC_NAME
    if manifest.should_run("media", force=force):
        manifest.start("media", force=force)
        try:
            artifact = dict(_extract(recording, flac, adapters))
            manifest.update_metadata(
                tool_versions={
                    "ffmpeg": _tool_version("ffmpeg"),
                    "ffprobe": _tool_version("ffprobe"),
                },
            )
            manifest.complete(
                "media",
                artifacts={"flac": flac.name},
                artifact_hashes={"flac": _sha256(flac)},
                validation={
                    "flac": {
                        "decodable": True,
                        "duration_difference_seconds": artifact.get(
                            "duration_difference_seconds"
                        ),
                    }
                },
            )
        except BaseException as error:
            manifest.fail("media", error)
            raise
    if not flac.is_file():
        raise FileNotFoundError("completed media stage has no canonical FLAC")

    transcription_path = run_dir / ".work" / "diarized_segments.json"
    if manifest.should_run("transcription", force=force):
        manifest.start("transcription", force=force)
        try:
            document = dict(
                _transcribe(flac, transcription_path, manifest.data["effective_config"], adapters)
            )
            if not transcription_path.is_file():
                _atomic_json(transcription_path, document)
            model = document.get("model")
            if isinstance(model, Mapping) and isinstance(model.get("name"), str):
                manifest.update_metadata(
                    model_versions={
                        "asr_model": model["name"],
                        "compute_type": str(model.get("compute_type")),
                        "device": str(model.get("device")),
                    }
                )
            if document.get("diarization_available") is not True:
                status = document.get("speaker_recognition")
                code = status.get("code") if isinstance(status, Mapping) else None
                _block(
                    manifest,
                    "transcription",
                    str(code or "diarization_unavailable"),
                    "speaker diarization requires operator action; the raw transcript was kept",
                )
                return False
            manifest.complete(
                "transcription",
                artifacts={"diarized_segments": ".work/diarized_segments.json"},
                artifact_hashes={"diarized_segments": _sha256(transcription_path)},
                validation={"diarized_segments": {"diarization_available": True}},
            )
        except BaseException as error:
            if manifest.stage("transcription")["status"] == "running":
                manifest.fail("transcription", error)
            raise
    return manifest.stage("transcription")["status"] == "completed"


def _event_signal(timestamp: str, *, authoritative: bool, signal_id: str, source: str) -> dict[str, Any]:
    parsed = _parse_iso(timestamp).astimezone(UTC)
    return {
        "id": signal_id,
        "kind": "user_override" if authoritative else "media_creation_time",
        "candidate_start_utc": parsed.isoformat().replace("+00:00", "Z"),
        "source": source,
        "detail": "operator-supplied event start" if authoritative else "embedded media creation timestamp",
        "strength": 1.0 if authoritative else 0.9,
        "independent_group": "operator" if authoritative else "media-container",
        "authoritative": authoritative,
    }


def _infer_event(
    run_dir: Path,
    *,
    evidence_path: Path | None = None,
    event_start: str | None = None,
    force: bool = False,
) -> bool:
    from .event_time import resolve_event_time, validate_event_evidence

    manifest = _manifest(run_dir)
    if not manifest.should_run("event", force=force):
        return manifest.stage("event")["status"] == "completed"
    data = manifest.data
    media = data["input"]["media"]
    duration = media.get("duration_seconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        raise _UsageProblem("recording duration is unavailable")
    if evidence_path is not None:
        raw = _load_json(evidence_path)
        if not isinstance(raw, Mapping):
            raise _UsageProblem("event evidence root must be an object")
        evidence = validate_event_evidence(raw)
        if abs(evidence.recording_duration_seconds - float(duration)) > 0.25:
            raise _UsageProblem("event evidence duration does not match the recording")
        signals = [signal.to_dict() for signal in evidence.signals]
    else:
        signals = []
        creation = media.get("creation_time")
        if isinstance(creation, str):
            signals.append(
                _event_signal(
                    creation,
                    authoritative=False,
                    signal_id="media-creation-time",
                    source="input media metadata",
                )
            )
    if event_start is not None:
        signals.append(
            _event_signal(
                event_start,
                authoritative=True,
                signal_id="operator-event-start",
                source="CLI event override",
            )
        )
    manifest.start("event", force=force)
    if not signals:
        _block(manifest, "event", "event_evidence_missing", "event time needs evidence or --event-start")
        return False
    evidence = validate_event_evidence(
        {
            "schema_version": 1,
            "recording_duration_seconds": float(duration),
            "signals": signals,
        }
    )
    work = run_dir / ".work"
    _atomic_json(work / "event_evidence.json", evidence.to_dict())
    resolution = resolve_event_time(
        evidence.signals,
        evidence.recording_duration_seconds,
        timezone_name=str(data["effective_config"].get("timezone", "UTC")),
    )
    _atomic_json(work / "event_resolution.json", resolution)
    if resolution.get("status") != "resolved":
        _block(manifest, "event", "ambiguous_event_time", "event time evidence is ambiguous")
        return False
    manifest.update_metadata(inferred_event=resolution)
    manifest.complete(
        "event",
        artifacts={
            "event_evidence": ".work/event_evidence.json",
            "event_resolution": ".work/event_resolution.json",
        },
        artifact_hashes={
            "event_evidence": _sha256(work / "event_evidence.json"),
            "event_resolution": _sha256(work / "event_resolution.json"),
        },
        validation={"event_resolution": {"status": "resolved"}},
    )
    return True


def _default_frame_stage(
    video: Path,
    diarized: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    work: Path,
    regions: Mapping[str, Sequence[float]] | None = None,
) -> Mapping[str, Any]:
    from .frames import (
        build_visual_evidence,
        create_contact_sheets,
        create_ocr_crops,
        extract_frames,
        merge_adjacent_speaker_segments,
        run_tesseract_ocr,
        select_frame_targets,
    )

    segments = diarized.get("segments", [])
    duration = diarized.get("duration_seconds")
    if not isinstance(duration, (int, float)):
        duration = max(
            (float(segment.get("end", 0)) for segment in segments if isinstance(segment, Mapping)),
            default=0.0,
        )
    turns = merge_adjacent_speaker_segments(segments if isinstance(segments, list) else ())
    targets = select_frame_targets(
        float(duration), segments=turns, events=events, periodic_interval=60
    )
    frame_directory = work / "frames"
    records = extract_frames(video, targets, frame_directory, managed_root=work)
    for record in records:
        if record.get("status") != "ok" or not isinstance(record.get("raw_path"), str):
            continue
        priority = record.get("priority")
        reasons = record.get("reasons")
        representative = (
            isinstance(priority, (int, float)) and not isinstance(priority, bool) and priority >= 85
        )
        periodic = isinstance(reasons, list) and "periodic_baseline" in reasons
        if not representative and not periodic:
            continue
        crops = create_ocr_crops(
            record["raw_path"], frame_directory / "ocr", regions=regions, managed_root=work
        )
        record["ocr"] = run_tesseract_ocr(crops)
    sheets = create_contact_sheets(records, frame_directory / "contact_sheets", managed_root=work)
    return build_visual_evidence(
        duration_seconds=float(duration), frame_records=records, contact_sheets=sheets
    )


def _json_list(path: Path | None, key: str, label: str) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    loaded = _load_json(path)
    if isinstance(loaded, Mapping):
        loaded = loaded.get(key)
    if not isinstance(loaded, list) or any(not isinstance(item, Mapping) for item in loaded):
        raise _UsageProblem(f"{label} must be a JSON array of objects")
    return loaded


def _extract_frames_stage(
    run_dir: Path,
    adapters: StageAdapters,
    *,
    events_path: Path | None = None,
    observations_path: Path | None = None,
    regions_path: Path | None = None,
    force: bool = False,
) -> bool:
    from .frames import (
        attach_presence_observations,
        prune_unreferenced_visual_derivatives,
        write_visual_evidence,
    )

    manifest = _manifest(run_dir)
    if observations_path is not None and manifest.stage("frames")["status"] == "completed":
        force = True
    if not manifest.should_run("frames", force=force):
        return manifest.stage("frames")["status"] == "completed"
    data = manifest.data
    if data["input"]["media"].get("has_video") is not True:
        raise _UsageProblem("frame extraction requires a recording with a video stream")
    diarized_path = run_dir / ".work" / "diarized_segments.json"
    if manifest.stage("transcription")["status"] != "completed" or not diarized_path.is_file():
        raise _UsageProblem("frame extraction requires a completed transcription")
    diarized = _load_json(diarized_path)
    if not isinstance(diarized, Mapping):
        raise ValueError("diarized segments are malformed")
    diarized = dict(diarized)
    diarized.setdefault("duration_seconds", data["input"]["media"].get("duration_seconds"))
    events = _json_list(events_path, "events", "frame events")
    observations = _json_list(observations_path, "observations", "observations")
    regions: Mapping[str, Sequence[float]] | None = None
    if regions_path is not None:
        loaded = _load_json(regions_path)
        if not isinstance(loaded, Mapping):
            raise _UsageProblem("OCR regions must be a JSON object of name -> [l, t, r, b]")
        regions = loaded
    work = run_dir / ".work"
    work_resolved = work.resolve(strict=True)
    for relative in ("frames", "frames/ocr", "frames/contact_sheets"):
        managed = work / relative
        if managed.is_symlink():
            raise ValueError("managed frame directory must not be a symlink")
        if managed.exists() and not managed.is_dir():
            raise ValueError("managed frame path must be a directory")
        try:
            managed.resolve(strict=False).relative_to(work_resolved)
        except ValueError as exc:
            raise ValueError("managed frame directory resolves outside .work") from exc
    source = Path(data["input"]["path"])
    manifest.start("frames", force=force)
    try:
        if adapters.extract_frames is not None:
            document = adapters.extract_frames(source, diarized, events, work)
        else:
            document = _default_frame_stage(source, diarized, events, work, regions)
        if observations:
            document = attach_presence_observations(document, observations)
        output = work / "visual_evidence.json"
        write_visual_evidence(output, document)
        removed = prune_unreferenced_visual_derivatives(document, work / "frames", managed_root=work)
        manifest.complete(
            "frames",
            artifacts={"visual_evidence": ".work/visual_evidence.json"},
            artifact_hashes={"visual_evidence": _sha256(output)},
            validation={
                "visual_evidence": {
                    "schema_version": document.get("schema_version"),
                    "observations": len(observations),
                    "removed_obsolete": removed,
                }
            },
        )
        return True
    except BaseException as error:
        manifest.fail("frames", error)
        raise


def _evidence_source_hashes(run_dir: Path, manifest: Any) -> dict[str, str] | None:
    """Hash the producer-owned sources that reviewed evidence cites."""

    stage, artifact, relative = _DIARIZED
    if not _stage_artifact_current(run_dir, manifest, stage=stage, name=artifact, relative_path=relative):
        return None
    hashes = {"diarized_segments": _sha256(run_dir / relative)}
    stage, artifact, relative = _VISUAL
    if manifest.stage("frames")["status"] == "completed":
        if not _stage_artifact_current(run_dir, manifest, stage=stage, name=artifact, relative_path=relative):
            return None
        hashes["visual_evidence"] = _sha256(run_dir / relative)
    return hashes


def _evidence_sources(run_dir: Path, manifest: Any) -> dict[str, Mapping[str, Any]]:
    work = run_dir / ".work"
    sources: dict[str, Mapping[str, Any]] = {}
    diarized = work / "diarized_segments.json"
    if not diarized.is_symlink() and diarized.is_file():
        document = _load_json(diarized)
        if isinstance(document, Mapping):
            sources["diarized"] = document
    visual = work / "visual_evidence.json"
    if (
        manifest.stage("frames")["status"] == "completed"
        and not visual.is_symlink()
        and visual.is_file()
    ):
        document = _load_json(visual)
        if isinstance(document, Mapping):
            sources["visual"] = document
    return sources


def _evidence_current(run_dir: Path, manifest: Any) -> bool:
    evidence = manifest.stage("evidence")
    validation = evidence.get("validation", {}).get("evidence")
    if (
        evidence.get("status") != "completed"
        or not isinstance(validation, Mapping)
        or validation.get("valid") is not True
    ):
        return False
    for name in _EVIDENCE_LANES:
        if not _stage_artifact_current(
            run_dir, manifest, stage="evidence", name=name, relative_path=f".work/{name}.json"
        ):
            return False
    edits = run_dir / ".work" / "transcript_edits.json"
    if edits.exists() and not _stage_artifact_current(
        run_dir,
        manifest,
        stage="evidence",
        name="transcript_edits",
        relative_path=".work/transcript_edits.json",
    ):
        return False
    current_sources = _evidence_source_hashes(run_dir, manifest)
    recorded = validation.get("source_hashes")
    return current_sources is not None and isinstance(recorded, Mapping) and dict(recorded) == current_sources


def _ingest_known_evidence(run_dir: Path, args: argparse.Namespace, manifest: Any) -> None:
    """Validate the complete candidate bundle before replacing any managed file."""

    work = run_dir / ".work"
    if work.is_symlink() or not work.is_dir():
        raise ValueError("run .work must be a real contained directory")
    if work.resolve(strict=True) != run_dir / ".work":
        raise ValueError("run .work resolves outside its expected location")
    managed = (
        ("speaker_evidence", "speaker_evidence.json"),
        ("qa_evidence", "qa_evidence.json"),
        ("transcript_edits", "transcript_edits.json"),
    )
    supplied: dict[str, Mapping[str, Any]] = {}
    targets: dict[str, Path] = {}
    for argument, filename in managed:
        value = getattr(args, argument, None)
        if value is None:
            continue
        source = Path(value).expanduser()
        if source.is_symlink() or not source.is_file():
            raise _EvidenceIngestValidationError(f"{argument} must be a real regular file")
        document = _load_json(source)
        if not isinstance(document, Mapping):
            raise _EvidenceIngestValidationError(f"{argument} must contain a JSON object")
        target = work / filename
        if target.is_symlink() or target.parent.resolve(strict=True) != work:
            raise ValueError(f"{argument} target is unsafe")
        supplied[argument] = document
        targets[argument] = target
    if not supplied:
        return

    candidate: dict[str, Mapping[str, Any]] = {}
    for argument, filename in managed:
        document = supplied.get(argument)
        target = work / filename
        if document is None and target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"managed {argument} is unsafe")
            loaded = _load_json(target)
            if isinstance(loaded, Mapping):
                document = loaded
        if document is not None:
            candidate[argument] = document

    sources = _evidence_sources(run_dir, manifest)
    if "diarized" not in sources:
        raise _EvidenceIngestValidationError("reviewed evidence requires the canonical transcript")
    diarized: Mapping[str, Any] = sources["diarized"]
    edits = candidate.get("transcript_edits")
    if edits is not None:
        from .transcript_edits import TranscriptEditError, apply_transcript_edits

        try:
            diarized = apply_transcript_edits(
                diarized,
                edits,
                canonical_sha256=_sha256(work / "diarized_segments.json"),
            )
        except TranscriptEditError as exc:
            raise _EvidenceIngestValidationError(str(exc)) from exc

    lanes_supplied = any(name in supplied for name in _EVIDENCE_LANES)
    lanes_present = all(name in candidate for name in _EVIDENCE_LANES)
    if lanes_supplied or (edits is not None and lanes_present):
        if not lanes_present:
            raise _EvidenceIngestValidationError(
                "both speaker and Q&A evidence are required"
            )
        from .evidence import validate_evidence_bundle
        from .validate import confirmed_speaker_names

        result = validate_evidence_bundle(
            candidate["speaker_evidence"],
            candidate["qa_evidence"],
            diarized,
            sources.get("visual"),
            raise_on_error=False,
            confirmed_names=confirmed_speaker_names(manifest.data),
        )
        if not isinstance(result, Mapping):
            raise _EvidenceIngestValidationError(
                "the supplied evidence failed validation: "
                + "; ".join(
                    f"{issue['path']} [{issue['code']}] {issue['message']}"
                    for issue in result[:20]
                )
            )

    # Every candidate was parsed and checked. Only supplied documents are
    # committed, so previously reviewed files stay byte-identical on failure.
    for argument, _filename in managed:
        if argument in supplied:
            _atomic_json(targets[argument], supplied[argument])


def _complete_render_artifacts(
    run_dir: Path,
    manifest: Any,
    artifacts: Mapping[str, Path],
    *,
    final_validation: Mapping[str, Any] | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> None:
    by_name = {Path(path).name: Path(path) for path in artifacts.values()}
    if set(_PUBLIC_OUTPUTS) - set(by_name):
        raise ValueError("renderer must return transcript, Q&A, and review")
    paths = {
        key: by_name[name].relative_to(run_dir).as_posix() for name, key in _PUBLIC_OUTPUTS.items()
    }
    if manifest.stage("render")["status"] != "running":
        manifest.start("render", force=True)
    validation: dict[str, Any] = {}
    if final_validation is not None:
        validation["final_validation"] = dict(final_validation)
    if input_hashes is not None:
        validation["input_hashes"] = dict(input_hashes)
    manifest.complete(
        "render",
        artifacts=paths,
        artifact_hashes={name: _sha256(run_dir / path) for name, path in paths.items()},
        validation=validation,
    )


def _fallback_review(run_dir: Path, payload: Mapping[str, Any]) -> None:
    from .render import NOT_READY_STATUS

    lines = [
        "# Review",
        f"{NOT_READY_STATUS} — final validation failed; resolve the issues below.",
        "",
        "## Validation issues",
    ]
    issues = payload.get("issues", [])
    for item in issues if isinstance(issues, list) else []:
        if isinstance(item, Mapping):
            code = str(item.get("code", "validation_failed")).replace("\n", " ")
            path = str(item.get("path", ".")).replace("\n", " ")
            message = str(item.get("message", "validation failed")).replace("\n", " ")
            lines.append(f"- [{code}] {path}: {message}")
    if len(lines) == 4:
        lines.append("- Final validation did not provide structured details.")
    _write_atomic(run_dir / "review.md", "\n".join(lines) + "\n")


def _render_stage(run_dir: Path, adapters: StageAdapters, *, force: bool = False) -> bool:
    from .validate import build_render_bundle

    manifest = _manifest(run_dir)

    def block_review(code: str, message: str) -> None:
        if manifest.stage("review")["status"] == "running":
            manifest.block("review", code=code, message=message)
        else:
            _block(manifest, "review", code, message)

    def demote(payload: Mapping[str, Any], input_hashes: Mapping[str, str] | None) -> bool:
        try:
            bundle = build_render_bundle(
                run_dir,
                manifest=manifest.data,
                artifact_validation={"valid": False, "issues": payload.get("issues", [])},
            )
            artifacts = _render(run_dir, bundle, adapters)
        except Exception:
            _fallback_review(run_dir, payload)
            artifacts = {name: run_dir / name for name in _PUBLIC_OUTPUTS}
        if all(path.is_file() for path in artifacts.values()):
            _complete_render_artifacts(
                run_dir,
                manifest,
                artifacts,
                final_validation=payload,
                input_hashes=input_hashes,
            )
        block_review("final_validation_failed", "outputs failed final validation")
        return False

    def finish_review(input_hashes: Mapping[str, str] | None) -> bool:
        review_status = manifest.stage("review")["status"]
        if review_status != "running":
            manifest.start("review", force=review_status in {"blocked", "completed"})
        try:
            report = _validate(run_dir, "final", adapters)
        except Exception as error:
            issue = {
                "path": ".",
                "code": "validator_exception",
                "message": f"final validator raised {type(error).__name__}",
                "blocking": True,
            }
            return demote({"valid": False, "issues": [issue], "blockers": [issue]}, input_hashes)
        if not bool(getattr(report, "valid", False)):
            payload = (
                report.to_dict()
                if hasattr(report, "to_dict")
                else {"valid": False, "issues": [], "blockers": []}
            )
            return demote(payload, input_hashes)
        manifest.complete("review", validation={"review": {"status": "ready"}})
        return True

    if not manifest.should_run("render", force=force):
        if _render_outputs_current(run_dir, manifest) and _render_inputs_current(run_dir, manifest):
            review = manifest.stage("review")["status"]
            if review == "completed":
                return True
            if review != "blocked":
                return finish_review(manifest.stage("render").get("validation", {}).get("input_hashes"))
        # A blocked review is re-derived from the sources by rendering again,
        # so a resolved decision unblocks it and an open one blocks it again.
        force = True

    if not _evidence_current(run_dir, manifest):
        if manifest.stage("render")["status"] == "completed":
            # Keep the previous outputs registered, but they no longer
            # describe validated evidence and must not be treated as final.
            _block(manifest, "review", "evidence_changed", "evidence changed after rendering; validate again")
        else:
            _block(
                manifest,
                "render",
                "evidence_not_validated",
                "rendering requires current, hash-bound, validated evidence",
            )
        return False
    input_hashes = _render_input_hashes(run_dir, manifest)
    if input_hashes is None:
        _block(manifest, "render", "render_input_stale", "a hash-bound render input is missing or changed")
        return False
    try:
        bundle = build_render_bundle(run_dir, manifest=manifest.data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        _block(manifest, "render", "render_evidence_missing", "rendering requires current transcript and evidence")
        return False

    manifest.start("render", force=force or manifest.stage("render")["status"] in {"blocked", "completed"})
    try:
        artifacts = _render(run_dir, bundle, adapters)
        _complete_render_artifacts(run_dir, manifest, artifacts, input_hashes=input_hashes)
    except BaseException as error:
        if manifest.stage("render")["status"] == "running":
            manifest.fail("render", error)
        raise

    from .render import collect_review_items

    if any(item["blocking"] for item in collect_review_items(bundle)):
        if manifest.stage("review")["status"] in {"blocked", "completed"}:
            manifest.start("review", force=True)
        _block(manifest, "review", "manual_review_required", "review.md lists blocking items")
        return False
    return finish_review(input_hashes)


def _reviewed_evidence_source_drift(run_dir: Path, manifest: Any) -> bool:
    """Return whether reviewed evidence is bound to an older source snapshot."""

    blockers = manifest.data.get("blockers")
    if isinstance(blockers, list) and any(
        isinstance(blocker, Mapping)
        and blocker.get("stage") == "evidence"
        and blocker.get("code") == "reviewed_evidence_stale"
        and blocker.get("resolved_at") is None
        for blocker in blockers
    ):
        return True
    validation = manifest.stage("evidence").get("validation", {}).get("evidence", {})
    recorded = validation.get("source_hashes") if isinstance(validation, Mapping) else None
    if not isinstance(recorded, Mapping):
        return False
    current = _evidence_source_hashes(run_dir, manifest)
    return current is None or dict(recorded) != current


def _ingest_validation_report(phase: str, message: str, *, code: str = "reviewed_evidence_invalid") -> Any:
    from .validate import ValidationIssue, ValidationReport

    issue = ValidationIssue(".work/evidence", code, message)
    return ValidationReport(False, phase, (issue,), (issue,))  # type: ignore[arg-type]


def _validate_stage(
    run_dir: Path,
    phase: str,
    adapters: StageAdapters,
    *,
    force: bool = False,
    args: argparse.Namespace | None = None,
) -> Any:
    """Validate and, for evidence/all, own the evidence manifest stage."""

    if phase == "final":
        return _validate(run_dir, phase, adapters)
    manifest = _manifest(run_dir)
    supplied = {
        field for field in _EVIDENCE_LANES if args is not None and getattr(args, field, None) is not None
    }
    if _reviewed_evidence_source_drift(run_dir, manifest) and supplied != set(_EVIDENCE_LANES):
        message = "the transcript or visual evidence changed; re-supply both evidence documents"
        current = manifest.stage("evidence")
        if current["status"] == "completed":
            manifest.start("evidence", force=True)
            current = manifest.stage("evidence")
        if current["status"] != "blocked":
            manifest.block("evidence", code="reviewed_evidence_stale", message=message)
        return _ingest_validation_report(phase, message, code="reviewed_evidence_stale")

    confirmations = getattr(args, "confirm_speaker", None) if args is not None else None
    for label, name in confirmations or ():
        from .validate import SPEAKER_CONFIRMATION

        if re.fullmatch(r"SPEAKER_[A-Za-z0-9_-]+", label) is None or not name.strip() or len(name) > 200:
            raise _UsageProblem("--confirm-speaker needs a SPEAKER_xx label and a non-empty name")
        manifest.record_manual_decision(
            f"speaker-{label}-{_stable_value_hash([label, name.strip()])[:12]}",
            SPEAKER_CONFIRMATION,
            details={"speaker_label": label, "display_name": name.strip()},
        )

    try:
        if args is not None:
            _ingest_known_evidence(run_dir, args, manifest)
    except _EvidenceIngestValidationError as exc:
        return _ingest_validation_report(phase, str(exc))

    explicit_update = bool(confirmations) or (
        args is not None
        and any(
            getattr(args, field, None) is not None
            for field in (*_EVIDENCE_LANES, "transcript_edits")
        )
    )
    stage_started = manifest.should_run("evidence", force=force or explicit_update)
    if stage_started:
        manifest.start("evidence", force=force or explicit_update)
    report = _validate(run_dir, "evidence", adapters)
    if not bool(getattr(report, "valid", False)):
        if stage_started:
            manifest.fail("evidence", {"type": "ValidationError", "message": "evidence validation failed"})
        return report
    paths = {name: f".work/{name}.json" for name in _EVIDENCE_LANES}
    if (run_dir / ".work" / "transcript_edits.json").is_file():
        paths["transcript_edits"] = ".work/transcript_edits.json"
    hashes = {name: _sha256(run_dir / path) for name, path in paths.items()}
    source_hashes = _evidence_source_hashes(run_dir, manifest)
    if source_hashes is None:
        if stage_started:
            manifest.fail("evidence", {"type": "ValidationError", "message": "evidence sources are stale"})
        raise ValueError("evidence source dependencies are stale")
    current = manifest.stage("evidence")
    evidence_validation = current.get("validation", {}).get("evidence", {})
    changed = (
        current.get("status") != "completed"
        or current.get("artifacts") != paths
        or current.get("artifact_hashes") != hashes
        or evidence_validation.get("valid") is not True
        or evidence_validation.get("source_hashes") != source_hashes
    )
    if changed:
        if not stage_started:
            manifest.start("evidence", force=True)
        manifest.complete(
            "evidence",
            artifacts=paths,
            artifact_hashes=hashes,
            validation={"evidence": {"valid": True, "source_hashes": source_hashes}},
        )
    if phase == "all":
        return _validate(run_dir, "all", adapters)
    return report


def _status_payload(run_dir: Path) -> dict[str, Any]:
    if not (run_dir / ".work" / "manifest.json").is_file():
        raise _UsageProblem("run manifest does not exist")
    manifest = _manifest(run_dir)
    data = manifest.data
    stages = {
        name: record.get("status")
        for name, record in data.get("stages", {}).items()
        if isinstance(record, Mapping)
    }
    unresolved = [
        {"stage": blocker.get("stage"), "code": blocker.get("code"), "message": blocker.get("message")}
        for blocker in data.get("blockers", [])
        if isinstance(blocker, Mapping) and blocker.get("resolved_at") is None
    ]
    next_action = "complete"
    refresh = _render_refresh_action(run_dir, manifest)
    if refresh is not None:
        next_action = refresh
    else:
        for stage in _STAGE_ORDER:
            status = stages.get(stage, "pending")
            if status == "blocked":
                next_action = "resolve_review" if stage == "review" else f"manual:{stage}"
                break
            if stage in _OPTIONAL_STAGES:
                if status == "failed":
                    next_action = _STAGE_COMMANDS[stage]
                    break
                continue
            if status != "completed":
                next_action = _STAGE_COMMANDS[stage]
                break
    return {
        "run": str(run_dir),
        "stages": stages,
        "artifacts": data.get("artifacts", {}),
        "blockers": unresolved,
        "next_action": next_action,
    }


def _resume(run_dir: Path, adapters: StageAdapters, *, force: bool = False) -> bool:
    manifest = _manifest(run_dir)
    if _reviewed_evidence_source_drift(run_dir, manifest):
        report = _validate_stage(run_dir, "evidence", adapters, force=force)
        return bool(getattr(report, "valid", False))
    data = manifest.data
    if any(manifest.stage(stage)["status"] != "completed" for stage in ("media", "transcription")):
        if not _prepare_run(
            Path(data["input"]["path"]), run_dir, data["effective_config"], adapters, force=force
        ):
            return False
        manifest = _manifest(run_dir)
    work = run_dir / ".work"
    if not all((work / f"{name}.json").is_file() for name in _EVIDENCE_LANES):
        return False
    review_blocked = manifest.stage("review")["status"] == "blocked"
    report = _validate_stage(run_dir, "evidence", adapters, force=force or review_blocked)
    if not getattr(report, "valid", False):
        return False
    return _render_stage(run_dir, adapters, force=force or review_blocked)


def _add_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, type=Path, help="existing run directory")
    parser.add_argument("--force-stage", action="store_true", help="rerun the selected stage")


def _add_evidence_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--speaker-evidence", type=Path, help="speaker naming evidence JSON")
    parser.add_argument("--qa-evidence", type=Path, help="question and answer evidence JSON")
    parser.add_argument("--transcript-edits", type=Path, help="reviewed transcript edit overlay JSON")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog=PROG,
        description="Resumable recording -> speaker-attributed transcript -> checked Q&A pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="probe media, create a run, extract FLAC, transcribe and diarize")
    prepare.add_argument("recording", type=Path)
    prepare.add_argument("--output-root", type=Path, default=Path("outputs"))
    prepare.add_argument("--title", help="title used in rendered headings (default: file name)")
    prepare.add_argument("--timezone", help="IANA timezone for rendered dates (default: UTC)")
    prepare.add_argument("--model", help="WhisperX ASR model (default: large-v3-turbo)")
    prepare.add_argument("--language", help="language code, or 'auto' to detect")
    prepare.add_argument("--min-speakers", type=_positive_int)
    prepare.add_argument("--max-speakers", type=_positive_int)
    prepare.add_argument("--num-speakers", type=_positive_int)
    prepare.add_argument("--force-stage", action="store_true")

    infer_event = subparsers.add_parser("infer-event", help="resolve when the recording started (optional)")
    _add_run(infer_event)
    infer_event.add_argument("--evidence", type=Path, help="event-time evidence JSON")
    infer_event.add_argument("--event-start", help="authoritative timezone-aware ISO-8601 start")

    frames = subparsers.add_parser("extract-frames", help="sample video frames and OCR them (optional)")
    _add_run(frames)
    frames.add_argument("--events", type=Path, help="extra timestamps to sample, JSON array")
    frames.add_argument("--observations", type=Path, help="reviewed frame observations, JSON array")
    frames.add_argument("--ocr-regions", type=Path, help="JSON object name -> [left, top, right, bottom]")

    validate = subparsers.add_parser("validate", help="ingest and validate evidence, or validate final outputs")
    _add_run(validate)
    validate.add_argument("--phase", choices=("evidence", "final", "all"), default="final")
    _add_evidence_paths(validate)
    validate.add_argument(
        "--confirm-speaker",
        nargs=2,
        action="append",
        metavar=("LABEL", "NAME"),
        help="record that a human confirmed SPEAKER_xx is NAME (audited in the manifest)",
    )

    render = subparsers.add_parser("render", help="render transcript.md, qa.md and review.md, then validate them")
    _add_run(render)

    status = subparsers.add_parser("status", help="show run state without loading the ML stack")
    status.add_argument("--run", required=True, type=Path)

    resume = subparsers.add_parser("resume", help="advance every deterministic stage that is ready")
    _add_run(resume)
    return parser


def _emit(payload: Mapping[str, Any], stream: TextIO) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    adapters: StageAdapters | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Parse and execute one command, returning a stable process exit code."""

    selected = adapters or StageAdapters()
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            recording = args.recording.expanduser().resolve(strict=False)
            if not recording.is_file():
                raise _UsageProblem("recording does not exist")
            args.recording = recording
            config = _config_from_args(args)
            metadata = dict(_probe(recording, selected))
            run_dir = args.output_root.expanduser().resolve(strict=False) / _run_name(
                metadata, config, _sha256(recording)
            )
            success = _prepare_run(
                recording,
                run_dir,
                config.to_dict(),
                selected,
                metadata=metadata,
                force=args.force_stage,
            )
            _emit({"run": str(run_dir), "status": "prepared" if success else "blocked"}, output)
            return SUCCESS_EXIT if success else BLOCKED_EXIT

        run_dir = args.run.expanduser().resolve(strict=False)
        if args.command == "status":
            _emit(_status_payload(run_dir), output)
            return SUCCESS_EXIT
        _existing_run_manifest(run_dir)
        if args.command == "infer-event":
            success = _infer_event(
                run_dir,
                evidence_path=args.evidence,
                event_start=args.event_start,
                force=args.force_stage,
            )
        elif args.command == "extract-frames":
            success = _extract_frames_stage(
                run_dir,
                selected,
                events_path=args.events,
                observations_path=args.observations,
                regions_path=args.ocr_regions,
                force=args.force_stage,
            )
        elif args.command == "validate":
            report = _validate_stage(run_dir, args.phase, selected, force=args.force_stage, args=args)
            payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            _emit(payload, output)
            return SUCCESS_EXIT if getattr(report, "valid", payload.get("valid")) else VALIDATION_FAILURE_EXIT
        elif args.command == "render":
            success = _render_stage(run_dir, selected, force=args.force_stage)
        elif args.command == "resume":
            success = _resume(run_dir, selected, force=args.force_stage)
        else:
            raise AssertionError("unreachable command")
        _emit(
            {
                "run": str(run_dir),
                "status": "completed" if success else "blocked",
                "next_action": _status_payload(run_dir)["next_action"],
            },
            output,
        )
        return SUCCESS_EXIT if success else BLOCKED_EXIT
    except (KeyboardInterrupt, SystemExit):
        raise
    except _UsageProblem as problem:
        _emit({"error": "usage", "message": str(problem)}, errors)
        return USAGE_EXIT
    except BaseException as error:
        # Only the exception type is printed: messages may echo paths or
        # library output that was never meant to leave the machine.
        _emit({"error": "stage_failed", "type": type(error).__name__}, errors)
        return STAGE_FAILURE_EXIT


__all__ = [
    "BLOCKED_EXIT",
    "STAGE_FAILURE_EXIT",
    "SUCCESS_EXIT",
    "USAGE_EXIT",
    "VALIDATION_FAILURE_EXIT",
    "StageAdapters",
    "build_parser",
    "main",
]
