"""End-to-end, credential-safe validation for one run directory."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

ValidationPhase = Literal["evidence", "final", "all"]

_SECRET_SCAN_CHUNK_BYTES = 1024 * 1024
_SECRET_SCAN_OVERLAP_BYTES = 64 * 1024
_BINARY_CREDENTIAL_PREFIX_BYTES = 4 * 1024
_SECRET_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9_-])[\"']?Authorization[\"']?\s*:\s*[\"']?"
        r"(?:Bearer|Bot|Basic)\s+\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9_-])[\"']?(?:[A-Za-z0-9]+_)*(?:api[_-]?key|access[_-]?token|"
        r"auth[_-]?token|refresh[_-]?token|hf[_-]?token|client[_-]?secret|"
        r"private[_-]?key|token|passwords?|passwd|secrets?|credentials?)[\"']?"
        r"\s*[:=]\s*(?:[^\s,;]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(
        r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@[^\s/]+",
        re.IGNORECASE,
    ),
)
# A URL-like user-info run can appear by chance in compressed media bytes. The
# preceding patterns are explicit credential syntax and remain safe to scan in
# every file type; only the final generic URL heuristic is text-only.
_BINARY_SAFE_SECRET_PATTERNS = _SECRET_PATTERNS[:-1]
_BINARY_SAFE_SECRET_BYTE_PATTERNS = tuple(
    re.compile(pattern.pattern.encode("ascii"), pattern.flags & re.IGNORECASE)
    for pattern in _BINARY_SAFE_SECRET_PATTERNS
)
_BINARY_MEDIA_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"fLaC",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
)
_REDACTED_MANAGED_PATH = "[redacted-managed-path]"
FLAC_NAME = "audio.flac"
PUBLIC_OUTPUTS = ("transcript.md", "qa.md")
REVIEW_NAME = "review.md"
_EVIDENCE_FILES = {
    "speaker": "speaker_evidence.json",
    "qa": "qa_evidence.json",
}
_SOURCE_STAGE_FILES = {
    "transcription": ".work/diarized_segments.json",
    "frames": ".work/visual_evidence.json",
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One safe, structured validation problem."""

    path: str
    code: str
    message: str
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """JSON-ready aggregate validation result."""

    valid: bool
    phase: ValidationPhase
    issues: tuple[ValidationIssue, ...]
    blockers: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "phase": self.phase,
            "issues": [issue.to_dict() for issue in self.issues],
            "blockers": [issue.to_dict() for issue in self.blockers],
        }


SPEAKER_CONFIRMATION = "speaker name confirmed"


def confirmed_speaker_names(manifest: Mapping[str, Any] | None) -> dict[str, str]:
    """Return the latest operator-confirmed name per speaker label."""

    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("manual_decisions"), list):
        return {}
    names: dict[str, str] = {}
    for record in manifest["manual_decisions"]:
        if not isinstance(record, Mapping) or record.get("decision") != SPEAKER_CONFIRMATION:
            continue
        details = record.get("details")
        if not isinstance(details, Mapping):
            continue
        label = details.get("speaker_label")
        name = details.get("display_name")
        if isinstance(label, str) and isinstance(name, str) and name.strip():
            names[label] = name.strip()
    return names


def _operator_decisions(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = manifest.get("manual_decisions")
    if not isinstance(decisions, list):
        return []
    return [
        {
            "decision": record.get("decision"),
            "recorded_at": record.get("recorded_at"),
            "details": record.get("details"),
        }
        for record in decisions
        if isinstance(record, Mapping)
    ]


def _registered(manifest: Mapping[str, Any], stage: str, relative: str) -> Mapping[str, Any] | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    return next(
        (
            record
            for record in artifacts.values()
            if isinstance(record, Mapping)
            and record.get("stage") == stage
            and record.get("path") == relative
        ),
        None,
    )


def _load_bound_json(path: Path, record: Mapping[str, Any] | None, label: str) -> Any:
    from .media import sha256_file

    if record is None:
        raise ValueError(f"{label} is not registered")
    expected = record.get("sha256")
    if not isinstance(expected, str):
        raise ValueError(f"{label} is not hash-bound")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} is stale")
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def build_render_bundle(
    run_dir: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any] | None = None,
    artifact_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the single normalized source contract consumed by every renderer."""

    directory = Path(run_dir).expanduser().resolve(strict=False)
    if manifest is None:
        with (directory / ".work" / "manifest.json").open("r", encoding="utf-8") as source:
            loaded = json.load(source)
        if not isinstance(loaded, Mapping):
            raise ValueError("manifest root must be an object")
        manifest = loaded
    config = manifest.get("effective_config")
    media = manifest.get("input", {}).get("media", {}) if isinstance(manifest.get("input"), Mapping) else {}
    bundle: dict[str, Any] = {
        "title": config.get("title") if isinstance(config, Mapping) else None,
        "event": manifest.get("inferred_event", {}),
        "duration_seconds": media.get("duration_seconds") if isinstance(media, Mapping) else None,
        "artifact_validation": dict(artifact_validation or {"valid": True, "issues": []}),
        "operator_decisions": _operator_decisions(manifest),
    }
    work = directory / ".work"
    for key, filename in (
        ("diarized", "diarized_segments.json"),
        ("speaker_evidence", "speaker_evidence.json"),
        ("qa_evidence", "qa_evidence.json"),
    ):
        path = work / filename
        if not path.is_file():
            raise ValueError(f"render source is missing: {filename}")
        with path.open("r", encoding="utf-8") as source:
            document = json.load(source)
        if not isinstance(document, Mapping):
            raise ValueError(f"render source must be an object: {filename}")
        bundle[key] = document

    visual_path = work / "visual_evidence.json"
    visual_record = _registered(manifest, "frames", ".work/visual_evidence.json")
    bundle["visual_evidence"] = (
        _load_bound_json(visual_path, visual_record, "visual evidence")
        if visual_record is not None and visual_path.is_file()
        else None
    )

    edits_path = work / "transcript_edits.json"
    if edits_path.is_file():
        from .media import sha256_file
        from .transcript_edits import apply_transcript_edits

        edits = _load_bound_json(
            edits_path,
            _registered(manifest, "evidence", ".work/transcript_edits.json"),
            "reviewed transcript edits",
        )
        if not isinstance(edits, Mapping):
            raise ValueError("reviewed transcript edits must be an object")
        bundle["diarized"] = apply_transcript_edits(
            bundle["diarized"],
            edits,
            canonical_sha256=sha256_file(work / "diarized_segments.json"),
        )
    return bundle


def _validate_source_bindings(
    run_dir: Path, manifest: Mapping[str, Any], issues: list[ValidationIssue]
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    for stage, relative in _SOURCE_STAGE_FILES.items():
        if not (run_dir / relative).is_file():
            continue
        record = next(
            (
                item
                for item in artifacts.values()
                if isinstance(item, Mapping) and item.get("path") == relative
            ),
            None,
        )
        if not isinstance(record, Mapping):
            _issue(issues, relative, "source_unregistered", f"source document must be hash-bound by its {stage} stage")
        elif record.get("stage") != stage:
            _issue(issues, relative, "source_stage_mismatch", f"source document must remain owned by its {stage} stage")


def _issue(
    issues: list[ValidationIssue],
    path: str,
    code: str,
    message: str,
    *,
    blocking: bool = True,
) -> None:
    issues.append(ValidationIssue(path, code, message, blocking))


def _strict_json(path: Path, issues: list[ValidationIssue]) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(
                source,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non-finite number")
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        _issue(issues, path.name, "invalid_json", "file is not strict UTF-8 JSON")
        return None


def _relative(run_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _safe_artifact_path(run_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return None
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        return None
    candidate = (run_dir / raw).resolve(strict=False)
    try:
        candidate.relative_to(run_dir)
    except ValueError:
        return None
    return candidate


def _manifest_document(run_dir: Path, issues: list[ValidationIssue]) -> dict[str, Any] | None:
    path = run_dir / ".work" / "manifest.json"
    if not path.is_file():
        _issue(issues, ".work/manifest.json", "manifest_missing", "run manifest is missing")
        return None
    raw = _strict_json(path, issues)
    if not isinstance(raw, dict):
        if raw is not None:
            _issue(issues, ".work/manifest.json", "manifest_invalid", "manifest root must be an object")
        return None
    try:
        from .manifest import Manifest

        Manifest(path, run_dir=run_dir)
    except (OSError, TypeError, ValueError, RuntimeError):
        _issue(issues, ".work/manifest.json", "manifest_invalid", "manifest does not satisfy the run-state contract")
    return raw


def _validate_hashes(
    run_dir: Path, manifest: Mapping[str, Any], issues: list[ValidationIssue]
) -> None:
    from .media import sha256_file

    input_record = manifest.get("input")
    if isinstance(input_record, Mapping):
        source = input_record.get("path")
        expected = input_record.get("sha256")
        if isinstance(source, str) and isinstance(expected, str):
            source_path = Path(source)
            if not source_path.is_file():
                _issue(issues, "input.path", "source_missing", "bound recording is missing")
            else:
                try:
                    if sha256_file(source_path) != expected:
                        _issue(issues, "input.path", "hash_mismatch", "recording hash differs from manifest binding")
                except OSError:
                    _issue(issues, "input.path", "file_unreadable", "recording cannot be hashed")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return
    for name, record in artifacts.items():
        path_label = f"artifacts.{name}"
        if not isinstance(record, Mapping):
            _issue(issues, path_label, "artifact_invalid", "artifact record must be an object")
            continue
        artifact = _safe_artifact_path(run_dir, record.get("path"))
        if artifact is None:
            _issue(issues, path_label, "unsafe_path", "artifact path escapes the run directory")
            continue
        if not artifact.is_file():
            _issue(issues, _relative(run_dir, artifact), "artifact_missing", "manifest artifact is missing")
            continue
        expected = record.get("sha256")
        if not isinstance(expected, str):
            _issue(issues, _relative(run_dir, artifact), "artifact_hash_missing", "manifest artifact has no SHA-256 hash")
            continue
        try:
            if sha256_file(artifact) != expected:
                _issue(issues, _relative(run_dir, artifact), "hash_mismatch", "artifact hash differs from manifest")
        except OSError:
            _issue(issues, _relative(run_dir, artifact), "file_unreadable", "artifact cannot be hashed")


def _validate_flac(
    run_dir: Path,
    manifest: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> None:
    flac = run_dir / FLAC_NAME
    if not flac.is_file():
        return
    try:
        from .media import validate_canonical_flac

        metadata = validate_canonical_flac(flac)
    except (OSError, RuntimeError, TypeError, ValueError):
        _issue(issues, flac.name, "flac_invalid", "canonical FLAC metadata or full decode failed")
        return
    if not isinstance(manifest, Mapping):
        return
    record = manifest.get("input", {})
    media = record.get("media", {}) if isinstance(record, Mapping) else {}
    expected = media.get("duration_seconds") if isinstance(media, Mapping) else None
    actual = metadata.get("duration_seconds")
    if (
        isinstance(expected, (int, float))
        and isinstance(actual, (int, float))
        and math.isfinite(float(expected))
        and abs(float(expected) - float(actual)) > 0.25
    ):
        _issue(issues, flac.name, "duration_mismatch", "canonical FLAC duration differs from source metadata")


def _diarized_with_edits(run_dir: Path, issues: list[ValidationIssue]) -> Any | None:
    """Return the canonical transcript with any reviewed edit overlay applied."""

    diarized_path = run_dir / ".work" / "diarized_segments.json"
    canonical = _strict_json(diarized_path, issues) if diarized_path.is_file() else None
    edits_path = run_dir / ".work" / "transcript_edits.json"
    if not isinstance(canonical, Mapping) or not edits_path.exists():
        return canonical
    if edits_path.is_symlink() or not edits_path.is_file():
        _issue(issues, ".work/transcript_edits.json", "transcript_edits_invalid", "reviewed transcript edits must be a real regular file")
        return canonical
    edits = _strict_json(edits_path, issues)
    if not isinstance(edits, Mapping):
        return canonical
    try:
        from .media import sha256_file
        from .transcript_edits import apply_transcript_edits

        return apply_transcript_edits(
            canonical,
            edits,
            canonical_sha256=sha256_file(diarized_path),
        )
    except (TypeError, ValueError):
        _issue(
            issues,
            ".work/transcript_edits.json",
            "transcript_edits_invalid",
            "reviewed transcript edits do not match the canonical transcript",
        )
        return canonical


def _validate_evidence(
    run_dir: Path,
    manifest: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> None:
    documents: dict[str, Any] = {}
    for kind, filename in _EVIDENCE_FILES.items():
        path = run_dir / ".work" / filename
        if path.is_file():
            documents[kind] = _strict_json(path, issues)
        else:
            _issue(issues, f".work/{filename}", "required_evidence_missing", "this reviewed evidence document is required")
    diarized = _diarized_with_edits(run_dir, issues)
    if diarized is None:
        _issue(issues, ".work/diarized_segments.json", "required_evidence_missing", "evidence validation requires the transcript")
        return
    visual = None
    visual_path = run_dir / ".work" / "visual_evidence.json"
    if (
        visual_path.is_file()
        and isinstance(manifest, Mapping)
        and _registered(manifest, "frames", ".work/visual_evidence.json") is not None
    ):
        visual = _strict_json(visual_path, issues)
    try:
        from .evidence import validate_qa_evidence, validate_speaker_evidence

        confirmed = confirmed_speaker_names(manifest)
        for kind, document in documents.items():
            if kind == "speaker":
                result = validate_speaker_evidence(
                    document, diarized, visual, raise_on_error=False, confirmed_names=confirmed
                )
            else:
                result = validate_qa_evidence(document, diarized, visual, raise_on_error=False)
            if isinstance(result, list):
                for item in result:
                    _issue(
                        issues,
                        f".work/{_EVIDENCE_FILES[kind]}:{item.get('path', '$')}",
                        str(item.get("code", "evidence_invalid")),
                        str(item.get("message", "evidence contract failed")),
                    )
    except (TypeError, ValueError, RuntimeError):
        _issue(issues, ".work", "evidence_invalid", "evidence validation could not complete")


def _unresolved_manifest_blockers(manifest: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("blockers"), list):
        return []
    return [
        blocker
        for blocker in manifest["blockers"]
        if isinstance(blocker, Mapping) and blocker.get("resolved_at") is None
    ]


def _surface_manifest_blockers(
    manifest: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
    *,
    phase: ValidationPhase,
) -> None:
    for blocker in _unresolved_manifest_blockers(manifest):
        stage = blocker.get("stage")
        code = blocker.get("code")
        label = stage if isinstance(stage, str) and stage else "unknown"
        safe_code = code if isinstance(code, str) and code else "manual_action"
        _issue(
            issues,
            f"stages.{label}",
            "manifest_blocker",
            f"run has unresolved blocker: {safe_code}",
            # Downstream blockers never invalidate the evidence they consume.
            blocking=not (phase == "evidence" and label in {"render", "review"}),
        )


def _parse_event_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("missing timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("naive timestamp")
    return parsed


def _validate_event(
    run_dir: Path, manifest: Mapping[str, Any], issues: list[ValidationIssue]
) -> None:
    event = manifest.get("inferred_event")
    if not isinstance(event, Mapping) or not event:
        return
    if event.get("status") != "resolved":
        _issue(issues, "inferred_event", "event_unresolved", "a recorded event time must be resolved")
        return
    local_date = event.get("local_date")
    dated_name = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_.+", run_dir.name)
    if dated_name is not None and local_date != dated_name.group(1):
        _issue(
            issues,
            "inferred_event.local_date",
            "run_date_mismatch",
            "resolved local date does not match the dated run directory",
            blocking=False,
        )
    try:
        start = _parse_event_timestamp(event.get("start_utc"))
        end = _parse_event_timestamp(event.get("end_utc"))
        record = manifest.get("input")
        media = record.get("media") if isinstance(record, Mapping) else None
        duration = media.get("duration_seconds") if isinstance(media, Mapping) else None
        if isinstance(duration, (int, float)) and abs((end - start).total_seconds() - float(duration)) > 0.25:
            _issue(issues, "inferred_event", "event_duration_mismatch", "event window does not cover the recording duration")
    except (TypeError, ValueError):
        _issue(issues, "inferred_event", "event_timestamp_invalid", "event start and end must be timezone-aware timestamps")


def _validate_final(
    run_dir: Path,
    manifest: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(manifest, Mapping):
        return
    registered_paths = {
        record.get("path")
        for record in (
            manifest.get("artifacts", {}).values()
            if isinstance(manifest.get("artifacts"), Mapping)
            else ()
        )
        if isinstance(record, Mapping)
    }
    for filename in (FLAC_NAME, *PUBLIC_OUTPUTS, REVIEW_NAME):
        if not (run_dir / filename).is_file():
            _issue(issues, filename, "required_artifact_missing", "required output is missing")
        elif filename not in registered_paths:
            _issue(issues, filename, "artifact_unregistered", "output is not hash-bound in the manifest")
    for filename in _EVIDENCE_FILES.values():
        if (run_dir / ".work" / filename).is_file() and _registered(
            manifest, "evidence", f".work/{filename}"
        ) is None:
            _issue(issues, f".work/{filename}", "artifact_unregistered", "reviewed evidence must be hash-bound by the evidence stage")

    review_path = run_dir / REVIEW_NAME
    blockers = _unresolved_manifest_blockers(manifest)
    if review_path.is_file():
        from .render import review_is_ready

        ready = review_is_ready(review_path.read_text(encoding="utf-8"))
        if ready and blockers:
            _issue(issues, REVIEW_NAME, "ready_with_blockers", "review cannot be marked ready while blockers remain")
        if not ready and not blockers:
            _issue(issues, REVIEW_NAME, "not_ready_without_blocker", "a not-ready review requires an unresolved manifest blocker")

    _validate_event(run_dir, manifest, issues)

    try:
        from .render import collect_review_items

        source_issues = [issue.to_dict() for issue in issues if issue.blocking]
        bundle = build_render_bundle(
            run_dir,
            manifest=manifest,
            artifact_validation={"valid": not source_issues, "issues": source_issues},
        )
        for item in collect_review_items(bundle):
            if item.get("blocking") and item.get("code") != "artifact_validation_failed":
                _issue(
                    issues,
                    ".work",
                    str(item.get("code", "review_item")),
                    "an unresolved review item blocks final outputs",
                )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError):
        _issue(issues, ".work", "render_source_invalid", "reviewed sources cannot produce safe final outputs")


def _contains_secret(value: str, known_secrets: Sequence[str]) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS) or any(
        secret in value for secret in known_secrets
    )


def _contains_binary_safe_secret(value: bytes) -> bool:
    return any(pattern.search(value) for pattern in _BINARY_SAFE_SECRET_BYTE_PATTERNS)


def _is_known_binary_media(prefix: bytes) -> bool:
    return any(prefix.startswith(magic) for magic in _BINARY_MEDIA_MAGIC)


def _redact_known_secret_issues(
    issues: Sequence[ValidationIssue], known_secrets: Sequence[str]
) -> list[ValidationIssue]:
    """Final defense so caller-provided secret values never escape in a report."""

    safe: list[ValidationIssue] = []
    for issue in issues:
        path = issue.path
        code = issue.code
        message = issue.message
        if any(secret in path for secret in known_secrets):
            path = _REDACTED_MANAGED_PATH
        if any(secret in code for secret in known_secrets):
            code = "redacted_issue"
        for secret in known_secrets:
            message = message.replace(secret, "[REDACTED]")
        safe.append(ValidationIssue(path=path, code=code, message=message, blocking=issue.blocking))
    return safe


def _secret_scan(
    run_dir: Path,
    known_secrets: Sequence[str],
    issues: list[ValidationIssue],
) -> None:
    secrets = tuple(value for value in known_secrets if isinstance(value, str) and value)
    encoded_secrets = tuple(secret.encode("utf-8") for secret in secrets)
    try:
        candidates: Iterable[Path] = run_dir.rglob("*")
        for path in candidates:
            relative = _relative(run_dir, path)
            path_has_secret = _contains_secret(relative, secrets)
            issue_path = _REDACTED_MANAGED_PATH if path_has_secret else relative
            # rglob does not follow directory symlinks; every symlink leaf is
            # rejected explicitly instead of being resolved.
            if path.is_symlink():
                _issue(issues, issue_path, "unsafe_path", "managed path is a symlink")
                continue
            if path_has_secret:
                _issue(issues, issue_path, "secret_detected", "credential-like material appears in a managed path")
            if not path.is_file():
                continue
            try:
                with path.open("rb") as source:
                    prefix = source.read(16)
                    binary_media = _is_known_binary_media(prefix)
                    source.seek(0)
                    overlap = b""
                    secret_found = False
                    if binary_media:
                        # Pipeline-generated media: scan only the metadata
                        # prefix for explicit credential syntax. Exact
                        # caller-supplied secrets still get a full byte scan.
                        if _contains_binary_safe_secret(source.read(_BINARY_CREDENTIAL_PREFIX_BYTES)):
                            secret_found = True
                        if not encoded_secrets:
                            if secret_found:
                                _issue(issues, issue_path, "secret_detected", "credential-like material was detected")
                            continue
                        source.seek(0)
                    while payload := source.read(_SECRET_SCAN_CHUNK_BYTES):
                        window = overlap + payload
                        if any(secret in window for secret in encoded_secrets):
                            secret_found = True
                            break
                        if not binary_media and _contains_secret(window.decode("utf-8", errors="ignore"), ()):
                            secret_found = True
                            break
                        overlap = window[-_SECRET_SCAN_OVERLAP_BYTES:]
            except OSError:
                _issue(issues, issue_path, "file_unreadable", "managed file could not be scanned")
                continue
            if secret_found:
                _issue(issues, issue_path, "secret_detected", "credential-like material was detected")
    except OSError:
        _issue(issues, ".", "run_unreadable", "run directory could not be scanned")


def validate_run(
    run_dir: str | os.PathLike[str],
    phase: ValidationPhase = "final",
    known_secrets: Sequence[str] = (),
) -> ValidationReport:
    """Validate one run without mutating it or returning secret values."""

    if phase not in {"evidence", "final", "all"}:
        raise ValueError("phase must be evidence, final, or all")
    directory = Path(run_dir).expanduser().resolve(strict=False)
    issues: list[ValidationIssue] = []
    if not directory.is_dir():
        _issue(issues, ".", "run_missing", "run directory does not exist")
        return ValidationReport(False, phase, tuple(issues), tuple(issues))

    secrets = tuple(known_secrets)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and hf_token.strip():
        secrets = (*secrets, hf_token.strip())

    manifest = _manifest_document(directory, issues)
    if manifest is not None:
        _validate_hashes(directory, manifest, issues)
        _validate_source_bindings(directory, manifest, issues)
    _surface_manifest_blockers(manifest, issues, phase=phase)
    _validate_flac(directory, manifest, issues)
    _validate_evidence(directory, manifest, issues)
    if phase in {"final", "all"}:
        _validate_final(directory, manifest, issues)
    _secret_scan(directory, secrets, issues)
    safe_issues = _redact_known_secret_issues(
        issues, tuple(value for value in secrets if isinstance(value, str) and value)
    )
    blockers = tuple(issue for issue in safe_issues if issue.blocking)
    return ValidationReport(not blockers, phase, tuple(safe_issues), blockers)


__all__ = [
    "SPEAKER_CONFIRMATION",
    "ValidationIssue",
    "ValidationReport",
    "build_render_bundle",
    "confirmed_speaker_names",
    "validate_run",
]
