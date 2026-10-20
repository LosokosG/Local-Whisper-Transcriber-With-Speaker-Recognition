"""Time-aligned, non-authoritative visual evidence from screen recordings.

All media coordinates in this module are seconds.  In particular, frame selection
never converts timestamps to frame numbers, so variable-frame-rate recordings do
not require a guessed FPS.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .redaction import redact_text

VISUAL_EVIDENCE_NAME = "visual_evidence.json"
DEFAULT_PERIODIC_INTERVAL_SECONDS = 45.0
DEFAULT_NEIGHBORHOODS_SECONDS = (2.0, 5.0)
DEFAULT_MAX_SAME_SPEAKER_GAP_SECONDS = 5.0
_DIAGNOSTIC_LIMIT = 500
_MANAGED_RAW_FRAME = re.compile(r"frame_\d{9,}\.png\Z")
_MANAGED_OCR_CROP = re.compile(r"frame_\d{9,}__[A-Za-z0-9_]+_ocr2x\.png\Z")
_REGION_NAME = re.compile(r"[A-Za-z0-9_]{1,64}\Z")
_MANAGED_CONTACT_SHEET = re.compile(r"contact_sheet_\d{3,}\.jpg\Z")


class FrameExtractionError(RuntimeError):
    """Raised when no requested visual target could be extracted."""

    def __init__(self, message: str, records: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.records = records


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _required_positive_number(value: object, name: str) -> float:
    result = _number(value)
    if result is None or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _first_number(record: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        if name in record:
            return _number(record[name])
    return None


def _source_ref(record: Mapping[str, object], fallback: str) -> str:
    for name in ("source_ref", "id", "event_id", "segment_id"):
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:240]
    return fallback


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def merge_adjacent_speaker_segments(
    segments: Iterable[Mapping[str, object]],
    *,
    max_gap_seconds: float = DEFAULT_MAX_SAME_SPEAKER_GAP_SECONDS,
) -> list[dict[str, object]]:
    """Compact consecutive diarization subsegments into speaker turns.

    WhisperX commonly emits many aligned text fragments for one uninterrupted
    speaker. Scheduling every fragment as a separate visual turn creates
    thousands of nearly identical frames, so a speaker change or a material
    silence starts a new visual turn. The boundary source references remain
    available for evidence traceability.
    """

    gap_limit = _number(max_gap_seconds)
    if gap_limit is None or gap_limit < 0:
        raise ValueError("maximum speaker-turn gap must be finite and non-negative")
    turns: list[dict[str, object]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            continue
        start = _first_number(segment, "start", "start_seconds", "timestamp")
        end = _first_number(segment, "end", "end_seconds")
        if start is None or end is None or end < start:
            continue
        raw_speaker = segment.get("speaker")
        speaker = raw_speaker.strip() if isinstance(raw_speaker, str) else ""
        reference = _source_ref(segment, f"segment:{index}")

        previous = turns[-1] if turns else None
        same_continuous_speaker = bool(
            previous is not None
            and speaker
            and previous["speaker"] == speaker
            and start - float(previous["end"]) <= gap_limit
        )
        if same_continuous_speaker:
            assert previous is not None
            previous["end"] = max(float(previous["end"]), end)
            previous["segment_count"] = int(previous["segment_count"]) + 1
            refs = previous["source_refs"]
            assert isinstance(refs, list)
            if reference != refs[-1]:
                if len(refs) == 1:
                    refs.append(reference)
                else:
                    refs[-1] = reference
            previous["id"] = f"turn:{refs[0]}:{refs[-1]}"
            continue

        turns.append(
            {
                "id": f"turn:{reference}:{reference}",
                "start": start,
                "end": end,
                "speaker": speaker or None,
                "source_refs": [reference],
                "segment_count": 1,
            }
        )
    return turns


def _referenced_managed_name(
    value: object,
    directory: Path,
    pattern: re.Pattern[str],
) -> str | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    candidate = Path(os.path.abspath(Path(value)))
    if candidate.parent != Path(os.path.abspath(directory)):
        return None
    return candidate.name if pattern.fullmatch(candidate.name) else None


def _prune_directory(
    directory: Path,
    pattern: re.Pattern[str],
    retained_names: set[str],
) -> int:
    for name in retained_names:
        retained = directory / name
        if retained.is_symlink() or not retained.is_file():
            raise ValueError("referenced visual derivative must be a real managed file")
    removed = 0
    for entry in directory.iterdir():
        if not pattern.fullmatch(entry.name) or entry.name in retained_names:
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
            removed += 1
    if removed:
        directory_fd = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return removed


def prune_unreferenced_visual_derivatives(
    document: Mapping[str, object],
    frame_directory: str | os.PathLike[str],
    *,
    managed_root: str | os.PathLike[str] | None = None,
) -> dict[str, int]:
    """Remove obsolete, recognizably pipeline-owned visual leaf artifacts.

    Unknown files and directories are preserved. Symlink leaves are unlinked,
    never followed. Call this only after the replacement visual-evidence
    document has been written successfully.
    """

    frames = _prepare_output_directory(
        frame_directory, managed_root=managed_root
    )
    ocr = _prepare_output_directory(frames / "ocr", managed_root=managed_root)
    sheets = _prepare_output_directory(
        frames / "contact_sheets", managed_root=managed_root
    )
    retained_raw: set[str] = set()
    retained_ocr: set[str] = set()
    retained_sheets: set[str] = set()
    sources = document.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            raw_name = _referenced_managed_name(
                source.get("raw_path"), frames, _MANAGED_RAW_FRAME
            )
            if raw_name is not None:
                retained_raw.add(raw_name)
            ocr_records = source.get("ocr")
            if not isinstance(ocr_records, list):
                continue
            for record in ocr_records:
                if not isinstance(record, Mapping):
                    continue
                crop_name = _referenced_managed_name(
                    record.get("crop_path"), ocr, _MANAGED_OCR_CROP
                )
                if crop_name is not None:
                    retained_ocr.add(crop_name)
    contact_sheets = document.get("contact_sheets")
    if isinstance(contact_sheets, list):
        for value in contact_sheets:
            sheet_name = _referenced_managed_name(
                value, sheets, _MANAGED_CONTACT_SHEET
            )
            if sheet_name is not None:
                retained_sheets.add(sheet_name)
    return {
        "raw_frames": _prune_directory(
            frames, _MANAGED_RAW_FRAME, retained_raw
        ),
        "ocr_crops": _prune_directory(ocr, _MANAGED_OCR_CROP, retained_ocr),
        "contact_sheets": _prune_directory(
            sheets, _MANAGED_CONTACT_SHEET, retained_sheets
        ),
    }


def select_frame_targets(
    duration_seconds: float,
    segments: Iterable[Mapping[str, object]] = (),
    mentions: Iterable[Mapping[str, object]] = (),
    events: Iterable[Mapping[str, object]] = (),
    *,
    periodic_interval: float = DEFAULT_PERIODIC_INTERVAL_SECONDS,
    neighborhoods: Sequence[float] = DEFAULT_NEIGHBORHOODS_SECONDS,
    dedupe_tolerance: float = 0.05,
    boundary_adjacency: float = 0.05,
) -> list[dict[str, object]]:
    """Return stable, deduplicated second-based visual sampling targets.

    Invalid individual source records are ignored.  Finite source timestamps
    outside the recording are clamped to its bounds, while invalid top-level
    configuration is rejected because a safe selection cannot be derived.
    """

    duration = _required_positive_number(duration_seconds, "duration")
    interval = _required_positive_number(periodic_interval, "periodic interval")
    if not 30 <= interval <= 60:
        raise ValueError("periodic interval must be between 30 and 60 seconds")
    tolerance = _number(dedupe_tolerance)
    adjacency = _number(boundary_adjacency)
    if tolerance is None or tolerance < 0:
        raise ValueError("dedupe tolerance must be a finite non-negative number")
    if adjacency is None or adjacency < 0:
        raise ValueError("boundary adjacency must be a finite non-negative number")

    neighbor_values: list[float] = []
    for value in neighborhoods:
        parsed = _number(value)
        if parsed is None or parsed <= 0:
            raise ValueError("neighborhoods must contain positive finite seconds")
        if parsed not in neighbor_values:
            neighbor_values.append(parsed)
    neighbor_values.sort()

    candidates: list[dict[str, object]] = []

    def add(
        timestamp: float,
        reason: str,
        source_ref: str | Sequence[str],
        priority: int,
    ) -> None:
        clamped = min(duration, max(0.0, timestamp))
        refs = [source_ref] if isinstance(source_ref, str) else list(source_ref)
        candidates.append(
            {
                "timestamp_seconds": round(clamped, 6),
                "reasons": [reason],
                "source_refs": [value for value in refs if value],
                "priority": priority,
            }
        )

    periodic = 0.0
    while periodic < duration:
        add(periodic, "periodic_baseline", "", 10)
        periodic += interval
    add(duration, "periodic_baseline", "", 10)

    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            continue
        start = _first_number(segment, "start", "start_seconds", "timestamp")
        end = _first_number(segment, "end", "end_seconds")
        if start is None or end is None or end < start:
            continue
        supplied_refs = segment.get("source_refs")
        reference: str | Sequence[str]
        if isinstance(supplied_refs, list) and all(
            isinstance(value, str) and value for value in supplied_refs
        ):
            reference = supplied_refs
        else:
            reference = _source_ref(segment, f"segment:{index}")
        midpoint = start + (end - start) / 2
        add(start, "turn_start", reference, 80)
        if end > start:
            add(min(end, start + adjacency), "turn_start_adjacent", reference, 75)
        for neighbor in reversed(neighbor_values):
            add(midpoint - neighbor, f"turn_midpoint_minus_{neighbor:g}s", reference, 70)
        add(midpoint, "turn_midpoint", reference, 85)
        for neighbor in neighbor_values:
            add(midpoint + neighbor, f"turn_midpoint_plus_{neighbor:g}s", reference, 70)
        if end > start:
            add(max(start, end - adjacency), "turn_end_adjacent", reference, 75)
        add(end, "turn_end", reference, 80)

    def add_event_records(
        records: Iterable[Mapping[str, object]], reason: str, base_priority: int
    ) -> None:
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            timestamp = _first_number(
                record,
                "timestamp_seconds",
                "timestamp",
                "time_seconds",
                "start",
            )
            if timestamp is None:
                continue
            reference = _source_ref(record, f"{reason}:{index}")
            record_reason = reason
            if reason == "event":
                supplied_reason = record.get("reason", record.get("type"))
                if isinstance(supplied_reason, str) and supplied_reason.strip():
                    record_reason = redact_text(supplied_reason.strip())[:120]
            for neighbor in reversed(neighbor_values):
                add(
                    timestamp - neighbor,
                    f"{record_reason}_minus_{neighbor:g}s",
                    reference,
                    base_priority - 10,
                )
            add(timestamp, record_reason, reference, base_priority)
            for neighbor in neighbor_values:
                add(
                    timestamp + neighbor,
                    f"{record_reason}_plus_{neighbor:g}s",
                    reference,
                    base_priority - 10,
                )

    add_event_records(mentions, "spoken_name", 100)
    add_event_records(events, "event", 95)

    candidates.sort(key=lambda item: float(item["timestamp_seconds"]))
    merged: list[dict[str, object]] = []
    for candidate in candidates:
        if (
            merged
            and float(candidate["timestamp_seconds"])
            - float(merged[-1]["timestamp_seconds"])
            <= tolerance + 1e-9
        ):
            previous = merged[-1]
            for reason in candidate["reasons"]:  # type: ignore[union-attr]
                _append_unique(previous["reasons"], reason)  # type: ignore[arg-type]
            for reference in candidate["source_refs"]:  # type: ignore[union-attr]
                _append_unique(previous["source_refs"], reference)  # type: ignore[arg-type]
            if int(candidate["priority"]) > int(previous["priority"]):
                previous["timestamp_seconds"] = candidate["timestamp_seconds"]
            previous["priority"] = max(
                int(previous["priority"]), int(candidate["priority"])
            )
        else:
            merged.append(candidate)
    return merged


def _bounded_error(value: object) -> str:
    if isinstance(value, BaseException):
        rendered = f"{type(value).__name__}: {value}"
    else:
        rendered = str(value)
    safe = redact_text(rendered.replace("\x00", ""))
    if len(safe) > _DIAGNOSTIC_LIMIT:
        return safe[: _DIAGNOSTIC_LIMIT - 1] + "…"
    return safe


def _frame_name(timestamp_seconds: float) -> str:
    milliseconds = max(0, round(timestamp_seconds * 1000))
    return f"frame_{milliseconds:09d}.png"


def _frame_id(timestamp_seconds: float) -> str:
    return _frame_name(timestamp_seconds).removesuffix(".png")


def _valid_existing_png(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.width <= 0 or opened.height <= 0:
                return False
            opened.verify()
    except (OSError, ValueError, SyntaxError):
        return False
    return True


def _prepare_output_directory(
    output_directory: str | os.PathLike[str],
    *,
    managed_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Create one real output directory contained by its managed root."""

    output = Path(output_directory)
    if output.is_symlink():
        raise ValueError("managed output directory must not be a symlink")
    root = Path(managed_root) if managed_root is not None else output.parent
    if managed_root is None:
        while not root.exists() and root != root.parent:
            root = root.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError("managed root must be a real directory, not a symlink")
    root_resolved = root.resolve(strict=True)
    try:
        output.resolve(strict=False).relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("managed output directory resolves outside its root") from exc

    root_absolute = Path(os.path.abspath(root))
    output_absolute = Path(os.path.abspath(output))
    try:
        relative = output_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError("managed output directory is outside its root") from exc
    current = root_absolute
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("managed output directory must not contain symlinks")
        if current.exists() and not current.is_dir():
            raise ValueError("managed output path must be a directory")

    output.mkdir(parents=True, exist_ok=True)
    try:
        output.resolve(strict=True).relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("managed output directory escaped its root") from exc
    return output


def _save_image_atomic(
    image: Image.Image,
    destination: Path,
    *,
    image_format: str,
    **save_options: object,
) -> None:
    """Persist one managed image without following or partially writing a leaf."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            image.save(temporary, format=image_format, **save_options)
            temporary.flush()
            os.fsync(temporary.fileno())
        # Replacing the directory entry atomically replaces a managed leaf
        # symlink itself; it never opens or writes through the symlink target.
        os.replace(temporary_path, destination)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_frames(
    video_path: str | os.PathLike[str],
    targets: Iterable[Mapping[str, object]],
    output_directory: str | os.PathLike[str],
    *,
    ffmpeg_binary: str = "ffmpeg",
    managed_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract full-resolution PNGs, retaining per-target failures."""

    source = Path(video_path)
    output = _prepare_output_directory(
        output_directory, managed_root=managed_root
    )
    records: list[dict[str, Any]] = []

    for target in targets:
        target_copy = dict(target) if isinstance(target, Mapping) else {}
        timestamp = _number(target_copy.get("timestamp_seconds"))
        base: dict[str, Any] = {
            "timestamp_seconds": timestamp,
            "reasons": list(target_copy.get("reasons", [])),
            "source_refs": list(target_copy.get("source_refs", [])),
            "priority": int(target_copy.get("priority", 0)),
            "raw_path": None,
        }
        if timestamp is not None and timestamp >= 0:
            base["frame_id"] = _frame_id(timestamp)
        if timestamp is None or timestamp < 0:
            base.update(status="error", error="invalid target timestamp")
            records.append(base)
            continue

        destination = output / _frame_name(timestamp)
        if _valid_existing_png(destination):
            base.update(status="ok", raw_path=str(destination))
            records.append(base)
            continue
        if destination.is_symlink() or destination.exists():
            destination.unlink(missing_ok=True)
        command = [
            ffmpeg_binary,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-an",
            "-sn",
            str(destination),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr or "ffmpeg returned an error")
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError("ffmpeg did not create a non-empty frame")
        except (OSError, RuntimeError) as exc:
            destination.unlink(missing_ok=True)
            base.update(status="error", error=_bounded_error(exc))
            records.append(base)
            continue

        base.update(status="ok", raw_path=str(destination))
        records.append(base)

    if not records:
        raise ValueError("at least one frame target is required")
    failure_count = sum(record["status"] == "error" for record in records)
    if failure_count == len(records):
        raise FrameExtractionError(
            f"all {failure_count} frame extractions failed", records
        )
    return records


DEFAULT_OCR_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "full": (0.0, 0.0, 1.0, 1.0),
}


def _validated_regions(
    regions: Mapping[str, Sequence[float]] | None,
) -> dict[str, tuple[float, float, float, float]]:
    selected = DEFAULT_OCR_REGIONS if regions is None else regions
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError("OCR regions must be a non-empty mapping")
    result: dict[str, tuple[float, float, float, float]] = {}
    for name, ratios in selected.items():
        if not isinstance(name, str) or _REGION_NAME.fullmatch(name) is None:
            raise ValueError("OCR region names must match [A-Za-z0-9_]{1,64}")
        values = tuple(_number(value) for value in ratios)
        if len(values) != 4 or any(value is None for value in values):
            raise ValueError(f"OCR region {name!r} needs four finite ratios")
        left, top, right, bottom = (float(value) for value in values)  # type: ignore[arg-type]
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError(
                f"OCR region {name!r} must satisfy 0 <= left < right <= 1 "
                "and 0 <= top < bottom <= 1"
            )
        result[name] = (left, top, right, bottom)
    return result


def get_crop_specs(
    width: int,
    height: int,
    regions: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    """Scale ratio-based OCR regions to pixel boxes clamped to image bounds.

    Regions are ``name -> (left, top, right, bottom)`` fractions of the frame.
    The default is one ``full`` region. Callers that know their recording's
    layout (for example a slide pane or a participant list) can pass narrower
    regions so OCR reads less noise.
    """

    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive integers")
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for region, ratios in _validated_regions(regions).items():
        left = min(width - 1, max(0, round(width * ratios[0])))
        top = min(height - 1, max(0, round(height * ratios[1])))
        right = min(width, max(left + 1, round(width * ratios[2])))
        bottom = min(height, max(top + 1, round(height * ratios[3])))
        boxes[region] = (left, top, right, bottom)
    return boxes


def create_ocr_crops(
    raw_frame_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    regions: Mapping[str, Sequence[float]] | None = None,
    managed_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Create non-destructive two-times OCR derivatives from a raw frame."""

    raw_path = Path(raw_frame_path)
    output = _prepare_output_directory(
        output_directory, managed_root=managed_root
    )
    crops: list[dict[str, Any]] = []
    with Image.open(raw_path) as opened:
        specs = get_crop_specs(opened.width, opened.height, regions)
        for region, box in specs.items():
            cropped = opened.crop(box)
            upscaled = cropped.resize(
                (cropped.width * 2, cropped.height * 2), Image.Resampling.LANCZOS
            )
            destination = output / f"{raw_path.stem}__{region}_ocr2x.png"
            _save_image_atomic(upscaled, destination, image_format="PNG")
            crops.append(
                {
                    "region": region,
                    "box": list(box),
                    "scale": 2,
                    "path": str(destination),
                }
            )
    return crops


def _parse_tesseract_tsv(value: str) -> tuple[str, float | None]:
    words: list[str] = []
    confidences: list[float] = []
    try:
        rows = csv.DictReader(io.StringIO(value), delimiter="\t")
        for row in rows:
            word = (row.get("text") or "").strip()
            confidence = _number(row.get("conf"))
            if not word:
                continue
            words.append(word)
            if confidence is not None and 0 <= confidence <= 100:
                confidences.append(confidence)
    except (csv.Error, TypeError):
        return "", None
    mean = sum(confidences) / len(confidences) / 100 if confidences else None
    return " ".join(words), round(mean, 6) if mean is not None else None


def run_tesseract_ocr(
    crops: Iterable[Mapping[str, object]],
    *,
    tesseract_binary: str = "tesseract",
) -> list[dict[str, Any]]:
    """Run TSV OCR independently for each crop and retain crop-level failures."""

    results: list[dict[str, Any]] = []
    for crop in crops:
        path = Path(str(crop.get("path", "")))
        base: dict[str, Any] = {
            "region": crop.get("region"),
            "crop_path": str(path),
        }
        command = [tesseract_binary, str(path), "stdout", "tsv"]
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr or "tesseract returned an error")
            text, confidence = _parse_tesseract_tsv(completed.stdout)
            base.update(
                status="ok",
                text=text,
                confidence=confidence,
                error=None,
            )
        except (OSError, RuntimeError) as exc:
            base.update(
                status="error",
                text="",
                confidence=None,
                error=_bounded_error(exc),
            )
        results.append(base)
    return results


def _timestamp_label(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def create_contact_sheets(
    frame_records: Iterable[Mapping[str, object]],
    output_directory: str | os.PathLike[str],
    *,
    columns: int = 3,
    page_size: int = 12,
    thumbnail_size: tuple[int, int] = (320, 180),
    managed_root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Create labeled, paginated sheets from raw frames without modifying them."""

    if columns <= 0 or page_size <= 0:
        raise ValueError("columns and page size must be positive")
    if thumbnail_size[0] <= 0 or thumbnail_size[1] <= 0:
        raise ValueError("thumbnail dimensions must be positive")
    output = _prepare_output_directory(
        output_directory, managed_root=managed_root
    )
    usable = [
        dict(record)
        for record in frame_records
        if record.get("status") == "ok"
        and isinstance(record.get("raw_path"), (str, os.PathLike))
        and Path(record["raw_path"]).is_file()  # type: ignore[arg-type]
    ]
    if not usable:
        return []

    label_height = 44
    cell_width, thumb_height = thumbnail_size
    cell_height = thumb_height + label_height
    rows = math.ceil(page_size / columns)
    sheet_paths: list[str] = []

    for page_index, offset in enumerate(range(0, len(usable), page_size), start=1):
        page_records = usable[offset : offset + page_size]
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        drawer = ImageDraw.Draw(sheet)
        for cell_index, record in enumerate(page_records):
            x = (cell_index % columns) * cell_width
            y = (cell_index // columns) * cell_height
            with Image.open(record["raw_path"]) as raw:
                thumbnail = ImageOps.contain(
                    raw.convert("RGB"), thumbnail_size, Image.Resampling.LANCZOS
                )
            image_x = x + (cell_width - thumbnail.width) // 2
            image_y = y + (thumb_height - thumbnail.height) // 2
            sheet.paste(thumbnail, (image_x, image_y))
            timestamp = _number(record.get("timestamp_seconds")) or 0.0
            reasons = ", ".join(str(value) for value in record.get("reasons", []))
            drawer.text((x + 4, y + thumb_height + 3), _timestamp_label(timestamp), fill="black")
            drawer.text(
                (x + 4, y + thumb_height + 20),
                reasons[: max(1, cell_width // 6)],
                fill="black",
            )
        destination = output / f"contact_sheet_{page_index:03d}.jpg"
        _save_image_atomic(
            sheet,
            destination,
            image_format="JPEG",
            quality=90,
        )
        sheet_paths.append(str(destination))
    return sheet_paths


PRESENCE_EVIDENCE_TYPES = frozenset({"speaking_indicator", "visible_label", "OCR"})


@dataclass(frozen=True, slots=True)
class PresenceObservation:
    """A local visual observation, not an identity decision."""

    timestamp_seconds: float
    region: str
    evidence_type: str
    confidence: float
    exact_visible_name: str | None = None
    observed_text: str | None = None
    truncated: bool = False
    canonical_name: str | None = None
    source_ref: str | None = None
    id: str | None = None
    frame_id: str | None = None

    def __post_init__(self) -> None:
        timestamp = _number(self.timestamp_seconds)
        confidence = _number(self.confidence)
        if timestamp is None or timestamp < 0:
            raise ValueError("observation timestamp must be finite and non-negative")
        if confidence is None or not 0 <= confidence <= 1:
            raise ValueError("observation confidence must be between zero and one")
        if not isinstance(self.region, str) or _REGION_NAME.fullmatch(self.region) is None:
            raise ValueError("presence region must match [A-Za-z0-9_]{1,64}")
        if self.evidence_type not in PRESENCE_EVIDENCE_TYPES:
            raise ValueError("unknown presence evidence type")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _observation_dict(value: PresenceObservation | Mapping[str, object]) -> dict[str, Any] | None:
    if isinstance(value, PresenceObservation):
        return value.to_dict()
    if not isinstance(value, Mapping):
        return None
    timestamp = _number(value.get("timestamp_seconds"))
    confidence = _number(value.get("confidence"))
    if timestamp is None or timestamp < 0 or confidence is None or not 0 <= confidence <= 1:
        return None
    return {
        "timestamp_seconds": timestamp,
        "exact_visible_name": value.get("exact_visible_name"),
        "observed_text": value.get("observed_text"),
        "truncated": bool(value.get("truncated", False)),
        "region": value.get("region"),
        "evidence_type": value.get("evidence_type"),
        "confidence": confidence,
        "canonical_name": value.get("canonical_name"),
        "source_ref": value.get("source_ref"),
        "id": value.get("id"),
        "frame_id": value.get("frame_id"),
    }


def _reliable_presence_key(observation: Mapping[str, object]) -> str | None:
    if observation.get("truncated") or observation.get("evidence_type") in {"OCR", "absence"}:
        return None
    exact = observation.get("exact_visible_name")
    if not isinstance(exact, str) or not exact.strip():
        return None
    return " ".join(exact.split()).casefold()


def _isolated(observation: Mapping[str, object], index: int) -> dict[str, Any]:
    return {
        "timestamp_seconds": float(observation["timestamp_seconds"]),
        "canonical_name": observation.get("canonical_name"),
        "exact_visible_name": observation.get("exact_visible_name"),
        "observed_text": observation.get("observed_text"),
        "truncated": bool(observation.get("truncated")),
        "region": observation.get("region"),
        "evidence_type": observation.get("evidence_type"),
        "confidence": min(float(observation["confidence"]), 0.49),
        "observation_index": index,
    }


def build_presence_intervals(
    observations: Iterable[PresenceObservation | Mapping[str, object]],
    *,
    max_gap_seconds: float = 15.0,
) -> dict[str, list[dict[str, Any]]]:
    """Build conservative intervals from repeated, reliable name observations.

    A missing name is never interpreted as departure.  Ambiguous, truncated, or
    OCR-only evidence stays isolated for later contradiction-aware resolution.
    """

    gap = _required_positive_number(max_gap_seconds, "maximum presence gap")
    normalized: list[tuple[int, dict[str, Any]]] = []
    for original_index, value in enumerate(observations):
        observation = _observation_dict(value)
        if observation is None or observation.get("evidence_type") == "absence":
            continue
        normalized.append((original_index, observation))
    normalized.sort(key=lambda item: (float(item[1]["timestamp_seconds"]), item[0]))

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    isolated: list[dict[str, Any]] = []
    for index, observation in normalized:
        key = _reliable_presence_key(observation)
        if key is None:
            isolated.append(_isolated(observation, index))
        else:
            grouped.setdefault(key, []).append((index, observation))

    intervals: list[dict[str, Any]] = []
    for key, items in grouped.items():
        chunk: list[tuple[int, dict[str, Any]]] = []

        def finish_chunk() -> None:
            if len(chunk) < 2:
                index, observation = chunk[0]
                isolated.append(_isolated(observation, index))
                return
            confidence = sum(float(item[1]["confidence"]) for item in chunk) / len(chunk)
            canonical_names = {
                canonical.strip().casefold()
                for _, observation in chunk
                if isinstance((canonical := observation.get("canonical_name")), str)
                and canonical.strip()
            }
            intervals.append(
                {
                    "exact_visible_name": key,
                    "canonical_name": (
                        next(iter(canonical_names))
                        if len(canonical_names) == 1
                        else None
                    ),
                    "start_seconds": float(chunk[0][1]["timestamp_seconds"]),
                    "end_seconds": float(chunk[-1][1]["timestamp_seconds"]),
                    "confidence": round(confidence, 6),
                    "observation_count": len(chunk),
                    "observation_indices": [item[0] for item in chunk],
                }
            )

        for item in items:
            if chunk and (
                float(item[1]["timestamp_seconds"])
                - float(chunk[-1][1]["timestamp_seconds"])
                > gap
            ):
                finish_chunk()
                chunk = []
            chunk.append(item)
        if chunk:
            finish_chunk()

    intervals.sort(
        key=lambda item: (
            item["start_seconds"],
            str(item.get("exact_visible_name") or item.get("canonical_name") or ""),
        )
    )
    isolated.sort(key=lambda item: (item["timestamp_seconds"], item["observation_index"]))
    return {"intervals": intervals, "isolated": isolated, "departures": []}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, PresenceObservation):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_visual_evidence(
    *,
    duration_seconds: float,
    frame_records: Iterable[Mapping[str, object]],
    presence_observations: Iterable[PresenceObservation | Mapping[str, object]] = (),
    presence_model: Mapping[str, object] | None = None,
    contact_sheets: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    """Build the JSON-ready `.work/frames` visual-evidence artifact."""

    duration = _required_positive_number(duration_seconds, "duration")
    sources: list[dict[str, Any]] = []
    for record in frame_records:
        source = _json_ready(dict(record))
        timestamp = _number(source.get("timestamp_seconds"))
        if timestamp is not None and timestamp >= 0:
            source["frame_id"] = _frame_id(timestamp)
        sources.append(source)
    observations = [_json_ready(value) for value in presence_observations]
    model = (
        _json_ready(dict(presence_model))
        if presence_model is not None
        else build_presence_intervals(observations)
    )
    errors = [
        {
            "frame_id": source.get("frame_id"),
            "timestamp_seconds": source.get("timestamp_seconds"),
            "error": source.get("error"),
        }
        for source in sources
        if source.get("status") == "error"
    ]
    document = {
        "schema_version": 1,
        "artifact_directory": ".work/frames",
        "duration_seconds": duration,
        "sources": sources,
        "contact_sheets": [str(path) for path in contact_sheets],
        "presence_observations": observations,
        "presence": model,
        "errors": errors,
    }
    json.dumps(document, allow_nan=False)
    return document


def attach_presence_observations(
    document: Mapping[str, object],
    observations: Iterable[Mapping[str, object]],
) -> dict[str, Any]:
    """Return visual evidence with reviewed observations bound to real frames.

    Observations are claims made by whoever looked at the frames (an agent or a
    human). Each one must name a unique ``id`` and an extracted ``frame_id``,
    and its timestamp must equal that frame's timestamp, so an observation can
    never float free of the image it describes.
    """

    sources = document.get("sources")
    frames_by_id = {
        str(source["frame_id"]): source
        for source in (sources if isinstance(sources, list) else [])
        if isinstance(source, Mapping)
        and source.get("status") == "ok"
        and isinstance(source.get("frame_id"), str)
    }
    attached: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(observations):
        if not isinstance(value, Mapping):
            raise ValueError(f"observation {index} must be an object")
        observation = PresenceObservation(
            timestamp_seconds=value.get("timestamp_seconds"),  # type: ignore[arg-type]
            region=value.get("region"),  # type: ignore[arg-type]
            evidence_type=value.get("evidence_type"),  # type: ignore[arg-type]
            confidence=value.get("confidence"),  # type: ignore[arg-type]
            exact_visible_name=value.get("exact_visible_name"),  # type: ignore[arg-type]
            observed_text=value.get("observed_text"),  # type: ignore[arg-type]
            truncated=bool(value.get("truncated", False)),
            canonical_name=value.get("canonical_name"),  # type: ignore[arg-type]
            source_ref=value.get("source_ref"),  # type: ignore[arg-type]
            id=value.get("id"),  # type: ignore[arg-type]
            frame_id=value.get("frame_id"),  # type: ignore[arg-type]
        )
        identifier = observation.id
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"observation {index} needs a unique non-empty id")
        seen.add(identifier)
        frame = frames_by_id.get(str(observation.frame_id))
        if frame is None:
            raise ValueError(f"observation {identifier} references an unknown frame")
        frame_time = _number(frame.get("timestamp_seconds"))
        if frame_time is None or abs(frame_time - observation.timestamp_seconds) > 0.001:
            raise ValueError(f"observation {identifier} timestamp contradicts its frame")
        attached.append(observation.to_dict())
    result = dict(_json_ready(dict(document)))
    result["presence_observations"] = attached
    result["presence"] = build_presence_intervals(attached)
    return result


def write_visual_evidence(
    output_path: str | os.PathLike[str], document: Mapping[str, object]
) -> None:
    """Atomically write strict, UTF-8 visual evidence JSON."""

    if not isinstance(document, Mapping):
        raise TypeError("visual evidence document must be a mapping")
    rendered = json.dumps(
        _json_ready(document),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path = Path(output_path)
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
            temporary.write(rendered)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_MAX_SAME_SPEAKER_GAP_SECONDS",
    "DEFAULT_NEIGHBORHOODS_SECONDS",
    "DEFAULT_OCR_REGIONS",
    "DEFAULT_PERIODIC_INTERVAL_SECONDS",
    "FrameExtractionError",
    "PRESENCE_EVIDENCE_TYPES",
    "PresenceObservation",
    "VISUAL_EVIDENCE_NAME",
    "attach_presence_observations",
    "build_presence_intervals",
    "build_visual_evidence",
    "create_contact_sheets",
    "create_ocr_crops",
    "extract_frames",
    "get_crop_specs",
    "merge_adjacent_speaker_segments",
    "prune_unreferenced_visual_derivatives",
    "run_tesseract_ocr",
    "select_frame_targets",
    "write_visual_evidence",
]
