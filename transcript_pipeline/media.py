"""Safe probing and canonical audio preparation for recordings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import TypedDict

from .redaction import redact_text

CANONICAL_FLAC_NAME = "audio.flac"
DIAGNOSTIC_LIMIT = 2048
_HASH_CHUNK_SIZE = 1024 * 1024
_DURATION_TOLERANCE_SECONDS = 0.25


class MediaError(RuntimeError):
    """Raised when input media cannot be safely prepared or verified."""


class FormatMetadata(TypedDict):
    name: str
    long_name: str | None
    duration_seconds: float


class StreamMetadata(TypedDict):
    index: int | None
    codec_type: str
    codec_name: str | None
    duration_seconds: float | None
    creation_time: str | None


class AudioMetadata(TypedDict):
    codec_name: str | None
    sample_rate: int | None
    channels: int | None
    duration_seconds: float | None
    creation_time: str | None


class VideoMetadata(TypedDict):
    codec_name: str | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    avg_frame_rate: str | None
    avg_frame_rate_fps: float | None
    r_frame_rate: str | None
    r_frame_rate_fps: float | None
    frame_rate_fps: float | None
    creation_time: str | None


class MediaMetadata(TypedDict):
    path: str
    duration_seconds: float
    format_name: str
    format: FormatMetadata
    streams: list[StreamMetadata]
    has_audio: bool
    audio_stream_count: int
    audio: AudioMetadata | None
    has_video: bool
    video_stream_count: int
    video: VideoMetadata | None
    creation_time: str | None
    creation_time_source: str | None


class CanonicalMediaArtifact(TypedDict):
    path: str
    sha256: str
    metadata: MediaMetadata
    source_audio_duration_seconds: float
    duration_difference_seconds: float


def _bounded_diagnostic(value: object) -> str:
    safe = redact_text(str(value)).strip()
    if not safe:
        safe = "no diagnostic output"
    if len(safe) > DIAGNOSTIC_LIMIT:
        return safe[: DIAGNOSTIC_LIMIT - 1] + "…"
    return safe


def _error(message: str, diagnostic: object | None = None) -> MediaError:
    if diagnostic is None:
        return MediaError(_bounded_diagnostic(message))
    return MediaError(_bounded_diagnostic(f"{message}: {diagnostic}"))


def _parse_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _parse_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _parse_rational(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", 1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)
            if denominator == 0:
                return None
            parsed = numerator / denominator
        else:
            parsed = float(text)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _string_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _tags(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _creation_time(value: object) -> str | None:
    tags = _tags(value)
    return _string_or_none(tags.get("creation_time"))


def _duration_from_streams(streams: list[dict[str, object]]) -> float | None:
    durations = [
        parsed
        for stream in streams
        if (parsed := _parse_float(stream.get("duration"))) is not None
    ]
    return max(durations, default=None)


def _run_probe(source: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
    except OSError as exc:
        raise _error("could not start ffprobe", exc) from exc
    if completed.returncode != 0:
        raise _error("ffprobe failed", completed.stderr)
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise _error("ffprobe returned malformed JSON", exc) from exc
    if not isinstance(payload, dict):
        raise _error("ffprobe returned malformed JSON", "root value is not an object")
    return payload


def probe_media(path: str | os.PathLike[str]) -> MediaMetadata:
    """Return normalized, JSON-serializable metadata from an ffprobe inspection."""

    source = Path(path)
    if not source.is_file():
        raise _error("input media does not exist or is not a file", source)

    payload = _run_probe(source)
    format_payload = payload.get("format")
    streams_payload = payload.get("streams")
    if not isinstance(format_payload, dict) or not isinstance(streams_payload, list):
        raise _error(
            "ffprobe returned unsupported metadata",
            "format or streams section is missing",
        )
    streams = [stream for stream in streams_payload if isinstance(stream, dict)]
    if len(streams) != len(streams_payload) or not streams:
        raise _error("ffprobe returned unsupported metadata", "no usable streams")

    format_name = _string_or_none(format_payload.get("format_name"))
    if format_name is None:
        raise _error("ffprobe returned unsupported metadata", "format name is missing")
    duration = _parse_float(format_payload.get("duration"))
    if duration is None:
        duration = _duration_from_streams(streams)
    if duration is None:
        raise _error("ffprobe returned unsupported metadata", "duration is missing")

    normalized_streams: list[StreamMetadata] = []
    audio_streams: list[dict[str, object]] = []
    video_streams: list[dict[str, object]] = []
    stream_creation_times: list[tuple[int | None, str]] = []
    for stream in streams:
        codec_type = _string_or_none(stream.get("codec_type")) or "unknown"
        creation_time = _creation_time(stream.get("tags"))
        index = _parse_int(stream.get("index"))
        normalized_streams.append(
            {
                "index": index,
                "codec_type": codec_type,
                "codec_name": _string_or_none(stream.get("codec_name")),
                "duration_seconds": _parse_float(stream.get("duration")),
                "creation_time": creation_time,
            }
        )
        if creation_time is not None:
            stream_creation_times.append((index, creation_time))
        if codec_type == "audio":
            audio_streams.append(stream)
        elif codec_type == "video":
            video_streams.append(stream)

    audio: AudioMetadata | None = None
    if audio_streams:
        first_audio = audio_streams[0]
        audio_duration = _parse_float(first_audio.get("duration"))
        audio = {
            "codec_name": _string_or_none(first_audio.get("codec_name")),
            "sample_rate": _parse_int(first_audio.get("sample_rate")),
            "channels": _parse_int(first_audio.get("channels")),
            "duration_seconds": duration if audio_duration is None else audio_duration,
            "creation_time": _creation_time(first_audio.get("tags")),
        }

    video: VideoMetadata | None = None
    if video_streams:
        first_video = video_streams[0]
        average_raw = _string_or_none(first_video.get("avg_frame_rate"))
        real_raw = _string_or_none(first_video.get("r_frame_rate"))
        average_fps = _parse_rational(average_raw)
        real_fps = _parse_rational(real_raw)
        video_duration = _parse_float(first_video.get("duration"))
        video = {
            "codec_name": _string_or_none(first_video.get("codec_name")),
            "width": _parse_int(first_video.get("width")),
            "height": _parse_int(first_video.get("height")),
            "duration_seconds": duration if video_duration is None else video_duration,
            "avg_frame_rate": average_raw,
            "avg_frame_rate_fps": average_fps,
            "r_frame_rate": real_raw,
            "r_frame_rate_fps": real_fps,
            "frame_rate_fps": average_fps if average_fps is not None else real_fps,
            "creation_time": _creation_time(first_video.get("tags")),
        }

    format_creation_time = _creation_time(format_payload.get("tags"))
    if format_creation_time is not None:
        embedded_creation_time = format_creation_time
        creation_time_source = "format"
    elif stream_creation_times:
        stream_index, embedded_creation_time = stream_creation_times[0]
        creation_time_source = f"stream:{stream_index}"
    else:
        embedded_creation_time = None
        creation_time_source = None

    return {
        "path": str(source),
        "duration_seconds": duration,
        "format_name": format_name,
        "format": {
            "name": format_name,
            "long_name": _string_or_none(format_payload.get("format_long_name")),
            "duration_seconds": duration,
        },
        "streams": normalized_streams,
        "has_audio": bool(audio_streams),
        "audio_stream_count": len(audio_streams),
        "audio": audio,
        "has_video": bool(video_streams),
        "video_stream_count": len(video_streams),
        "video": video,
        "creation_time": embedded_creation_time,
        "creation_time_source": creation_time_source,
    }


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file incrementally so large recordings are not loaded into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_partial(destination: Path) -> None:
    try:
        destination.unlink(missing_ok=True)
    except OSError as exc:
        raise _error("could not remove partial canonical FLAC", exc) from exc


def _audio_duration(metadata: MediaMetadata, role: str) -> float:
    audio = metadata["audio"]
    if audio is None or audio["duration_seconds"] is None:
        raise _error(f"{role} has no usable audio duration")
    return audio["duration_seconds"]


def _validate_canonical_metadata(metadata: MediaMetadata) -> None:
    audio = metadata["audio"]
    if metadata["audio_stream_count"] != 1 or audio is None:
        raise _error("canonical FLAC must contain exactly one audio stream")
    if metadata["has_video"]:
        raise _error("canonical FLAC must not contain video")
    if audio["codec_name"] != "flac":
        raise _error("canonical audio codec is not FLAC", audio["codec_name"])
    if audio["sample_rate"] != 16_000:
        raise _error("canonical FLAC sample rate is not 16000 Hz", audio["sample_rate"])
    if audio["channels"] != 1:
        raise _error("canonical FLAC is not mono", audio["channels"])


# A decode may stop short of the duration in the FLAC header by less than one
# block; a truncated file loses whole blocks (~0.25 s each at 16 kHz).
DECODE_SHORTFALL_TOLERANCE_SECONDS = 0.1


def _verify_complete_decode(path: Path, expected_seconds: float) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(path),
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
    except OSError as exc:
        raise _error("could not start canonical FLAC decoder", exc) from exc
    # ffmpeg 6.x reports a truncated frame on stderr but still exits 0, so the
    # exit code alone is not enough.
    if completed.returncode != 0 or completed.stderr.strip():
        raise _error("canonical FLAC decode failed", completed.stderr)
    decoded_seconds = _decoded_seconds(completed.stdout)
    if decoded_seconds is None:
        raise _error("canonical FLAC decode failed: no progress report from ffmpeg")
    if decoded_seconds < expected_seconds - DECODE_SHORTFALL_TOLERANCE_SECONDS:
        raise _error(
            "canonical FLAC decode failed: decoded "
            f"{decoded_seconds:.3f}s of {expected_seconds:.3f}s"
        )


def _decoded_seconds(progress: str | None) -> float | None:
    last: float | None = None
    for line in (progress or "").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "out_time_us":
            try:
                last = int(value.strip()) / 1_000_000
            except ValueError:
                continue
    return last


def validate_canonical_flac(
    path: str | os.PathLike[str],
) -> MediaMetadata:
    """Verify canonical metadata and decode every audio frame without mutation."""

    artifact_path = Path(path)
    metadata = probe_media(artifact_path)
    _validate_canonical_metadata(metadata)
    _verify_complete_decode(artifact_path, _audio_duration(metadata, "canonical FLAC"))
    return metadata


def extract_canonical_flac(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> CanonicalMediaArtifact:
    """Extract and verify the run-local canonical mono 16 kHz FLAC artifact."""

    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.name != CANONICAL_FLAC_NAME:
        raise _error(f"destination filename must be {CANONICAL_FLAC_NAME}")
    if not destination_path.parent.is_dir():
        raise _error(
            "canonical FLAC parent directory does not exist",
            destination_path.parent,
        )
    if source_path.resolve(strict=False) == destination_path.resolve(strict=False):
        raise _error("source media and canonical FLAC destination must differ")

    source_metadata = probe_media(source_path)
    if not source_metadata["has_audio"]:
        raise _error("input media has no audio stream")
    source_duration = _audio_duration(source_metadata, "input media")

    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(destination_path),
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
    except OSError as exc:
        _remove_partial(destination_path)
        raise _error("could not start ffmpeg", exc) from exc
    if completed.returncode != 0:
        _remove_partial(destination_path)
        raise _error("ffmpeg failed", completed.stderr)
    if not destination_path.is_file() or destination_path.stat().st_size == 0:
        _remove_partial(destination_path)
        raise _error("ffmpeg did not create a non-empty canonical FLAC")

    try:
        output_metadata = validate_canonical_flac(destination_path)
        output_duration = _audio_duration(output_metadata, "canonical FLAC")
        duration_difference = abs(source_duration - output_duration)
        if duration_difference > _DURATION_TOLERANCE_SECONDS:
            raise _error(
                "canonical FLAC duration differs from input audio by more than "
                "0.25 seconds",
                f"difference={duration_difference:.6f}",
            )
        artifact_hash = sha256_file(destination_path)
    except (MediaError, OSError) as exc:
        _remove_partial(destination_path)
        if isinstance(exc, MediaError):
            raise
        raise _error("could not hash canonical FLAC", exc) from exc

    return {
        "path": str(destination_path),
        "sha256": artifact_hash,
        "metadata": output_metadata,
        "source_audio_duration_seconds": source_duration,
        "duration_difference_seconds": duration_difference,
    }


__all__ = [
    "CANONICAL_FLAC_NAME",
    "DIAGNOSTIC_LIMIT",
    "CanonicalMediaArtifact",
    "MediaError",
    "MediaMetadata",
    "extract_canonical_flac",
    "probe_media",
    "sha256_file",
    "validate_canonical_flac",
]
