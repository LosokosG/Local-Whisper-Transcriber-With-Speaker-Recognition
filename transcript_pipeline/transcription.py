"""Lazy, headless WhisperX transcription with stable artifact output."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from .redaction import REDACTED, redact_text

DIARIZED_SEGMENTS_NAME = "diarized_segments.json"
DEFAULT_MODEL_NAME = "large-v3-turbo"
DEFAULT_BATCH_SIZE = 8
HF_TOKEN_ENV = "HF_TOKEN"
DIAGNOSTIC_LIMIT = 2048


class TranscriptionError(RuntimeError):
    """Raised when WhisperX fails or returns an unsafe transcript shape."""


class Processor(Protocol):
    def process(self, audio_path: str) -> object: ...


ProcessorFactory = Callable[..., Processor]


class NormalizedSegment(TypedDict):
    id: str
    start: float
    end: float
    text: str
    speaker: str | None


class SpeakerRecognitionStatus(TypedDict):
    status: str
    code: str | None
    reason: str | None


class ModelMetadata(TypedDict):
    name: str
    device: str
    compute_type: str
    batch_size: int
    language: str | None
    min_speakers: int | None
    max_speakers: int | None
    num_speakers: int | None


class PhaseMetadata(TypedDict):
    status: str
    code: str | None
    error: str | None


class ProcessingMetadata(TypedDict):
    alignment: PhaseMetadata
    diarization: PhaseMetadata


class TranscriptionDocument(TypedDict):
    language: str
    diarization_available: bool
    speaker_recognition: SpeakerRecognitionStatus
    processing: ProcessingMetadata
    model: ModelMetadata
    segments: list[NormalizedSegment]


def _positive_speaker_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class TranscriptionSettings:
    """Credential-free transcription configuration.

    ``device`` and ``compute_type`` remain unresolved when omitted so importing
    this module and constructing settings never imports the ML stack. The
    Hugging Face token itself is read only by :func:`transcribe`.
    """

    model_name: str = DEFAULT_MODEL_NAME
    device: str | None = None
    compute_type: str | None = None
    language: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None
    hf_token_env: str | None = HF_TOKEN_ENV

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        object.__setattr__(self, "model_name", self.model_name.strip())

        if self.device is not None:
            if not isinstance(self.device, str):
                raise TypeError("device must be 'cuda', 'cpu', or None")
            device = self.device.strip().lower()
            if device not in {"cuda", "cpu"}:
                raise ValueError("device must be 'cuda', 'cpu', or None")
            object.__setattr__(self, "device", device)

        if self.compute_type is not None:
            if not isinstance(self.compute_type, str) or not self.compute_type.strip():
                raise ValueError("compute_type must be a nonempty string or None")
            object.__setattr__(self, "compute_type", self.compute_type.strip())

        if self.language is not None:
            if not isinstance(self.language, str):
                raise TypeError("language must be a language code, 'auto', or None")
            language = self.language.strip()
            object.__setattr__(
                self,
                "language",
                None if language.lower() == "auto" or not language else language,
            )

        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for name in ("min_speakers", "max_speakers", "num_speakers"):
            _positive_speaker_count(name, getattr(self, name))
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot exceed max_speakers")
        if self.num_speakers is not None:
            if (
                self.min_speakers is not None
                and self.num_speakers < self.min_speakers
            ):
                raise ValueError("num_speakers cannot be below min_speakers")
            if (
                self.max_speakers is not None
                and self.num_speakers > self.max_speakers
            ):
                raise ValueError("num_speakers cannot exceed max_speakers")

        if self.hf_token_env is not None:
            if not isinstance(self.hf_token_env, str) or not self.hf_token_env.strip():
                raise ValueError("hf_token_env must be a nonempty name or None")
            object.__setattr__(self, "hf_token_env", self.hf_token_env.strip())


@dataclass(frozen=True, slots=True)
class _ResolvedSettings:
    model_name: str
    device: str
    compute_type: str
    language: str | None
    batch_size: int
    min_speakers: int | None
    max_speakers: int | None
    num_speakers: int | None


def _cuda_is_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except (AttributeError, ImportError, RuntimeError):
        return False


def _resolve_settings(settings: TranscriptionSettings) -> _ResolvedSettings:
    device = settings.device
    if device is None:
        device = "cuda" if _cuda_is_available() else "cpu"
    compute_type = settings.compute_type
    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"
    return _ResolvedSettings(
        model_name=settings.model_name,
        device=device,
        compute_type=compute_type,
        language=settings.language,
        batch_size=settings.batch_size,
        min_speakers=settings.min_speakers,
        max_speakers=settings.max_speakers,
        num_speakers=settings.num_speakers,
    )


def _default_processor_factory(**kwargs: Any) -> Processor:
    from .whisperx_processor import WhisperXProcessor

    return WhisperXProcessor(**kwargs)


def _processor_kwargs(
    settings: _ResolvedSettings,
    *,
    batch_size: int,
    hf_token: str | None,
) -> dict[str, Any]:
    return {
        "model_name": settings.model_name,
        "device": settings.device,
        "compute_type": settings.compute_type,
        "language": settings.language,
        "hf_token": hf_token,
        "batch_size": batch_size,
        "min_speakers": settings.min_speakers,
        "max_speakers": settings.max_speakers,
        "num_speakers": settings.num_speakers,
        "log": _discard_log,
    }


def _discard_log(_message: object) -> None:
    """Keep the headless adapter silent, including credential-bearing errors."""


def _is_cuda_oom(error: BaseException) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    error_type = type(error).__name__.replace("_", "").lower()
    if "outofmemoryerror" in error_type:
        return True
    message = " ".join(str(error).lower().split())
    compact_message = message.replace("_", "").replace(" ", "")
    return (
        "outofmemoryerror" in compact_message
        or "cuda out of memory" in message
        or ("cuda" in message and "out of memory" in message)
    )


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
        torch.cuda.empty_cache()
    except (AttributeError, ImportError, RuntimeError):
        pass


def _safe_failure(error: BaseException, hf_token: str | None) -> TranscriptionError:
    message = str(error)
    if hf_token:
        message = message.replace(hf_token, REDACTED)
    message = redact_text(message).strip() or type(error).__name__
    if len(message) > DIAGNOSTIC_LIMIT:
        message = message[: DIAGNOSTIC_LIMIT - 1] + "…"
    return TranscriptionError(f"WhisperX processing failed: {message}")


def _run_processor(
    audio_path: str | os.PathLike[str],
    settings: _ResolvedSettings,
    factory: ProcessorFactory,
    hf_token: str | None,
) -> tuple[object, int]:
    batch_size = settings.batch_size
    for attempt in range(2):
        try:
            kwargs = _processor_kwargs(
                settings,
                batch_size=batch_size,
                hf_token=hf_token,
            )
            if factory is _default_processor_factory:
                kwargs.pop("hf_token")
                kwargs.pop("log")
                processor = factory(**kwargs)
                result = processor.process(
                    os.fspath(audio_path),
                    hf_token=hf_token,
                )
            else:
                processor = factory(**kwargs)
                result = processor.process(os.fspath(audio_path))
            return result, batch_size
        except Exception as error:
            if attempt == 0 and settings.device == "cuda" and _is_cuda_oom(error):
                batch_size = max(1, batch_size // 2)
                _clear_cuda_cache()
                continue
            raise _safe_failure(error, hf_token) from None
    raise AssertionError("unreachable")


def _timestamp(value: object, *, segment_number: int, field: str) -> float:
    if isinstance(value, bool):
        raise TranscriptionError(
            f"segment {segment_number} {field} must be a numeric timestamp"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TranscriptionError(
            f"segment {segment_number} {field} must be a numeric timestamp"
        ) from error
    if not math.isfinite(parsed):
        raise TranscriptionError(
            f"segment {segment_number} {field} must be finite"
        )
    if parsed < 0:
        raise TranscriptionError(
            f"segment {segment_number} {field} must be nonnegative"
        )
    return parsed


def _safe_metadata_text(value: str, hf_token: str | None) -> str:
    if hf_token:
        value = value.replace(hf_token, REDACTED)
    safe = redact_text(value).strip()
    if len(safe) > DIAGNOSTIC_LIMIT:
        safe = safe[: DIAGNOSTIC_LIMIT - 1] + "…"
    return safe


def _phase_metadata(
    value: object,
    *,
    phase: str,
    hf_token: str | None,
) -> PhaseMetadata:
    if not isinstance(value, Mapping):
        raise TranscriptionError(f"processing {phase} status must be an object")
    status = value.get("status")
    if status not in {"completed", "failed", "blocked", "unknown"}:
        raise TranscriptionError(f"processing {phase} status is invalid")

    code_value = value.get("code")
    if code_value is not None and (
        not isinstance(code_value, str) or not code_value.strip()
    ):
        raise TranscriptionError(f"processing {phase} code must be text or null")
    code = code_value.strip() if isinstance(code_value, str) else None

    error_value = value.get("error")
    if error_value is not None and not isinstance(error_value, str):
        raise TranscriptionError(f"processing {phase} error must be text or null")
    error = (
        _safe_metadata_text(error_value, hf_token)
        if isinstance(error_value, str) and error_value.strip()
        else None
    )
    return {"status": status, "code": code, "error": error}


def _processing_metadata(
    result: Mapping[str, object],
    *,
    diarization_available: bool,
    hf_token: str | None,
) -> ProcessingMetadata:
    value = result.get("processing")
    if value is not None:
        if not isinstance(value, Mapping):
            raise TranscriptionError("processing status must be an object")
        return {
            "alignment": _phase_metadata(
                value.get("alignment"),
                phase="alignment",
                hf_token=hf_token,
            ),
            "diarization": _phase_metadata(
                value.get("diarization"),
                phase="diarization",
                hf_token=hf_token,
            ),
        }

    alignment: PhaseMetadata = {
        "status": "unknown",
        "code": "processor_status_unavailable",
        "error": None,
    }
    if diarization_available:
        diarization: PhaseMetadata = {
            "status": "completed",
            "code": None,
            "error": None,
        }
    elif hf_token is None:
        diarization = {
            "status": "blocked",
            "code": "hf_token_unavailable",
            "error": None,
        }
    else:
        diarization = {
            "status": "failed",
            "code": "speaker_assignments_missing",
            "error": "WhisperX did not produce speaker assignments.",
        }
    return {"alignment": alignment, "diarization": diarization}


def _normalize_result(
    result: object,
    settings: _ResolvedSettings,
    *,
    batch_size: int,
    hf_token: str | None,
) -> TranscriptionDocument:
    if not isinstance(result, Mapping):
        raise TranscriptionError("WhisperX result must be an object")

    language_value = result.get("language")
    if not isinstance(language_value, str) or not language_value.strip():
        raise TranscriptionError("WhisperX result language must be nonempty")
    language = language_value.strip()

    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list):
        raise TranscriptionError("WhisperX result segments must be a list")
    if not raw_segments:
        raise TranscriptionError("WhisperX result must contain at least one segment")

    segments: list[NormalizedSegment] = []
    previous_start = -1.0
    diarization_available = False
    for number, raw_segment in enumerate(raw_segments, start=1):
        if not isinstance(raw_segment, Mapping):
            raise TranscriptionError(f"segment {number} must be an object")
        start = _timestamp(
            raw_segment.get("start"), segment_number=number, field="start"
        )
        end = _timestamp(raw_segment.get("end"), segment_number=number, field="end")
        if end < start:
            raise TranscriptionError(f"segment {number} end is before start")
        if start < previous_start:
            raise TranscriptionError("segment starts must be chronological")
        previous_start = start

        text_value = raw_segment.get("text")
        if not isinstance(text_value, str):
            raise TranscriptionError(f"segment {number} text must be a string")
        text = text_value.strip()
        if not text:
            raise TranscriptionError(f"segment {number} must have nonempty text")

        speaker_value = raw_segment.get("speaker")
        if speaker_value is not None and not isinstance(speaker_value, str):
            raise TranscriptionError(
                f"segment {number} speaker must be a string or null"
            )
        speaker = speaker_value.strip() if isinstance(speaker_value, str) else None
        speaker = speaker or None
        diarization_available = diarization_available or speaker is not None
        segments.append(
            {
                "id": f"seg-{number:06d}",
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
            }
        )

    processing = _processing_metadata(
        result,
        diarization_available=diarization_available,
        hf_token=hf_token,
    )
    if diarization_available:
        speaker_recognition: SpeakerRecognitionStatus = {
            "status": "completed",
            "code": None,
            "reason": None,
        }
    else:
        diarization = processing["diarization"]
        code = diarization["code"] or "speaker_assignments_missing"
        if diarization["error"]:
            reason = diarization["error"]
        elif code == "hf_token_unavailable":
            reason = "Speaker recognition requires a Hugging Face token."
        else:
            reason = "WhisperX did not produce speaker assignments."
        speaker_recognition = {
            "status": "blocked",
            "code": code,
            "reason": reason,
        }

    return {
        "language": language,
        "diarization_available": diarization_available,
        "speaker_recognition": speaker_recognition,
        "processing": processing,
        "model": {
            "name": settings.model_name,
            "device": settings.device,
            "compute_type": settings.compute_type,
            "batch_size": batch_size,
            "language": settings.language,
            "min_speakers": settings.min_speakers,
            "max_speakers": settings.max_speakers,
            "num_speakers": settings.num_speakers,
        },
        "segments": segments,
    }


def transcribe(
    audio_path: str | os.PathLike[str],
    settings: TranscriptionSettings | None = None,
    processor_factory: ProcessorFactory | None = None,
) -> TranscriptionDocument:
    """Process audio and return a stable, JSON-compatible transcript document."""

    effective_settings = settings or TranscriptionSettings()
    if not isinstance(effective_settings, TranscriptionSettings):
        raise TypeError("settings must be TranscriptionSettings or None")
    resolved = _resolve_settings(effective_settings)
    token = None
    if effective_settings.hf_token_env is not None:
        token = os.environ.get(effective_settings.hf_token_env)
        token = token.strip() if token else None
    result, used_batch_size = _run_processor(
        audio_path,
        resolved,
        processor_factory or _default_processor_factory,
        token,
    )
    return _normalize_result(
        result,
        resolved,
        batch_size=used_batch_size,
        hf_token=token,
    )


def write_transcription_atomic(
    output_path: str | os.PathLike[str],
    document: Mapping[str, Any],
) -> None:
    """Atomically persist the canonical strict-JSON transcription artifact."""

    path = Path(output_path)
    if path.name != DIARIZED_SEGMENTS_NAME:
        raise ValueError(f"output basename must be {DIARIZED_SEGMENTS_NAME}")
    if not isinstance(document, Mapping):
        raise TypeError("transcription document must be a mapping")
    rendered = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

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
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def transcribe_to_file(
    audio_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    settings: TranscriptionSettings | None = None,
    processor_factory: ProcessorFactory | None = None,
) -> TranscriptionDocument:
    """Transcribe audio, atomically write it, and return the same document."""

    document = transcribe(audio_path, settings, processor_factory)
    write_transcription_atomic(output_path, document)
    return document


__all__ = [
    "DIARIZED_SEGMENTS_NAME",
    "TranscriptionDocument",
    "TranscriptionError",
    "TranscriptionSettings",
    "transcribe",
    "transcribe_to_file",
    "write_transcription_atomic",
]
