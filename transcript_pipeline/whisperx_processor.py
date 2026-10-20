"""Package-owned WhisperX processor with phase-conscious model cleanup."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from .redaction import REDACTED, redact_text

DIAGNOSTIC_LIMIT = 2048


class PhaseStatus(TypedDict):
    status: str
    code: str | None
    error: str | None


class ProcessingStatus(TypedDict):
    alignment: PhaseStatus
    diarization: PhaseStatus


class ProcessorResult(TypedDict):
    language: str
    segments: list[dict[str, Any]]
    processing: ProcessingStatus


DependencyLoader = Callable[[], tuple[Any, Any]]


def _load_dependencies() -> tuple[Any, Any]:
    """Import the ML stack only when processing actually starts."""

    return importlib.import_module("torch"), importlib.import_module("whisperx")


def _diarization_pipeline_factory(whisperx: Any) -> Any:
    """Resolve the diarizer across WhisperX root and submodule APIs lazily."""

    factory = getattr(whisperx, "DiarizationPipeline", None)
    if factory is None:
        diarize = importlib.import_module("whisperx.diarize")
        factory = diarize.DiarizationPipeline
    return factory


def _positive_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _status(
    status: str,
    code: str | None = None,
    error: str | None = None,
) -> PhaseStatus:
    return {"status": status, "code": code, "error": error}


def _safe_error(error: BaseException, runtime_token: str | None) -> str:
    message = str(error)
    if runtime_token:
        message = message.replace(runtime_token, REDACTED)
    message = redact_text(message).strip() or type(error).__name__
    if len(message) > DIAGNOSTIC_LIMIT:
        return message[: DIAGNOSTIC_LIMIT - 1] + "…"
    return message


def _is_cuda_oom(error: BaseException) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    error_type = type(error).__name__.replace("_", "").lower()
    message = " ".join(str(error).lower().split())
    compact_message = message.replace("_", "").replace(" ", "")
    return (
        "outofmemoryerror" in error_type
        or "outofmemoryerror" in compact_message
        or "cuda out of memory" in message
        or ("cuda" in message and "out of memory" in message)
    )


@dataclass(frozen=True, slots=True)
class WhisperXProcessor:
    """Run transcription, alignment, and optional speaker diarization.

    The processor stores only safe model settings. The Hugging Face token is a
    process-call argument so it is neither retained on the instance nor shown
    in its representation.
    """

    model_name: str = "large-v3-turbo"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    batch_size: int = 8
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None
    dependency_loader: DependencyLoader = field(
        default=_load_dependencies,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        object.__setattr__(self, "model_name", self.model_name.strip())

        if not isinstance(self.device, str) or self.device.strip().lower() not in {
            "cpu",
            "cuda",
        }:
            raise ValueError("device must be 'cpu' or 'cuda'")
        object.__setattr__(self, "device", self.device.strip().lower())

        if not isinstance(self.compute_type, str) or not self.compute_type.strip():
            raise ValueError("compute_type must be a nonempty string")
        object.__setattr__(self, "compute_type", self.compute_type.strip())

        if self.language is not None:
            if not isinstance(self.language, str):
                raise TypeError("language must be a language code, 'auto', or None")
            language = self.language.strip()
            object.__setattr__(
                self,
                "language",
                None if not language or language.lower() == "auto" else language,
            )

        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for name in ("min_speakers", "max_speakers", "num_speakers"):
            _positive_count(name, getattr(self, name))
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
        if not callable(self.dependency_loader):
            raise TypeError("dependency_loader must be callable")

    def _cleanup(self, torch: Any) -> None:
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _speaker_bounds(self) -> dict[str, int]:
        if self.num_speakers is not None:
            return {
                "min_speakers": self.num_speakers,
                "max_speakers": self.num_speakers,
            }
        bounds: dict[str, int] = {}
        if self.min_speakers is not None:
            bounds["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            bounds["max_speakers"] = self.max_speakers
        return bounds

    @staticmethod
    def _has_speaker_assignments(result: Mapping[str, Any]) -> bool:
        segments = result.get("segments")
        if not isinstance(segments, list):
            return False
        return any(
            isinstance(segment, Mapping)
            and isinstance(speaker := segment.get("speaker"), str)
            and bool(speaker.strip())
            for segment in segments
        )

    def process(
        self,
        audio_path: str,
        *,
        hf_token: str | None = None,
    ) -> ProcessorResult:
        """Run all phases and retain useful transcript evidence on soft failures."""

        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError("audio_path must be a nonempty string")
        if hf_token is not None and not isinstance(hf_token, str):
            raise TypeError("hf_token must be a string or None")
        runtime_token = hf_token.strip() if hf_token else None
        torch, whisperx = self.dependency_loader()
        audio = whisperx.load_audio(audio_path)

        model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        try:
            transcription = model.transcribe(audio, batch_size=self.batch_size)
        finally:
            del model
            self._cleanup(torch)
        if not isinstance(transcription, Mapping):
            raise TypeError("WhisperX transcription result must be an object")
        raw_segments = transcription.get("segments")
        if not isinstance(raw_segments, list):
            raise TypeError("WhisperX transcription segments must be a list")
        language = transcription.get("language")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("WhisperX did not return a detected language")
        language = language.strip()
        result: dict[str, Any] = dict(transcription)

        align_model: Any = None
        align_metadata: Any = None
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            if not isinstance(aligned, Mapping):
                raise TypeError("WhisperX alignment result must be an object")
            result = dict(aligned)
            alignment_status = _status("completed")
        except Exception as error:
            if _is_cuda_oom(error):
                raise
            alignment_status = _status(
                "failed",
                "alignment_failed",
                _safe_error(error, runtime_token),
            )
        finally:
            del align_model, align_metadata
            self._cleanup(torch)

        if runtime_token is None:
            diarization_status = _status("blocked", "hf_token_unavailable")
        else:
            diarizer: Any = None
            diarization_segments: Any = None
            try:
                diarization_pipeline = _diarization_pipeline_factory(whisperx)
                diarizer = diarization_pipeline(
                    token=runtime_token,
                    device=self.device,
                )
                diarization_segments = diarizer(audio, **self._speaker_bounds())
                assigned = whisperx.assign_word_speakers(
                    diarization_segments,
                    result,
                )
                if not isinstance(assigned, Mapping):
                    raise TypeError("WhisperX speaker result must be an object")
                result = dict(assigned)
                if self._has_speaker_assignments(result):
                    diarization_status = _status("completed")
                else:
                    diarization_status = _status(
                        "failed",
                        "speaker_assignments_missing",
                        "Diarization completed without speaker assignments.",
                    )
            except Exception as error:
                if _is_cuda_oom(error):
                    raise
                diarization_status = _status(
                    "failed",
                    "diarization_failed",
                    _safe_error(error, runtime_token),
                )
            finally:
                del diarizer, diarization_segments
                self._cleanup(torch)

        segments = result.get("segments")
        if not isinstance(segments, list):
            raise TypeError("WhisperX processing result segments must be a list")
        return {
            "language": language,
            "segments": segments,
            "processing": {
                "alignment": alignment_status,
                "diarization": diarization_status,
            },
        }


__all__ = ["ProcessorResult", "WhisperXProcessor"]
