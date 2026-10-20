from __future__ import annotations

import json
import os
import subprocess
import sys
import types
import builtins
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


class FakeProcessor:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.audio_paths: list[str] = []

    def process(self, audio_path: str) -> object:
        self.audio_paths.append(audio_path)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class RecordingFactory:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.processors: list[FakeProcessor] = []

    def __call__(self, **kwargs: Any) -> FakeProcessor:
        self.calls.append(kwargs)
        processor = FakeProcessor(self.outcomes.pop(0))
        self.processors.append(processor)
        return processor


def _raw_result(
    *segments: Mapping[str, object], language: str = "en"
) -> dict[str, object]:
    return {"language": language, "segments": list(segments)}


def test_import_is_light_and_does_not_import_ml_stack() -> None:
    script = """
import json
import sys

before = set(sys.modules)
import transcript_pipeline.transcription
loaded = set(sys.modules) - before
print(json.dumps(sorted(
    name for name in loaded
    if name == 'torch' or name.startswith('torch.')
    or name == 'whisperx' or name.startswith('whisperx.')
    or name == 'whisperx_backend'
)))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    assert json.loads(completed.stdout) == []


def test_successful_diarized_transcription_is_normalized_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe

    token = "hf_test_secret-that-must-not-persist"
    monkeypatch.setenv("HF_TOKEN", token)
    factory = RecordingFactory(
        _raw_result(
            {"start": 0, "end": 1.25, "text": "  Hello  ", "speaker": " SPEAKER_00 "},
            {"start": 1.25, "end": 2, "text": "world", "speaker": "SPEAKER_01"},
            language="pl",
        )
    )
    settings = TranscriptionSettings(
        device="cpu",
        min_speakers=2,
        max_speakers=4,
    )

    document = transcribe("recording.flac", settings, factory)

    assert document == {
        "language": "pl",
        "diarization_available": True,
        "speaker_recognition": {
            "status": "completed",
            "code": None,
            "reason": None,
        },
        "processing": {
            "alignment": {
                "status": "unknown",
                "code": "processor_status_unavailable",
                "error": None,
            },
            "diarization": {
                "status": "completed",
                "code": None,
                "error": None,
            },
        },
        "model": {
            "name": "large-v3-turbo",
            "device": "cpu",
            "compute_type": "int8",
            "batch_size": 8,
            "language": None,
            "min_speakers": 2,
            "max_speakers": 4,
            "num_speakers": None,
        },
        "segments": [
            {
                "id": "seg-000001",
                "start": 0.0,
                "end": 1.25,
                "text": "Hello",
                "speaker": "SPEAKER_00",
            },
            {
                "id": "seg-000002",
                "start": 1.25,
                "end": 2.0,
                "text": "world",
                "speaker": "SPEAKER_01",
            },
        ],
    }
    assert factory.calls[0]["model_name"] == "large-v3-turbo"
    assert factory.calls[0]["language"] is None
    assert factory.calls[0]["hf_token"] == token
    assert factory.calls[0]["batch_size"] == 8
    assert factory.calls[0]["min_speakers"] == 2
    assert factory.calls[0]["max_speakers"] == 4
    assert factory.calls[0]["num_speakers"] is None
    assert callable(factory.calls[0]["log"])
    assert factory.processors[0].audio_paths == ["recording.flac"]
    assert token not in repr(settings)
    assert token not in json.dumps(document)


def test_missing_diarization_returns_transcript_with_structured_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe

    monkeypatch.delenv("HF_TOKEN", raising=False)
    factory = RecordingFactory(
        _raw_result(
            {"start": 0, "end": 1, "text": "Still useful", "speaker": None},
            {"start": 1, "end": 2, "text": "evidence"},
        )
    )

    document = transcribe(
        "recording.flac",
        TranscriptionSettings(device="cpu"),
        factory,
    )

    assert document["diarization_available"] is False
    assert document["speaker_recognition"]["status"] == "blocked"
    assert document["speaker_recognition"]["code"] == "hf_token_unavailable"
    assert document["speaker_recognition"]["reason"]
    assert [segment["text"] for segment in document["segments"]] == [
        "Still useful",
        "evidence",
    ]
    assert [segment["speaker"] for segment in document["segments"]] == [None, None]
    assert factory.calls[0]["hf_token"] is None


def test_token_presence_without_assignments_is_still_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe

    monkeypatch.setenv("HF_TOKEN", "hf_present-but-no-assignments")
    factory = RecordingFactory(
        _raw_result({"start": 0, "end": 1, "text": "No speaker"})
    )

    document = transcribe(
        "recording.flac",
        TranscriptionSettings(device="cpu"),
        factory,
    )

    assert document["diarization_available"] is False
    assert document["speaker_recognition"]["status"] == "blocked"
    assert document["speaker_recognition"]["code"] == "speaker_assignments_missing"


def test_processor_phase_status_is_preserved_and_drives_diarization_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe

    token = "hf_phase-status-secret"
    password = "alignment-status-password"
    monkeypatch.setenv("HF_TOKEN", token)
    result = _raw_result({"start": 0, "end": 1, "text": "Useful transcript"})
    result["processing"] = {
        "alignment": {
            "status": "failed",
            "code": "alignment_failed",
            "error": f"password={password}",
        },
        "diarization": {
            "status": "failed",
            "code": "diarization_failed",
            "error": f"request rejected for {token}",
        },
    }

    document = transcribe(
        "recording.flac",
        TranscriptionSettings(device="cpu"),
        RecordingFactory(result),
    )

    rendered = json.dumps(document)
    assert token not in rendered
    assert password not in rendered
    assert document["processing"] == {
        "alignment": {
            "status": "failed",
            "code": "alignment_failed",
            "error": "password=[REDACTED]",
        },
        "diarization": {
            "status": "failed",
            "code": "diarization_failed",
            "error": "request rejected for [REDACTED]",
        },
    }
    assert document["speaker_recognition"] == {
        "status": "blocked",
        "code": "diarization_failed",
        "reason": "request rejected for [REDACTED]",
    }


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (None, "object"),
        ({"language": "en"}, "segments"),
        ({"language": "en", "segments": "not-a-list"}, "segments"),
        ({"language": "en", "segments": []}, "at least one"),
        ({"language": "", "segments": [{}]}, "language"),
        (_raw_result("not-a-segment"), "segment 1"),
        (_raw_result({"start": -1, "end": 1, "text": "x"}), "nonnegative"),
        (_raw_result({"start": float("nan"), "end": 1, "text": "x"}), "finite"),
        (_raw_result({"start": 0, "end": float("inf"), "text": "x"}), "finite"),
        (_raw_result({"start": 2, "end": 1, "text": "x"}), "before start"),
        (
            _raw_result(
                {"start": 2, "end": 3, "text": "later"},
                {"start": 1, "end": 2, "text": "earlier"},
            ),
            "chronological",
        ),
        (_raw_result({"start": 0, "end": 1, "text": "  "}), "nonempty text"),
        (_raw_result({"start": 0, "end": 1, "text": 42}), "text"),
        (
            _raw_result({"start": 0, "end": 1, "text": "x", "speaker": 42}),
            "speaker",
        ),
    ],
)
def test_malformed_processor_results_are_rejected(
    result: object,
    message: str,
) -> None:
    from transcript_pipeline.transcription import (
        TranscriptionError,
        TranscriptionSettings,
        transcribe,
    )

    factory = RecordingFactory(result)

    with pytest.raises(TranscriptionError, match=message):
        transcribe(
            "recording.flac",
            TranscriptionSettings(device="cpu"),
            factory,
        )


def test_cuda_oom_retries_once_with_half_batch_and_clears_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe

    cache_calls: list[str] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: cache_calls.append("empty_cache")
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    factory = RecordingFactory(
        RuntimeError("CUDA out of memory while allocating tensor"),
        _raw_result(
            {"start": 0, "end": 1, "text": "Recovered", "speaker": "SPEAKER_00"}
        ),
    )

    document = transcribe(
        "recording.flac",
        TranscriptionSettings(device="cuda", batch_size=8),
        factory,
    )

    assert [call["batch_size"] for call in factory.calls] == [8, 4]
    assert [processor.audio_paths for processor in factory.processors] == [
        ["recording.flac"],
        ["recording.flac"],
    ]
    assert cache_calls == ["empty_cache"]
    assert document["model"]["batch_size"] == 4


def test_runtime_error_named_out_of_memory_is_recognized_for_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
    )
    factory = RecordingFactory(
        RuntimeError("OutOfMemoryError while running CUDA inference"),
        _raw_result(
            {"start": 0, "end": 1, "text": "Recovered", "speaker": "SPEAKER_00"}
        ),
    )

    transcribe(
        "recording.flac",
        TranscriptionSettings(device="cuda", batch_size=4),
        factory,
    )

    assert [call["batch_size"] for call in factory.calls] == [4, 2]


def test_non_oom_exception_is_not_retried_and_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import (
        TranscriptionError,
        TranscriptionSettings,
        transcribe,
    )

    secret = "hf_non-oom-secret"
    monkeypatch.setenv("HF_TOKEN", secret)
    factory = RecordingFactory(RuntimeError(f"authentication failed; token={secret}"))

    with pytest.raises(TranscriptionError) as captured:
        transcribe(
            "recording.flac",
            TranscriptionSettings(device="cuda"),
            factory,
        )

    assert len(factory.calls) == 1
    assert secret not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)


def test_second_failure_after_oom_is_not_retried_and_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import (
        TranscriptionError,
        TranscriptionSettings,
        transcribe,
    )

    token = "hf_retry-secret"
    password = "uri-password-secret"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
    )
    factory = RecordingFactory(
        RuntimeError("CUDA out of memory"),
        RuntimeError(
            f"token={token}; download https://user:{password}@example.test/model"
        ),
    )

    with pytest.raises(TranscriptionError) as captured:
        transcribe(
            "recording.flac",
            TranscriptionSettings(device="cuda", batch_size=3),
            factory,
        )

    assert [call["batch_size"] for call in factory.calls] == [3, 1]
    message = str(captured.value)
    assert token not in message
    assert password not in message
    assert "example.test/model" in message


def test_default_factory_uses_packaged_backend_without_root_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe

    constructor_calls: list[dict[str, Any]] = []
    process_calls: list[tuple[str, str | None]] = []
    backend = types.ModuleType("transcript_pipeline.whisperx_processor")

    class StubWhisperXProcessor(FakeProcessor):
        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(kwargs)
            super().__init__(
                _raw_result(
                    {
                        "start": 0,
                        "end": 1,
                        "text": "Default backend",
                        "speaker": "SPEAKER_00",
                    }
                )
            )

        def process(self, audio_path: str, *, hf_token: str | None = None) -> object:
            process_calls.append((audio_path, hf_token))
            return super().process(audio_path)

    backend.WhisperXProcessor = StubWhisperXProcessor
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transcript_pipeline.whisperx_processor", backend)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "whisperx_backend":
            raise AssertionError("default adapter imported the untracked root backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("HF_TOKEN", "hf_runtime-only")

    document = transcribe("recording.flac", TranscriptionSettings())

    assert constructor_calls[0]["device"] == "cuda"
    assert constructor_calls[0]["compute_type"] == "int8_float16"
    assert "hf_token" not in constructor_calls[0]
    assert process_calls == [("recording.flac", "hf_runtime-only")]
    assert document["model"]["device"] == "cuda"


def test_transcribe_to_file_writes_exact_artifact_with_strict_json(
    tmp_path: Path,
) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings, transcribe_to_file

    output = tmp_path / ".work" / "diarized_segments.json"
    factory = RecordingFactory(
        _raw_result(
            {"start": 0, "end": 1, "text": "Saved", "speaker": "SPEAKER_00"}
        )
    )

    document = transcribe_to_file(
        "recording.flac",
        output,
        TranscriptionSettings(device="cpu"),
        factory,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert output.read_bytes().endswith(b"\n")


def test_atomic_writer_preserves_old_file_and_cleans_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.transcription as transcription

    output = tmp_path / "diarized_segments.json"
    old_bytes = b'{"old": true}\n'
    output.write_bytes(old_bytes)

    def interrupted_replace(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(transcription.os, "replace", interrupted_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        transcription.write_transcription_atomic(output, {"new": True})

    assert output.read_bytes() == old_bytes
    assert not list(tmp_path.glob(".diarized_segments.json.*.tmp"))


def test_atomic_writer_rejects_nonfinite_json_without_replacing_old_file(
    tmp_path: Path,
) -> None:
    from transcript_pipeline.transcription import write_transcription_atomic

    output = tmp_path / "diarized_segments.json"
    old_bytes = b'{"old": true}\n'
    output.write_bytes(old_bytes)

    with pytest.raises(ValueError, match="JSON compliant"):
        write_transcription_atomic(output, {"invalid": float("nan")})

    assert output.read_bytes() == old_bytes
    assert not list(tmp_path.glob(".diarized_segments.json.*.tmp"))


def test_atomic_writer_requires_canonical_artifact_basename(tmp_path: Path) -> None:
    from transcript_pipeline.transcription import write_transcription_atomic

    with pytest.raises(ValueError, match="diarized_segments.json"):
        write_transcription_atomic(tmp_path / "other.json", {"segments": []})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"min_speakers": 0},
        {"min_speakers": 4, "max_speakers": 2},
        {"num_speakers": 3, "max_speakers": 2},
        {"device": "mps"},
        {"model_name": "  "},
        {"hf_token_env": ""},
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, object]) -> None:
    from transcript_pipeline.transcription import TranscriptionSettings

    with pytest.raises((TypeError, ValueError)):
        TranscriptionSettings(**kwargs)
