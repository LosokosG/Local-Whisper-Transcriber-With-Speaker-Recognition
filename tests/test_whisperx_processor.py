from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest


RAW_SEGMENT = {"start": 0.0, "end": 1.0, "text": "hello"}


def _fake_dependencies(
    *,
    alignment_error: BaseException | None = None,
    diarization_error: BaseException | None = None,
    assign_speakers: bool = True,
) -> tuple[object, object, list[tuple[object, ...]]]:
    events: list[tuple[object, ...]] = []

    class FakeModel:
        def transcribe(self, audio: object, *, batch_size: int) -> dict[str, object]:
            events.append(("transcribe", audio, batch_size))
            return {"language": "en", "segments": [dict(RAW_SEGMENT)]}

    class FakeDiarizer:
        def __call__(self, audio: object, **kwargs: object) -> object:
            events.append(("diarize", audio, kwargs))
            if diarization_error is not None:
                raise diarization_error
            return "diarization-table"

    def load_audio(path: str) -> object:
        events.append(("load_audio", path))
        return "decoded-audio"

    def load_model(
        model_name: str,
        device: str,
        *,
        compute_type: str,
        language: str | None,
    ) -> FakeModel:
        events.append(
            ("load_model", model_name, device, compute_type, language)
        )
        return FakeModel()

    def load_align_model(*, language_code: str, device: str) -> tuple[object, object]:
        events.append(("load_align_model", language_code, device))
        return "alignment-model", "alignment-metadata"

    def align(
        segments: list[dict[str, object]],
        model: object,
        metadata: object,
        audio: object,
        device: str,
        *,
        return_char_alignments: bool,
    ) -> dict[str, object]:
        events.append(
            (
                "align",
                segments,
                model,
                metadata,
                audio,
                device,
                return_char_alignments,
            )
        )
        if alignment_error is not None:
            raise alignment_error
        return {
            "segments": [
                {**RAW_SEGMENT, "text": " aligned ", "words": [{"word": "aligned"}]}
            ]
        }

    def diarization_pipeline(*, token: str, device: str) -> FakeDiarizer:
        events.append(("DiarizationPipeline", token, device))
        return FakeDiarizer()

    def assign_word_speakers(
        diarization: object,
        result: dict[str, object],
    ) -> dict[str, object]:
        events.append(("assign_word_speakers", diarization, result))
        segments = [dict(segment) for segment in result["segments"]]
        if assign_speakers:
            segments[0]["speaker"] = "SPEAKER_00"
        return {**result, "segments": segments}

    fake_whisperx = types.SimpleNamespace(
        load_audio=load_audio,
        load_model=load_model,
        load_align_model=load_align_model,
        align=align,
        DiarizationPipeline=diarization_pipeline,
        assign_word_speakers=assign_word_speakers,
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            empty_cache=lambda: events.append(("empty_cache",))
        )
    )
    return fake_torch, fake_whisperx, events


def test_module_import_does_not_import_ml_stack() -> None:
    script = """
import json
import sys

before = set(sys.modules)
import transcript_pipeline.whisperx_processor
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


def test_full_pipeline_uses_current_api_and_cleans_each_gpu_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.whisperx_processor as backend

    torch, whisperx, events = _fake_dependencies()
    monkeypatch.setattr(backend.gc, "collect", lambda: events.append(("gc",)))
    processor = backend.WhisperXProcessor(
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="int8_float16",
        language=None,
        batch_size=8,
        min_speakers=2,
        max_speakers=4,
        dependency_loader=lambda: (torch, whisperx),
    )

    result = processor.process("recording.flac", hf_token="hf_runtime-only")

    assert [event[0] for event in events] == [
        "load_audio",
        "load_model",
        "transcribe",
        "gc",
        "empty_cache",
        "load_align_model",
        "align",
        "gc",
        "empty_cache",
        "DiarizationPipeline",
        "diarize",
        "assign_word_speakers",
        "gc",
        "empty_cache",
    ]
    assert events[1] == (
        "load_model",
        "large-v3-turbo",
        "cuda",
        "int8_float16",
        None,
    )
    assert events[2] == ("transcribe", "decoded-audio", 8)
    assert events[5] == ("load_align_model", "en", "cuda")
    assert events[6][-1] is False
    assert events[9] == ("DiarizationPipeline", "hf_runtime-only", "cuda")
    assert events[10] == (
        "diarize",
        "decoded-audio",
        {"min_speakers": 2, "max_speakers": 4},
    )
    assert result["language"] == "en"
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["processing"] == {
        "alignment": {"status": "completed", "code": None, "error": None},
        "diarization": {"status": "completed", "code": None, "error": None},
    }


def test_diarization_pipeline_falls_back_to_whisperx_diarize_submodule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.whisperx_processor as backend

    torch, whisperx, events = _fake_dependencies()
    diarization_pipeline = whisperx.DiarizationPipeline
    del whisperx.DiarizationPipeline
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        assert name == "whisperx.diarize"
        return types.SimpleNamespace(DiarizationPipeline=diarization_pipeline)

    monkeypatch.setattr(backend.importlib, "import_module", import_module)
    processor = backend.WhisperXProcessor(
        device="cpu",
        dependency_loader=lambda: (torch, whisperx),
    )

    result = processor.process("recording.flac", hf_token="hf_runtime-only")

    assert imported == ["whisperx.diarize"]
    assert any(event[0] == "DiarizationPipeline" for event in events)
    assert result["processing"]["diarization"] == {
        "status": "completed",
        "code": None,
        "error": None,
    }


def test_diarization_pipeline_prefers_whisperx_root_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.whisperx_processor as backend

    torch, whisperx, events = _fake_dependencies()

    def unexpected_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(backend.importlib, "import_module", unexpected_import)
    processor = backend.WhisperXProcessor(
        device="cpu",
        dependency_loader=lambda: (torch, whisperx),
    )

    result = processor.process("recording.flac", hf_token="hf_runtime-only")

    assert any(event[0] == "DiarizationPipeline" for event in events)
    assert result["processing"]["diarization"]["status"] == "completed"


def test_fixed_speaker_count_maps_to_equal_current_api_bounds() -> None:
    from transcript_pipeline.whisperx_processor import WhisperXProcessor

    torch, whisperx, events = _fake_dependencies()
    processor = WhisperXProcessor(
        device="cpu",
        num_speakers=3,
        dependency_loader=lambda: (torch, whisperx),
    )

    processor.process("recording.flac", hf_token="hf_runtime-only")

    diarize_event = next(event for event in events if event[0] == "diarize")
    assert diarize_event[2] == {"min_speakers": 3, "max_speakers": 3}


def test_missing_token_skips_diarization_with_truthful_blocked_status() -> None:
    from transcript_pipeline.whisperx_processor import WhisperXProcessor

    torch, whisperx, events = _fake_dependencies()
    processor = WhisperXProcessor(
        device="cpu",
        dependency_loader=lambda: (torch, whisperx),
    )

    result = processor.process("recording.flac")

    assert not any(event[0] == "DiarizationPipeline" for event in events)
    assert result["segments"][0].get("speaker") is None
    assert result["processing"]["diarization"] == {
        "status": "blocked",
        "code": "hf_token_unavailable",
        "error": None,
    }


def test_alignment_failure_keeps_transcript_and_continues_diarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.whisperx_processor as backend

    secret = "alignment-password-secret"
    torch, whisperx, events = _fake_dependencies(
        alignment_error=RuntimeError(f"password={secret}")
    )
    monkeypatch.setattr(backend.gc, "collect", lambda: events.append(("gc",)))
    processor = backend.WhisperXProcessor(
        device="cuda",
        dependency_loader=lambda: (torch, whisperx),
    )

    result = processor.process("recording.flac", hf_token="hf_runtime-only")

    assert result["segments"][0]["text"] == "hello"
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    alignment = result["processing"]["alignment"]
    assert alignment["status"] == "failed"
    assert alignment["code"] == "alignment_failed"
    assert secret not in alignment["error"]
    assert "[REDACTED]" in alignment["error"]
    assert any(event[0] == "DiarizationPipeline" for event in events)
    align_index = next(i for i, event in enumerate(events) if event[0] == "align")
    assert events[align_index + 1 : align_index + 3] == [
        ("gc",),
        ("empty_cache",),
    ]


def test_diarization_failure_keeps_transcript_and_redacts_runtime_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.whisperx_processor as backend

    token = "hf_diarization-runtime-secret"
    torch, whisperx, events = _fake_dependencies(
        diarization_error=RuntimeError(f"request rejected for {token}")
    )
    monkeypatch.setattr(backend.gc, "collect", lambda: events.append(("gc",)))
    processor = backend.WhisperXProcessor(
        device="cuda",
        dependency_loader=lambda: (torch, whisperx),
    )

    result = processor.process("recording.flac", hf_token=token)

    assert result["segments"][0]["text"] == " aligned "
    assert result["segments"][0].get("speaker") is None
    diarization = result["processing"]["diarization"]
    assert diarization["status"] == "failed"
    assert diarization["code"] == "diarization_failed"
    assert token not in diarization["error"]
    assert "[REDACTED]" in diarization["error"]
    assert not any(event[0] == "assign_word_speakers" for event in events)
    diarize_index = next(i for i, event in enumerate(events) if event[0] == "diarize")
    assert events[diarize_index + 1 : diarize_index + 3] == [
        ("gc",),
        ("empty_cache",),
    ]


def test_diarization_without_assignments_is_not_reported_as_completed() -> None:
    from transcript_pipeline.whisperx_processor import WhisperXProcessor

    torch, whisperx, _events = _fake_dependencies(assign_speakers=False)
    processor = WhisperXProcessor(
        device="cpu",
        dependency_loader=lambda: (torch, whisperx),
    )

    result = processor.process("recording.flac", hf_token="hf_runtime-only")

    assert result["processing"]["diarization"] == {
        "status": "failed",
        "code": "speaker_assignments_missing",
        "error": "Diarization completed without speaker assignments.",
    }


@pytest.mark.parametrize("phase", ["alignment", "diarization"])
def test_cuda_oom_in_optional_phase_is_propagated_for_adapter_retry(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.whisperx_processor as backend

    error = RuntimeError("CUDA out of memory while allocating tensor")
    torch, whisperx, events = _fake_dependencies(
        alignment_error=error if phase == "alignment" else None,
        diarization_error=error if phase == "diarization" else None,
    )
    monkeypatch.setattr(backend.gc, "collect", lambda: events.append(("gc",)))
    processor = backend.WhisperXProcessor(
        device="cuda",
        dependency_loader=lambda: (torch, whisperx),
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        processor.process("recording.flac", hf_token="hf_runtime-only")

    failing_event = "align" if phase == "alignment" else "diarize"
    failure_index = next(
        index for index, event in enumerate(events) if event[0] == failing_event
    )
    assert events[failure_index + 1 : failure_index + 3] == [
        ("gc",),
        ("empty_cache",),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"device": "mps"},
        {"min_speakers": 4, "max_speakers": 2},
        {"num_speakers": 3, "max_speakers": 2},
    ],
)
def test_invalid_processor_settings_are_rejected(kwargs: dict[str, Any]) -> None:
    from transcript_pipeline.whisperx_processor import WhisperXProcessor

    with pytest.raises((TypeError, ValueError)):
        WhisperXProcessor(**kwargs)
