"""Shared synthetic fixtures. Nothing here needs a GPU, a model, or a network."""

from __future__ import annotations

import copy
import io
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
DURATION = 56.0
CREATION_TIME = "2026-03-14T09:30:00.000000Z"

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


@pytest.fixture
def flac_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real FLAC decoder check when ffmpeg exists, a stub otherwise."""

    if HAVE_FFMPEG:
        return
    import transcript_pipeline.media as media

    monkeypatch.setattr(
        media, "validate_canonical_flac", lambda _path: {"duration_seconds": DURATION}
    )


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def diarized() -> dict[str, Any]:
    return load_fixture("diarized.json")


@pytest.fixture
def speaker_evidence() -> dict[str, Any]:
    return load_fixture("speaker-evidence.json")


@pytest.fixture
def qa_evidence() -> dict[str, Any]:
    return load_fixture("qa-evidence.json")


def make_silent_flac(path: Path, duration: float = DURATION) -> None:
    completed = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={duration}",
            "-c:a", "flac", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(f"could not generate FLAC fixture: {completed.stderr[-500:]}")


def media_metadata(path: Path, *, has_video: bool = False, creation_time: str | None = CREATION_TIME) -> dict[str, Any]:
    return {
        "path": str(path),
        "duration_seconds": DURATION,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "has_audio": True,
        "has_video": has_video,
        "creation_time": creation_time,
    }


class FakeStages:
    """StageAdapters factory that records calls and fakes the expensive work."""

    def __init__(
        self,
        *,
        diarized: dict[str, Any] | None = None,
        has_video: bool = False,
        creation_time: str | None = CREATION_TIME,
        real_flac: bool | None = None,
    ) -> None:
        self.diarized = copy.deepcopy(diarized or load_fixture("diarized.json"))
        self.has_video = has_video
        self.creation_time = creation_time
        self.real_flac = HAVE_FFMPEG if real_flac is None else real_flac
        self.calls: list[str] = []

    def probe(self, path: Path) -> dict[str, Any]:
        self.calls.append("probe")
        return media_metadata(path, has_video=self.has_video, creation_time=self.creation_time)

    def extract(self, source: Path, destination: Path) -> dict[str, Any]:
        self.calls.append("extract")
        if self.real_flac:
            make_silent_flac(destination)
        else:
            destination.write_bytes(b"fLaC synthetic placeholder")
        return {"path": str(destination), "duration_difference_seconds": 0.0}

    def transcribe(self, audio: Path, output: Path) -> dict[str, Any]:
        self.calls.append("transcribe")
        output.write_text(json.dumps(self.diarized, indent=2, sort_keys=True), encoding="utf-8")
        return self.diarized

    def adapters(self, **overrides: Any) -> Any:
        from transcript_pipeline.cli import StageAdapters

        values: dict[str, Any] = {
            "probe_media": self.probe,
            "extract_media": self.extract,
            "transcribe": self.transcribe,
        }
        values.update(overrides)
        return StageAdapters(**values)


def run_cli(argv: list[str], adapters: Any = None) -> tuple[int, dict[str, Any] | None, str]:
    from transcript_pipeline.cli import main

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(argv, adapters=adapters, stdout=stdout, stderr=stderr)
    text = stdout.getvalue().strip()
    payload = json.loads(text.splitlines()[-1]) if text else None
    return code, payload, stderr.getvalue()


def prepare_run(tmp_path: Path, stages: FakeStages, *extra: str) -> Path:
    recording = tmp_path / "synthetic-session.mp4"
    if not recording.exists():
        recording.write_bytes(b"synthetic recording bytes")
    code, payload, _ = run_cli(
        ["prepare", str(recording), "--output-root", str(tmp_path / "outputs"), "--title", "Synthetic session", *extra],
        stages.adapters(),
    )
    assert payload is not None
    run_dir = Path(payload["run"])
    assert code in (0, 3)
    return run_dir


def write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def evidence_files(tmp_path: Path) -> Callable[..., tuple[Path, Path]]:
    def build(
        speaker: dict[str, Any] | None = None,
        qa: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        directory = tmp_path / "proposals"
        directory.mkdir(exist_ok=True)
        return (
            write_json(directory / "speakers.json", speaker or load_fixture("speaker-evidence.json")),
            write_json(directory / "qa.json", qa or load_fixture("qa-evidence.json")),
        )

    return build
