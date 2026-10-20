from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _run_ffmpeg(arguments: list[str]) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *arguments],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"could not generate media fixture: {completed.stderr[-1000:]}")


@pytest.fixture(scope="module")
def media_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")
    directory = tmp_path_factory.mktemp("media")
    mixed = directory / "mixed.mp4"
    audio_only = directory / "audio-only.wav"
    video_only = directory / "video-only.mp4"
    valid_flac = directory / "valid.flac"
    corrupt_frame_flac = directory / "corrupt-frame.flac"

    _run_ffmpeg(
        [
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=25:duration=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=5",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-metadata",
            "creation_time=2026-07-17T19:30:00Z",
            "-shortest",
            str(mixed),
        ]
    )
    _run_ffmpeg(
        [
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=22050:duration=2",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "pcm_s16le",
            str(audio_only),
        ]
    )
    _run_ffmpeg(
        [
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x72:rate=10:duration=2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video_only),
        ]
    )
    _run_ffmpeg(
        [
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=2",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            str(valid_flac),
        ]
    )
    shutil.copyfile(valid_flac, corrupt_frame_flac)
    with corrupt_frame_flac.open("r+b") as corrupt_file:
        corrupt_file.truncate(valid_flac.stat().st_size * 3 // 4)
    return {
        "mixed": mixed,
        "audio_only": audio_only,
        "video_only": video_only,
        "valid_flac": valid_flac,
        "corrupt_frame_flac": corrupt_frame_flac,
    }


def test_probe_media_reports_audio_video_and_embedded_creation_time(
    media_files: dict[str, Path],
) -> None:
    from transcript_pipeline.media import probe_media

    metadata = probe_media(media_files["mixed"])

    assert metadata["duration_seconds"] == pytest.approx(5.0, abs=0.15)
    assert "mp4" in metadata["format"]["name"]
    assert metadata["has_audio"] is True
    assert metadata["audio_stream_count"] == 1
    assert metadata["audio"] is not None
    assert metadata["audio"]["codec_name"] == "aac"
    assert metadata["audio"]["sample_rate"] == 48_000
    assert metadata["audio"]["channels"] == 2
    assert metadata["audio"]["duration_seconds"] == pytest.approx(5.0, abs=0.15)
    assert metadata["has_video"] is True
    assert metadata["video_stream_count"] == 1
    assert metadata["video"] is not None
    assert metadata["video"]["codec_name"] == "h264"
    assert metadata["video"]["width"] == 160
    assert metadata["video"]["height"] == 90
    assert metadata["video"]["avg_frame_rate_fps"] == pytest.approx(25.0)
    assert metadata["video"]["r_frame_rate_fps"] == pytest.approx(25.0)
    assert metadata["creation_time"] == "2026-07-17T19:30:00.000000Z"
    assert metadata["creation_time_source"] == "format"
    assert [stream["codec_type"] for stream in metadata["streams"]] == [
        "video",
        "audio",
    ]


def test_probe_media_reports_audio_only_file(
    media_files: dict[str, Path],
) -> None:
    from transcript_pipeline.media import probe_media

    metadata = probe_media(media_files["audio_only"])

    assert metadata["has_audio"] is True
    assert metadata["audio_stream_count"] == 1
    assert metadata["audio"] is not None
    assert metadata["audio"]["codec_name"] == "pcm_s16le"
    assert metadata["audio"]["sample_rate"] == 22_050
    assert metadata["audio"]["channels"] == 1
    assert metadata["has_video"] is False
    assert metadata["video_stream_count"] == 0
    assert metadata["video"] is None


def test_probe_media_tolerates_zero_and_unavailable_rational_frame_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    source = tmp_path / "odd-rate.mp4"
    source.touch()
    payload = {
        "format": {"format_name": "mov,mp4", "duration": "1.0", "tags": {}},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 64,
                "height": 64,
                "duration": "1.0",
                "avg_frame_rate": "0/0",
                "r_frame_rate": "N/A",
                "tags": {"creation_time": "2026-07-17T19:30:00Z"},
            }
        ],
    }
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(media.subprocess, "run", fake_run)

    metadata = media.probe_media(source)

    assert metadata["video"] is not None
    assert metadata["video"]["avg_frame_rate"] == "0/0"
    assert metadata["video"]["avg_frame_rate_fps"] is None
    assert metadata["video"]["r_frame_rate"] == "N/A"
    assert metadata["video"]["r_frame_rate_fps"] is None
    assert metadata["video"]["frame_rate_fps"] is None
    assert metadata["creation_time"] == "2026-07-17T19:30:00Z"
    assert metadata["creation_time_source"] == "stream:0"
    assert calls[0][0][-1] == str(source)
    assert calls[0][1]["shell"] is False


def test_probe_media_rejects_missing_input(tmp_path: Path) -> None:
    from transcript_pipeline.media import MediaError, probe_media

    missing = tmp_path / "token=do-not-leak.mp4"

    with pytest.raises(MediaError) as captured:
        probe_media(missing)

    assert "does not exist" in str(captured.value)
    assert "do-not-leak" not in str(captured.value)


def test_probe_media_bounds_and_redacts_ffprobe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    source = tmp_path / "broken.mp4"
    source.touch()
    secret = "diagnostic-secret"

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr=f"token={secret} " + ("x" * 10_000),
        )

    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(media.MediaError) as captured:
        media.probe_media(source)

    message = str(captured.value)
    assert "ffprobe failed" in message
    assert secret not in message
    assert "[REDACTED]" in message
    assert len(message) <= media.DIAGNOSTIC_LIMIT


def test_probe_media_rejects_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    source = tmp_path / "malformed.mp4"
    source.touch()

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="not ffprobe json",
            stderr="",
        )

    monkeypatch.setattr(media.subprocess, "run", fake_run)

    with pytest.raises(media.MediaError, match="malformed JSON"):
        media.probe_media(source)


def test_extract_canonical_flac_produces_verified_hashed_artifact(
    media_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    from transcript_pipeline.media import extract_canonical_flac, probe_media, sha256_file

    destination = tmp_path / "audio.flac"

    artifact = extract_canonical_flac(media_files["mixed"], destination)

    assert destination.is_file()
    assert artifact["path"] == str(destination)
    assert artifact["sha256"] == sha256_file(destination)
    assert artifact["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert artifact["metadata"] == probe_media(destination)
    assert artifact["metadata"]["format"]["name"] == "flac"
    assert artifact["metadata"]["has_video"] is False
    assert artifact["metadata"]["audio"] is not None
    assert artifact["metadata"]["audio"]["codec_name"] == "flac"
    assert artifact["metadata"]["audio"]["sample_rate"] == 16_000
    assert artifact["metadata"]["audio"]["channels"] == 1
    assert artifact["duration_difference_seconds"] <= 0.25
    assert artifact["metadata"]["audio"]["duration_seconds"] == pytest.approx(
        artifact["source_audio_duration_seconds"],
        abs=0.25,
    )


def test_validate_canonical_flac_fully_decodes_without_mutating_input(
    media_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    corrupt_flac = media_files["corrupt_frame_flac"]
    metadata = media.probe_media(corrupt_flac)
    before_hash = media.sha256_file(corrupt_flac)
    real_run = subprocess.run
    calls: list[tuple[list[str], dict[str, object]]] = []

    def recording_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", recording_run)

    assert metadata["format"]["name"] == "flac"
    assert metadata["audio"] is not None
    assert metadata["audio"]["sample_rate"] == 16_000
    assert metadata["audio"]["channels"] == 1
    assert metadata["audio"]["duration_seconds"] == pytest.approx(2.0)
    with pytest.raises(media.MediaError, match="decode failed"):
        media.validate_canonical_flac(corrupt_flac)

    assert corrupt_flac.is_file()
    assert media.sha256_file(corrupt_flac) == before_hash
    decode_commands = [
        (command, kwargs)
        for command, kwargs in calls
        if command[:4] == ["ffmpeg", "-nostdin", "-v", "error"]
    ]
    assert decode_commands == [
        (
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-progress",
                "pipe:1",
                "-nostats",
                "-i",
                str(corrupt_flac),
                "-f",
                "null",
                "-",
            ],
            {
                "check": False,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "shell": False,
            },
        )
    ]


def test_validate_canonical_flac_redacts_and_bounds_decode_failure(
    media_files: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    real_run = subprocess.run
    secret = "decode-secret"

    def failing_decode(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[:4] == ["ffmpeg", "-nostdin", "-v", "error"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=f"token={secret} " + ("z" * 10_000),
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", failing_decode)

    with pytest.raises(media.MediaError) as captured:
        media.validate_canonical_flac(media_files["valid_flac"])

    message = str(captured.value)
    assert "decode failed" in message
    assert secret not in message
    assert "[REDACTED]" in message
    assert len(message) <= media.DIAGNOSTIC_LIMIT


def test_extract_canonical_flac_rejects_noncanonical_destination(
    media_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    from transcript_pipeline.media import MediaError, extract_canonical_flac

    with pytest.raises(MediaError, match="audio.flac"):
        extract_canonical_flac(media_files["audio_only"], tmp_path / "other.flac")


def test_extract_rejects_video_without_audio_before_running_ffmpeg(
    media_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    real_run = subprocess.run
    commands: list[list[str]] = []

    def recording_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", recording_run)

    with pytest.raises(media.MediaError, match="no audio stream"):
        media.extract_canonical_flac(
            media_files["video_only"],
            tmp_path / "audio.flac",
        )

    assert [Path(command[0]).name for command in commands] == ["ffprobe"]
    assert not (tmp_path / "audio.flac").exists()


def test_extract_passes_weird_filename_as_one_non_shell_argument(
    media_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    source = tmp_path / "audio;$(printf-injected);token=fake-secret.wav"
    shutil.copyfile(media_files["audio_only"], source)
    destination = tmp_path / "audio.flac"
    real_run = subprocess.run
    calls: list[tuple[list[str], dict[str, object]]] = []

    def recording_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", recording_run)

    media.extract_canonical_flac(source, destination)

    assert all(kwargs["shell"] is False for _, kwargs in calls)
    extraction_commands = [
        command
        for command, _ in calls
        if command[0] == "ffmpeg" and "-c:a" in command
    ]
    assert extraction_commands == [
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            str(destination),
        ]
    ]


def test_failed_extraction_removes_partial_output_and_redacts_bounded_stderr(
    media_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    destination = tmp_path / "audio.flac"
    real_run = subprocess.run
    secret = "ffmpeg-secret"

    def failing_ffmpeg(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "ffmpeg":
            destination.write_bytes(b"partial output")
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=f"password={secret} " + ("y" * 10_000),
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", failing_ffmpeg)

    with pytest.raises(media.MediaError) as captured:
        media.extract_canonical_flac(media_files["audio_only"], destination)

    message = str(captured.value)
    assert not destination.exists()
    assert "ffmpeg failed" in message
    assert secret not in message
    assert "[REDACTED]" in message
    assert len(message) <= media.DIAGNOSTIC_LIMIT


def test_extraction_removes_valid_header_flac_with_corrupt_frames(
    media_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    destination = tmp_path / "audio.flac"
    real_run = subprocess.run

    def corrupting_extraction(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "ffmpeg" and "-c:a" in command:
            shutil.copyfile(media_files["corrupt_frame_flac"], destination)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return real_run(command, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", corrupting_extraction)

    with pytest.raises(media.MediaError, match="decode failed"):
        media.extract_canonical_flac(media_files["audio_only"], destination)

    assert not destination.exists()


def test_sha256_file_hashes_large_file_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcript_pipeline.media as media

    artifact = tmp_path / "large.bin"
    artifact.write_bytes((b"0123456789abcdef" * 70_000) + b"tail")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    original_open = Path.open
    requested_read_sizes: list[int] = []

    class TrackingReader:
        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def __enter__(self) -> "TrackingReader":
            self.wrapped.__enter__()  # type: ignore[union-attr]
            return self

        def __exit__(self, *args: object) -> object:
            return self.wrapped.__exit__(*args)  # type: ignore[union-attr]

        def read(self, size: int = -1) -> bytes:
            requested_read_sizes.append(size)
            return self.wrapped.read(size)  # type: ignore[union-attr,no-any-return]

    def tracking_open(path: Path, *args: object, **kwargs: object) -> TrackingReader:
        return TrackingReader(original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", tracking_open)

    assert media.sha256_file(artifact) == expected
    assert requested_read_sizes
    assert all(0 < size <= 1024 * 1024 for size in requested_read_sizes)
    assert len(requested_read_sizes) >= 2


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "message"),
    [
        (0, "out_time_us=1088000\nprogress=end\n", "", "decoded 1.088s of 2.000s"),
        (0, "out_time_us=1984000\nprogress=end\n", "Invalid data found", "decode failed"),
        (0, "progress=end\n", "", "no progress report"),
        (1, "out_time_us=2000000\n", "", "decode failed"),
    ],
)
def test_decode_check_does_not_trust_exit_code_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    stdout: str,
    stderr: str,
    message: str,
) -> None:
    import transcript_pipeline.media as media

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    with pytest.raises(media.MediaError, match=message):
        media._verify_complete_decode(tmp_path / "audio.flac", 2.0)


def test_decode_check_accepts_last_block_rounding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import transcript_pipeline.media as media

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        progress = "out_time_us=1000000\nprogress=continue\nout_time_us=1984000\nprogress=end\n"
        return subprocess.CompletedProcess(command, 0, stdout=progress, stderr="")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    media._verify_complete_decode(tmp_path / "audio.flac", 2.0)
