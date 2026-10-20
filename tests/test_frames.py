from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


PANE_REGIONS = {
    "speakers": (0.0, 0.12, 0.20, 0.921),
    "slides": (0.20, 0.06, 0.79, 0.94),
    "notes": (0.79, 0.0, 1.0, 0.94),
}


def _target_at(targets: list[dict[str, object]], timestamp: float) -> dict[str, object]:
    return next(
        target
        for target in targets
        if float(target["timestamp_seconds"]) == pytest.approx(timestamp, abs=0.001)
    )


def _make_video(path: Path, *, duration: float = 1.2) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x90:rate=7:duration={duration}",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=True,
        shell=False,
    )


def test_merge_adjacent_speaker_subsegments_into_one_bounded_turn() -> None:
    from transcript_pipeline.frames import merge_adjacent_speaker_segments, select_frame_targets

    segments = [
        {
            "id": f"seg-{index:06d}",
            "start": index / 10,
            "end": (index + 1) / 10,
            "speaker": "SPEAKER_00",
            "text": "word",
        }
        for index in range(1_000)
    ]

    turns = merge_adjacent_speaker_segments(segments)

    assert turns == [
        {
            "id": "turn:seg-000000:seg-000999",
            "start": 0.0,
            "end": 100.0,
            "speaker": "SPEAKER_00",
            "source_refs": ["seg-000000", "seg-000999"],
            "segment_count": 1_000,
        }
    ]
    targets = select_frame_targets(100, segments=turns, periodic_interval=60)
    turn_targets = [
        target
        for target in targets
        if any(str(reason).startswith("turn_") for reason in target["reasons"])
    ]
    assert len(turn_targets) <= 9
    assert all(
        target["source_refs"] == ["seg-000000", "seg-000999"]
        for target in turn_targets
    )


def test_merge_splits_resumed_same_speaker_after_five_second_silence() -> None:
    from transcript_pipeline.frames import merge_adjacent_speaker_segments

    turns = merge_adjacent_speaker_segments(
        [
            {"id": "seg-1", "start": 10.0, "end": 11.0, "speaker": "SPEAKER_00"},
            {"id": "seg-2", "start": 11.5, "end": 12.0, "speaker": "SPEAKER_00"},
            {"id": "seg-3", "start": 17.001, "end": 18.0, "speaker": "SPEAKER_00"},
        ]
    )

    assert [turn["source_refs"] for turn in turns] == [
        ["seg-1", "seg-2"],
        ["seg-3"],
    ]
    assert turns[1]["start"] == 17.001


def test_select_targets_combines_turn_boundaries_neighborhoods_and_periodic() -> None:
    from transcript_pipeline.frames import select_frame_targets

    targets = select_frame_targets(
        100,
        segments=[{"id": "turn-1", "start": 10.0, "end": 20.0}],
        mentions=[{"id": "mention-1", "timestamp": 40.0, "name": "Ada"}],
        events=[{"id": "event-1", "timestamp_seconds": 70.0}],
        periodic_interval=50,
    )

    timestamps = [target["timestamp_seconds"] for target in targets]
    assert timestamps == sorted(timestamps)
    for expected in (0, 10, 13, 15, 17, 20, 35, 38, 40, 42, 45):
        assert any(value == pytest.approx(expected, abs=0.001) for value in timestamps)
    for expected in (50, 65, 68, 70, 72, 75, 100):
        assert any(value == pytest.approx(expected, abs=0.001) for value in timestamps)

    mention = _target_at(targets, 40)
    assert "spoken_name" in mention["reasons"]
    assert "mention-1" in mention["source_refs"]
    assert mention["priority"] > _target_at(targets, 50)["priority"]
    assert "turn_start_adjacent" in _target_at(targets, 10)["reasons"]
    assert "turn_end_adjacent" in _target_at(targets, 20)["reasons"]


def test_select_targets_clamps_deduplicates_stably_and_skips_nonfinite_records() -> None:
    from transcript_pipeline.frames import select_frame_targets

    targets = select_frame_targets(
        10,
        segments=[
            {"id": "bad", "start": math.nan, "end": math.inf},
            {"id": "clamped", "start": -20, "end": 99},
        ],
        events=[
            {"id": "first", "timestamp": 5.0},
            {"id": "second", "timestamp": 5.04},
            {"id": "invalid", "timestamp": "not-a-time"},
        ],
        periodic_interval=30,
        dedupe_tolerance=0.05,
    )

    assert all(0 <= float(target["timestamp_seconds"]) <= 10 for target in targets)
    merged = _target_at(targets, 5)
    assert merged["source_refs"][-2:] == ["first", "second"]
    assert sum(4.99 <= float(t["timestamp_seconds"]) <= 5.05 for t in targets) == 1

    with pytest.raises(ValueError, match="duration"):
        select_frame_targets(math.inf)
    with pytest.raises(ValueError, match="periodic"):
        select_frame_targets(10, periodic_interval=61)


def test_event_at_2700_gets_two_and_five_second_neighbors() -> None:
    from transcript_pipeline.frames import select_frame_targets

    targets = select_frame_targets(
        3_000,
        events=[
            {
                "id": "slide-change-12",
                "timestamp_seconds": 2_700,
                "type": "slide_change",
            }
        ],
        periodic_interval=60,
    )

    timestamps = {target["timestamp_seconds"] for target in targets}
    assert {2_695.0, 2_698.0, 2_700.0, 2_702.0, 2_705.0} <= timestamps
    event_target = _target_at(targets, 2_700)
    assert "slide-change-12" in event_target["source_refs"]
    assert "slide_change" in event_target["reasons"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_frames_from_real_video_keeps_full_resolution(tmp_path: Path) -> None:
    from transcript_pipeline.frames import extract_frames

    video = tmp_path / "tiny video.mp4"
    _make_video(video)
    records = extract_frames(
        video,
        [
            {
                "timestamp_seconds": 0.25,
                "reasons": ["periodic"],
                "source_refs": [],
                "priority": 10,
            }
        ],
        tmp_path / ".work" / "frames",
    )

    raw_path = Path(records[0]["raw_path"])
    assert raw_path.name == "frame_000000250.png"
    assert records[0]["frame_id"] == "frame_000000250"
    assert Image.open(raw_path).size == (160, 90)
    assert records[0]["status"] == "ok"


def test_extract_frames_uses_second_seek_list_command_and_safe_weird_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    video = tmp_path / "video;$(touch owned) [x].mp4"
    video.write_bytes(b"not decoded by fake")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Image.new("RGB", (17, 11), "red").save(command[-1])
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)
    records = frames.extract_frames(
        video,
        [{"timestamp_seconds": 1.234, "reasons": ["event"]}],
        tmp_path / "frames",
    )

    command = commands[0]
    assert command[command.index("-ss") + 1] == "1.234"
    assert command[command.index("-i") + 1] == str(video)
    assert command[command.index("-frames:v") + 1] == "1"
    assert not any("fps" in argument or "select=" in argument for argument in command)
    assert records[0]["status"] == "ok"
    assert not (tmp_path / "owned").exists()


def test_extract_frames_reuses_valid_exact_existing_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "frames"
    output.mkdir()
    existing = output / "frame_000001250.png"
    Image.new("RGB", (32, 18), "green").save(existing)
    before = existing.read_bytes()

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ffmpeg must not run for a valid exact existing frame")

    monkeypatch.setattr(frames.subprocess, "run", unexpected_run)

    records = frames.extract_frames(
        video,
        [{"timestamp_seconds": 1.25, "reasons": ["turn_midpoint"]}],
        output,
    )

    assert records[0]["status"] == "ok"
    assert records[0]["frame_id"] == "frame_000001250"
    assert records[0]["raw_path"] == str(existing)
    assert existing.read_bytes() == before


def test_extract_frames_reextracts_corrupt_existing_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "frames"
    output.mkdir()
    existing = output / "frame_000001250.png"
    existing.write_bytes(b"not-a-png")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Image.new("RGB", (32, 18), "blue").save(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)

    records = frames.extract_frames(
        video,
        [{"timestamp_seconds": 1.25, "reasons": ["turn_midpoint"]}],
        output,
    )

    assert len(commands) == 1
    assert records[0]["status"] == "ok"
    with Image.open(existing) as reextracted:
        assert reextracted.size == (32, 18)


def test_prune_visual_derivatives_removes_only_safe_unreferenced_leaves(
    tmp_path: Path,
) -> None:
    from transcript_pipeline.frames import prune_unreferenced_visual_derivatives

    work = tmp_path / ".work"
    frames = work / "frames"
    ocr = frames / "ocr"
    sheets = frames / "contact_sheets"
    ocr.mkdir(parents=True)
    sheets.mkdir()
    retained_raw = frames / "frame_000001000.png"
    stale_raw = frames / "frame_000002000.png"
    retained_crop = ocr / "frame_000001000__speakers_ocr2x.png"
    stale_crop = ocr / "frame_000002000__speakers_ocr2x.png"
    retained_sheet = sheets / "contact_sheet_001.jpg"
    stale_sheet = sheets / "contact_sheet_002.jpg"
    unknown = frames / "operator-notes.txt"
    for path in (
        retained_raw,
        stale_raw,
        retained_crop,
        stale_crop,
        retained_sheet,
        stale_sheet,
        unknown,
    ):
        path.write_bytes(b"managed")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    stale_raw.unlink()
    stale_raw.symlink_to(outside)
    document = {
        "sources": [
            {
                "raw_path": str(retained_raw),
                "ocr": [{"crop_path": str(retained_crop)}],
            }
        ],
        "contact_sheets": [str(retained_sheet)],
    }

    removed = prune_unreferenced_visual_derivatives(
        document, frames, managed_root=work
    )

    assert removed == {"raw_frames": 1, "ocr_crops": 1, "contact_sheets": 1}
    assert retained_raw.is_file()
    assert retained_crop.is_file()
    assert retained_sheet.is_file()
    assert unknown.read_bytes() == b"managed"
    assert not stale_raw.exists()
    assert not stale_crop.exists()
    assert not stale_sheet.exists()
    assert outside.read_bytes() == b"outside"


def test_extract_frames_records_one_failure_and_continues_with_redacted_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    video = tmp_path / "video.mp4"
    video.touch()
    sentinel = "frame-secret-2789"
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                command, 1, "", f"token={sentinel} " + "x" * 2_000
            )
        Image.new("RGB", (10, 10)).save(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)
    records = frames.extract_frames(
        video,
        [
            {"timestamp_seconds": 0.1, "reasons": ["a"]},
            {"timestamp_seconds": 0.2, "reasons": ["b"]},
        ],
        tmp_path / "frames",
    )

    assert [record["status"] for record in records] == ["error", "ok"]
    assert [record["frame_id"] for record in records] == [
        "frame_000000100",
        "frame_000000200",
    ]
    assert sentinel not in records[0]["error"]
    assert "[REDACTED]" in records[0]["error"]
    assert len(records[0]["error"]) <= 600


def test_extract_frames_raises_only_after_every_target_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    video = tmp_path / "video.mp4"
    video.touch()
    monkeypatch.setattr(
        frames.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "bad"),
    )

    with pytest.raises(frames.FrameExtractionError, match="all 2 frame extractions failed") as caught:
        frames.extract_frames(
            video,
            [{"timestamp_seconds": 1}, {"timestamp_seconds": 2}],
            tmp_path / "frames",
        )

    assert len(caught.value.records) == 2


def test_frame_artifact_writers_reject_symlinked_output_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    raw = tmp_path / "raw.png"
    Image.new("RGB", (160, 90), "navy").save(raw)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Image.new("RGB", (16, 9), "red").save(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)
    operations = (
        lambda output: frames.extract_frames(
            video, [{"timestamp_seconds": 0.1}], output
        ),
        lambda output: frames.create_ocr_crops(raw, output, regions=PANE_REGIONS),
        lambda output: frames.create_contact_sheets(
            [
                {
                    "timestamp_seconds": 0.1,
                    "reasons": ["periodic"],
                    "raw_path": str(raw),
                    "status": "ok",
                }
            ],
            output,
        ),
    )
    for index, operation in enumerate(operations):
        external = tmp_path / f"external-{index}"
        external.mkdir()
        sentinel = external / "sentinel.txt"
        sentinel.write_text("untouched\n", encoding="utf-8")
        managed = tmp_path / f"managed-{index}"
        managed.symlink_to(external, target_is_directory=True)

        with pytest.raises(ValueError, match="symlink"):
            operation(managed)

        assert sentinel.read_text(encoding="utf-8") == "untouched\n"
        assert list(external.iterdir()) == [sentinel]


def test_ocr_regions_scale_and_clamp_to_frame_geometry() -> None:
    from transcript_pipeline.frames import get_crop_specs

    assert get_crop_specs(1278, 720) == {"full": (0, 0, 1278, 720)}
    specs = get_crop_specs(1278, 720, PANE_REGIONS)
    assert specs == {
        "speakers": (0, 86, 256, 663),
        "slides": (256, 43, 1010, 677),
        "notes": (1010, 0, 1278, 677),
    }
    for left, top, right, bottom in specs.values():
        assert 0 <= left < right <= 1278
        assert 0 <= top < bottom <= 720


@pytest.mark.parametrize(
    "regions",
    [
        {},
        {"bad name": (0, 0, 1, 1)},
        {"inverted": (0.5, 0, 0.2, 1)},
        {"outside": (0, 0, 1.5, 1)},
        {"short": (0, 0, 1)},
    ],
)
def test_invalid_ocr_regions_are_rejected(regions: dict[str, object]) -> None:
    from transcript_pipeline.frames import get_crop_specs

    with pytest.raises(ValueError):
        get_crop_specs(100, 100, regions)  # type: ignore[arg-type]


def test_ocr_crops_are_upscaled_without_mutating_raw_frame(tmp_path: Path) -> None:
    from transcript_pipeline.frames import create_ocr_crops

    raw = tmp_path / "frame.png"
    Image.new("RGB", (1278, 720), "navy").save(raw)
    before = raw.read_bytes()

    crops = create_ocr_crops(raw, tmp_path / "ocr", regions=PANE_REGIONS)

    assert raw.read_bytes() == before
    assert {crop["region"] for crop in crops} == {"speakers", "slides", "notes"}
    assert Image.open(crops[0]["path"]).size == (
        (crops[0]["box"][2] - crops[0]["box"][0]) * 2,
        (crops[0]["box"][3] - crops[0]["box"][1]) * 2,
    )
    assert all(Path(crop["path"]) != raw for crop in crops)


def test_ocr_crop_replaces_leaf_symlink_without_overwriting_target(
    tmp_path: Path,
) -> None:
    from transcript_pipeline.frames import create_ocr_crops

    raw = tmp_path / "frame.png"
    Image.new("RGB", (160, 90), "navy").save(raw)
    output = tmp_path / "ocr"
    output.mkdir()
    external = tmp_path / "external.png"
    external.write_bytes(b"outside-sentinel")
    destination = output / "frame__speakers_ocr2x.png"
    destination.symlink_to(external)

    crops = create_ocr_crops(raw, output, regions=PANE_REGIONS)

    assert external.read_bytes() == b"outside-sentinel"
    assert not destination.is_symlink()
    assert destination.is_file()
    with Image.open(destination) as rendered:
        assert rendered.format == "PNG"
    assert crops[0]["path"] == str(destination)


def test_tesseract_records_text_and_tsv_confidence_per_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (80, 40), "white").save(crop_path)
    crop = {"region": "speakers", "path": str(crop_path), "box": [0, 0, 40, 20]}
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t20\t10\t91.5\tAda\n"
            "5\t1\t1\t1\t1\t2\t21\t0\t20\t10\t88.5\tExample\n"
        )
        return subprocess.CompletedProcess(command, 0, tsv, "")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)
    results = frames.run_tesseract_ocr([crop])

    assert commands == [["tesseract", str(crop_path), "stdout", "tsv"]]
    assert results[0]["status"] == "ok"
    assert results[0]["text"] == "Ada Example"
    assert results[0]["confidence"] == pytest.approx(0.9)


def test_tesseract_failure_keeps_crop_and_records_bounded_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (80, 40)).save(crop_path)
    monkeypatch.setattr(
        frames.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", "password=hunter2 " + "z" * 2_000
        ),
    )

    result = frames.run_tesseract_ocr(
        [{"region": "notes", "path": str(crop_path), "box": [0, 0, 40, 20]}]
    )[0]

    assert crop_path.exists()
    assert result["status"] == "error"
    assert "hunter2" not in result["error"]
    assert len(result["error"]) <= 600


def test_contact_sheets_use_raw_thumbnails_label_and_paginate_without_mutation(
    tmp_path: Path,
) -> None:
    from transcript_pipeline.frames import create_contact_sheets

    records = []
    original_bytes = []
    for index in range(5):
        path = tmp_path / f"raw-{index}.png"
        image = Image.new("RGB", (160, 90), (index * 30, 50, 100))
        ImageDraw.Draw(image).text((2, 2), str(index), fill="white")
        image.save(path)
        original_bytes.append(path.read_bytes())
        records.append(
            {
                "timestamp_seconds": index + 0.125,
                "reasons": ["turn_midpoint", "spoken_name"],
                "raw_path": str(path),
                "status": "ok",
            }
        )

    sheets = create_contact_sheets(
        records,
        tmp_path / "sheets",
        columns=2,
        page_size=4,
        thumbnail_size=(120, 68),
    )

    assert [Path(path).name for path in sheets] == [
        "contact_sheet_001.jpg",
        "contact_sheet_002.jpg",
    ]
    assert all(Path(path).is_file() for path in sheets)
    assert [Path(record["raw_path"]).read_bytes() for record in records] == original_bytes


def test_contact_sheet_replaces_leaf_symlink_without_overwriting_target(
    tmp_path: Path,
) -> None:
    from transcript_pipeline.frames import create_contact_sheets

    raw = tmp_path / "raw.png"
    Image.new("RGB", (160, 90), "navy").save(raw)
    output = tmp_path / "sheets"
    output.mkdir()
    external = tmp_path / "external.jpg"
    external.write_bytes(b"outside-sentinel")
    destination = output / "contact_sheet_001.jpg"
    destination.symlink_to(external)

    sheets = create_contact_sheets(
        [
            {
                "timestamp_seconds": 1.25,
                "reasons": ["turn_midpoint"],
                "raw_path": str(raw),
                "status": "ok",
            }
        ],
        output,
    )

    assert external.read_bytes() == b"outside-sentinel"
    assert not destination.is_symlink()
    assert destination.is_file()
    with Image.open(destination) as rendered:
        assert rendered.format == "JPEG"
    assert sheets == [str(destination)]


@pytest.mark.parametrize("artifact", ["ocr", "contact_sheet"])
def test_image_artifact_atomic_replace_failure_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    import transcript_pipeline.frames as frames

    raw = tmp_path / "raw.png"
    Image.new("RGB", (160, 90), "navy").save(raw)
    output = tmp_path / artifact
    output.mkdir()
    if artifact == "ocr":
        destination = output / "raw__speakers_ocr2x.png"
    else:
        destination = output / "contact_sheet_001.jpg"
    destination.write_bytes(b"existing-artifact")

    monkeypatch.setattr(
        frames.os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("interrupted")),
    )

    with pytest.raises(OSError, match="interrupted"):
        if artifact == "ocr":
            frames.create_ocr_crops(raw, output, regions=PANE_REGIONS)
        else:
            frames.create_contact_sheets(
                [
                    {
                        "timestamp_seconds": 1.25,
                        "reasons": ["turn_midpoint"],
                        "raw_path": str(raw),
                        "status": "ok",
                    }
                ],
                output,
            )

    assert destination.read_bytes() == b"existing-artifact"
    assert not list(output.glob(f".{destination.name}.*.tmp"))


def test_presence_intervals_require_repeated_normalized_exact_observations() -> None:
    from transcript_pipeline.frames import PresenceObservation, build_presence_intervals

    observations = [
        PresenceObservation(
            timestamp_seconds=10,
            exact_visible_name="Ada Example",
            canonical_name="ada",
            region="speakers",
            evidence_type="visible_label",
            confidence=0.96,
        ),
        PresenceObservation(
            timestamp_seconds=18,
            exact_visible_name="  ada   example ",
            canonical_name="ada",
            region="speakers",
            evidence_type="speaking_indicator",
            confidence=0.9,
        ),
        PresenceObservation(
            timestamp_seconds=50,
            exact_visible_name="Ada Example",
            canonical_name="ada",
            region="participants",
            evidence_type="visible_label",
            confidence=0.91,
        ),
    ]

    model = build_presence_intervals(observations, max_gap_seconds=15)

    assert model["intervals"] == [
        {
            "exact_visible_name": "ada example",
            "canonical_name": "ada",
            "start_seconds": 10.0,
            "end_seconds": 18.0,
            "confidence": 0.93,
            "observation_count": 2,
            "observation_indices": [0, 1],
        }
    ]
    assert model["isolated"][0]["timestamp_seconds"] == 50.0
    assert model["isolated"][0]["confidence"] <= 0.49


def test_ambiguous_truncated_ocr_and_absence_do_not_create_departure_intervals() -> None:
    from transcript_pipeline.frames import build_presence_intervals

    observations = [
        {
            "timestamp_seconds": 1,
            "observed_text": "Vic...",
            "truncated": True,
            "region": "notes",
            "evidence_type": "OCR",
            "confidence": 0.8,
        },
        {
            "timestamp_seconds": 3,
            "canonical_name": "ada",
            "exact_visible_name": "Ada",
            "region": "speakers",
            "evidence_type": "visible_label",
            "confidence": 0.9,
        },
        {
            "timestamp_seconds": 4,
            "canonical_name": "ada",
            "region": "speakers",
            "evidence_type": "absence",
            "confidence": 1.0,
        },
    ]

    model = build_presence_intervals(observations, max_gap_seconds=10)

    assert model["intervals"] == []
    assert len(model["isolated"]) == 2
    assert all(item["confidence"] <= 0.49 for item in model["isolated"])
    assert model["departures"] == []


def test_presence_does_not_merge_different_exact_labels_via_canonical_metadata() -> None:
    from transcript_pipeline.frames import build_presence_intervals

    model = build_presence_intervals(
        [
            {
                "timestamp_seconds": 10,
                "exact_visible_name": "Alice",
                "canonical_name": "ada",
                "region": "speakers",
                "evidence_type": "visible_label",
                "confidence": 0.95,
            },
            {
                "timestamp_seconds": 15,
                "exact_visible_name": "Bob",
                "canonical_name": "ada",
                "region": "speakers",
                "evidence_type": "speaking_indicator",
                "confidence": 0.94,
            },
        ],
        max_gap_seconds=10,
    )

    assert model["intervals"] == []
    assert [item["exact_visible_name"] for item in model["isolated"]] == [
        "Alice",
        "Bob",
    ]
    assert all(item["canonical_name"] == "ada" for item in model["isolated"])


def test_presence_intervals_sort_same_start_with_nullable_canonical_metadata() -> None:
    from transcript_pipeline.frames import build_presence_intervals

    observations = []
    for timestamp, label, canonical in (
        (10, "Alice", None),
        (10, "Bob", "bob"),
        (15, "Alice", None),
        (15, "Bob", "bob"),
    ):
        observations.append(
            {
                "timestamp_seconds": timestamp,
                "exact_visible_name": label,
                "canonical_name": canonical,
                "region": "speakers",
                "evidence_type": "visible_label",
                "confidence": 0.9,
            }
        )

    model = build_presence_intervals(observations, max_gap_seconds=10)

    assert len(model["intervals"]) == 2
    assert [interval["exact_visible_name"] for interval in model["intervals"]] == [
        "alice",
        "bob",
    ]
    assert [interval["canonical_name"] for interval in model["intervals"]] == [
        None,
        "bob",
    ]


def test_visual_evidence_document_is_json_ready_and_atomic_writer_is_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.frames as frames

    document = frames.build_visual_evidence(
        duration_seconds=12,
        frame_records=[
            {
                "timestamp_seconds": 2,
                "reasons": ["spoken_name"],
                "source_refs": ["mention-1"],
                "priority": 100,
                "raw_path": str(tmp_path / ".work/frames/frame_000002000.png"),
                "status": "ok",
                "crops": [{"region": "notes", "path": "crop.png"}],
                "ocr": [{"region": "notes", "status": "ok", "text": "Ada"}],
            }
        ],
        presence_observations=[],
        presence_model={"intervals": [], "isolated": [], "departures": []},
        contact_sheets=["contact_sheet_001.jpg"],
    )
    output = tmp_path / ".work" / "frames" / "visual_evidence.json"
    frames.write_visual_evidence(output, document)

    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert document["schema_version"] == 1
    assert document["sources"][0]["frame_id"] == "frame_000002000"
    assert document["sources"][0]["raw_path"].endswith("frame_000002000.png")

    previous = output.read_bytes()
    with pytest.raises(ValueError):
        frames.write_visual_evidence(output, {"invalid": math.nan})
    assert output.read_bytes() == previous

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(frames.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        frames.write_visual_evidence(output, {"schema_version": 1})
    assert output.read_bytes() == previous
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def _visual_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sources": [
            {"frame_id": "frame_000012000", "timestamp_seconds": 12.0, "status": "ok"},
            {"frame_id": "frame_000020000", "timestamp_seconds": 20.0, "status": "error"},
        ],
        "presence_observations": [],
    }


def _observation(**changes: object) -> dict[str, object]:
    observation: dict[str, object] = {
        "id": "obs-1",
        "frame_id": "frame_000012000",
        "timestamp_seconds": 12.0,
        "region": "speakers",
        "evidence_type": "speaking_indicator",
        "confidence": 0.9,
        "exact_visible_name": "Ada Example",
    }
    observation.update(changes)
    return observation


def test_attached_observations_are_bound_to_extracted_frames() -> None:
    from transcript_pipeline.frames import attach_presence_observations

    document = _visual_document()
    result = attach_presence_observations(
        document, [_observation(), _observation(id="obs-2", exact_visible_name="Ada Example")]
    )

    assert [item["id"] for item in result["presence_observations"]] == ["obs-1", "obs-2"]
    assert [
        (interval["exact_visible_name"], interval["observation_count"])
        for interval in result["presence"]["intervals"]
    ] == [("ada example", 2)]
    assert document["presence_observations"] == []


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"id": ""}, "unique"),
        ({"frame_id": "frame_000099000"}, "unknown frame"),
        ({"frame_id": "frame_000020000"}, "unknown frame"),
        ({"timestamp_seconds": 12.5}, "contradicts"),
        ({"region": "two words"}, "region"),
        ({"evidence_type": "speaking_outline"}, "evidence type"),
        ({"confidence": 1.5}, "confidence"),
    ],
)
def test_invalid_observations_are_rejected(changes: dict[str, object], match: str) -> None:
    from transcript_pipeline.frames import attach_presence_observations

    with pytest.raises(ValueError, match=match):
        attach_presence_observations(_visual_document(), [_observation(**changes)])


def test_duplicate_observation_ids_are_rejected() -> None:
    from transcript_pipeline.frames import attach_presence_observations

    with pytest.raises(ValueError, match="unique"):
        attach_presence_observations(_visual_document(), [_observation(), _observation()])
