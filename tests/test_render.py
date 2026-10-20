from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest

from conftest import load_fixture

SNAPSHOTS = Path(__file__).parent / "snapshots"
RESOLVED_EVENT = {
    "status": "resolved",
    "start_utc": "2026-03-14T09:30:00Z",
    "start_local": "2026-03-14T10:30:00+01:00",
    "local_date": "2026-03-14",
    "timezone": "Europe/Berlin",
}


def bundle(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "title": "Synthetic session",
        "event": copy.deepcopy(RESOLVED_EVENT),
        "duration_seconds": 56.0,
        "diarized": load_fixture("diarized.json"),
        "speaker_evidence": load_fixture("speaker-evidence.json"),
        "qa_evidence": load_fixture("qa-evidence.json"),
        "visual_evidence": None,
        "artifact_validation": {"valid": True, "issues": []},
    }
    document.update(changes)
    return document


def _snapshot(name: str) -> str:
    return (SNAPSHOTS / name).read_text(encoding="utf-8")


def test_renderers_match_approved_snapshots() -> None:
    from transcript_pipeline.render import render_qa, render_review, render_transcript

    data = bundle()

    assert render_transcript(data) == _snapshot("transcript.md")
    assert render_qa(data) == _snapshot("qa.md")
    assert render_review(data) == _snapshot("review.md")


def test_transcript_renders_every_segment_exactly_once_in_order() -> None:
    from transcript_pipeline.render import render_transcript

    data = bundle()
    text = render_transcript(data)

    positions = [text.index(segment["text"]) for segment in data["diarized"]["segments"]]
    assert positions == sorted(positions)
    for segment in data["diarized"]["segments"]:
        assert text.count(segment["text"]) == 1


def test_transcript_never_publishes_a_name_that_was_not_identified() -> None:
    from transcript_pipeline.render import render_qa, render_transcript

    data = bundle()
    guest = data["speaker_evidence"]["speakers"][1]
    guest.update(status="manual_review", display_name="Bo Sample", confidence="medium")

    for rendered in (render_transcript(data), render_qa(data)):
        assert "Bo Sample" not in rendered
        assert "Unidentified Speaker 1" in rendered


def test_reviewed_transcript_text_overrides_canonical_text_for_display() -> None:
    from transcript_pipeline.render import render_transcript

    data = bundle()
    data["diarized"]["segments"][3]["finalized_text"] = "For a 10 GB archive it takes about 40 minutes."

    text = render_transcript(data)

    assert "For a 10 GB archive it takes about 40 minutes." in text
    assert "ten gigabyte" not in text


def test_segments_without_a_speaker_get_a_non_colliding_fallback_label() -> None:
    from transcript_pipeline.render import collect_review_items, render_transcript

    data = bundle()
    data["diarized"]["segments"][7]["speaker"] = None

    text = render_transcript(data)

    assert "**[00:00:50] Unidentified Speaker 3:**" in text
    assert any(item["code"] == "segments_without_speaker" for item in collect_review_items(data))


def test_date_is_explicitly_unknown_without_a_resolved_event() -> None:
    from transcript_pipeline.render import render_transcript

    text = render_transcript(bundle(event={"status": "blocked"}))

    assert "**Date:** Not established" in text


def test_titles_are_markdown_escaped() -> None:
    from transcript_pipeline.render import render_transcript

    text = render_transcript(bundle(title="Q*3 [draft] _notes_"))

    assert text.splitlines()[0] == "# Transcript: Q\\*3 \\[draft\\] \\_notes\\_"


def test_review_is_ready_only_without_blocking_items() -> None:
    from transcript_pipeline.render import collect_review_items, render_review, review_is_ready

    ready = bundle()
    assert review_is_ready(render_review(ready))
    assert not [item for item in collect_review_items(ready) if item["blocking"]]

    pending = bundle()
    pending["speaker_evidence"]["speakers"][1].update(
        status="manual_review", display_name="Bo Sample", contradictions=["voice differs later"]
    )
    review = render_review(pending)
    assert not review_is_ready(review)
    assert "**speaker_manual_review:**" in review
    assert "Bo Sample" in review


def test_failed_artifact_validation_blocks_the_review() -> None:
    from transcript_pipeline.render import collect_review_items, render_review, review_is_ready

    data = bundle(
        artifact_validation={
            "valid": False,
            "issues": [{"code": "hash_mismatch", "path": "qa.md", "message": "artifact hash differs"}],
        }
    )

    review = render_review(data)
    assert not review_is_ready(review)
    assert "hash_mismatch" in review
    assert collect_review_items(data)[0]["code"] == "artifact_validation_failed"


def test_missing_speaker_record_is_blocking_even_if_upstream_said_valid() -> None:
    from transcript_pipeline.render import collect_review_items

    data = bundle()
    data["speaker_evidence"]["speakers"].pop()

    codes = {item["code"] for item in collect_review_items(data) if item["blocking"]}
    assert codes == {"speaker_record_missing"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["diarized"]["segments"][1].update(id="seg-000001"),
        lambda data: data["diarized"]["segments"][1].update(start=99.0, end=100.0),
        lambda data: data["diarized"]["segments"][1].update(end=1.0),
        lambda data: data["diarized"]["segments"][1].update(text="  "),
        lambda data: data["speaker_evidence"]["speakers"][1].update(output_label="Bo Sample"),
        lambda data: data["qa_evidence"]["topics"][0]["questions"].append(
            copy.deepcopy(data["qa_evidence"]["topics"][0]["questions"][0])
        ),
        lambda data: data["qa_evidence"]["topics"][0]["questions"][0].update(status="maybe"),
        lambda data: data.update(duration_seconds=None),
        lambda data: data.update(event={**RESOLVED_EVENT, "local_date": "2026-03-15"}),
    ],
)
def test_malformed_bundle_is_rejected_instead_of_synthesizing_content(
    mutate: Any, tmp_path: Path
) -> None:
    from transcript_pipeline.render import RenderValidationError, render_run_artifacts

    data = bundle()
    mutate(data)
    target = tmp_path / "run"

    with pytest.raises(RenderValidationError):
        render_run_artifacts(target, data)
    assert not target.exists()


def test_render_run_artifacts_writes_exactly_three_files(tmp_path: Path) -> None:
    from transcript_pipeline.render import (
        QA_FILENAME,
        REVIEW_FILENAME,
        TRANSCRIPT_FILENAME,
        render_run_artifacts,
    )

    written = render_run_artifacts(tmp_path, bundle())

    assert set(written) == {TRANSCRIPT_FILENAME, QA_FILENAME, REVIEW_FILENAME}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["qa.md", "review.md", "transcript.md"]


def test_atomic_writer_preserves_old_text_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.render as render

    output = tmp_path / "artifact.txt"
    output.write_text("old\n", encoding="utf-8")

    def interrupted_replace(_source: os.PathLike[str], _target: os.PathLike[str]) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(render.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        render.write_text_atomic(output, "new\n")

    assert output.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".artifact.txt.*.tmp")) == []


def test_run_writer_rolls_back_every_output_when_one_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcript_pipeline.render as render

    old = {
        render.TRANSCRIPT_FILENAME: "old transcript\n",
        render.QA_FILENAME: "old qa\n",
        render.REVIEW_FILENAME: "old review\n",
    }
    for name, content in old.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    original_replace = render.os.replace
    calls = 0

    def fail_second(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated set interruption")
        original_replace(source, target)

    monkeypatch.setattr(render.os, "replace", fail_second)
    with pytest.raises(OSError, match="simulated set interruption"):
        render.render_run_artifacts(tmp_path, bundle())

    assert {name: (tmp_path / name).read_text(encoding="utf-8") for name in old} == old
    assert list(tmp_path.glob(".*.render-*")) == []


def test_filename_constants_are_exact() -> None:
    from transcript_pipeline.render import QA_FILENAME, REVIEW_FILENAME, TRANSCRIPT_FILENAME

    assert (TRANSCRIPT_FILENAME, QA_FILENAME, REVIEW_FILENAME) == (
        "transcript.md",
        "qa.md",
        "review.md",
    )


def test_review_lists_every_transcript_edit() -> None:
    from transcript_pipeline.render import render_review

    data = bundle()
    data["diarized"]["segments"][3]["finalized_text"] = "For a 10 GB archive it takes about 40 minutes."

    review = render_review(data)

    assert "## Transcript edits" in review
    assert "| seg-000004 | For a ten gigabyte archive" in review
    assert "For a 10 GB archive it takes about 40 minutes. |" in review
