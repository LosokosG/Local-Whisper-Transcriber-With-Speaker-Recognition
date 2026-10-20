"""Pure, deterministic rendering of validated run outputs.

The renderer consumes one normalized, credential-free bundle (see
``validate.build_render_bundle``). It never retrieves data and never invents
transcript or Q&A content: a speaker is shown by name only when the validated
speaker evidence identified it, and every Q&A answer is followed by the
timestamps of the transcript segments it cites. Review items are derived again
from the sources so a stale upstream "ready" flag cannot bypass the review
checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

TRANSCRIPT_FILENAME = "transcript.md"
QA_FILENAME = "qa.md"
REVIEW_FILENAME = "review.md"
READY_STATUS = "**Status:** READY"
NOT_READY_STATUS = "**Status:** NOT READY"

_UNIDENTIFIED = re.compile(r"^Unidentified Speaker ([1-9][0-9]*)$")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


class RenderValidationError(ValueError):
    """The normalized render bundle is structurally unsafe or incomplete."""


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RenderValidationError(f"{path} must be an object")
    return value


def _records(value: object, path: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise RenderValidationError(f"{path} must be an array")
    return [_mapping(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderValidationError(f"{path} must be nonempty text")
    return value.strip()


def _number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderValidationError(f"{path} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise RenderValidationError(f"{path} must be a finite number >= {minimum:g}")
    return parsed


def _hms(seconds: float) -> str:
    # The containing whole second: deterministic and never later than the audio.
    whole = math.floor(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _markdown_inline(value: object, path: str) -> str:
    text = " ".join(_text(value, path).splitlines())
    return (
        text.replace("\\", "\\\\")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _single_line(value: object, path: str) -> str:
    return " ".join(_text(value, path).split())


def _table_cell(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    return " ".join(text.splitlines()).replace("\\", "\\\\").replace("|", "\\|")


def _title(bundle: Mapping[str, Any]) -> str:
    return _markdown_inline(bundle.get("title"), "title")


def _event_date(bundle: Mapping[str, Any], *, weekday: bool) -> str:
    event = bundle.get("event")
    if not isinstance(event, Mapping) or event.get("status") != "resolved":
        return "Not established"
    raw = _text(event.get("start_local"), "event.start_local")
    try:
        local = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise RenderValidationError("event.start_local must be ISO-8601") from exc
    if local.tzinfo is None:
        raise RenderValidationError("event.start_local must include a timezone")
    local_date = event.get("local_date")
    if local_date is not None and local_date != local.date().isoformat():
        raise RenderValidationError("event.local_date does not match event.start_local")
    prefix = f"{_WEEKDAYS[local.weekday()]}, " if weekday else ""
    return f"{prefix}{_MONTHS[local.month - 1]} {local.day}, {local.year}"


def _speaker_records(bundle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    evidence = _mapping(bundle.get("speaker_evidence"), "speaker_evidence")
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(_records(evidence.get("speakers"), "speaker_evidence.speakers")):
        base = f"speaker_evidence.speakers[{index}]"
        label = _text(record.get("speaker_label"), f"{base}.speaker_label")
        if label in result:
            raise RenderValidationError(f"duplicate speaker record {label}")
        output = _text(record.get("output_label"), f"{base}.output_label")
        published = record.get("status") == "identified" and record.get("confidence") == "high"
        if not published and _UNIDENTIFIED.fullmatch(output) is None:
            raise RenderValidationError(
                "a speaker that is not identified must render as Unidentified Speaker N"
            )
        result[label] = record
    return result


def _segments(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diarized = _mapping(bundle.get("diarized"), "diarized")
    return _records(diarized.get("segments"), "diarized.segments")


def _labelled_segments(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return segments in source order with their public speaker label."""

    speakers = _speaker_records(bundle)
    used = {
        int(match.group(1))
        for record in speakers.values()
        if (match := _UNIDENTIFIED.fullmatch(str(record.get("output_label")))) is not None
    }
    fallback: dict[str, str] = {}

    def fallback_label(key: str) -> str:
        if key not in fallback:
            number = 1
            while number in used:
                number += 1
            used.add(number)
            fallback[key] = f"Unidentified Speaker {number}"
        return fallback[key]

    seen: set[str] = set()
    previous_start = -1.0
    result: list[dict[str, Any]] = []
    for index, segment in enumerate(_segments(bundle)):
        base = f"diarized.segments[{index}]"
        segment_id = _text(segment.get("id"), f"{base}.id")
        if segment_id in seen:
            raise RenderValidationError(f"duplicate transcript segment ID {segment_id}")
        seen.add(segment_id)
        start = _number(segment.get("start"), f"{base}.start")
        end = _number(segment.get("end"), f"{base}.end")
        if end < start:
            raise RenderValidationError(f"{base}.end precedes start")
        if start < previous_start:
            raise RenderValidationError("diarized segments are not in source order")
        previous_start = start
        text = _text(
            segment.get("finalized_text") if "finalized_text" in segment else segment.get("text"),
            f"{base}.finalized_text/text",
        )
        raw_speaker = segment.get("speaker")
        speaker = raw_speaker if isinstance(raw_speaker, str) and raw_speaker else None
        record = speakers.get(speaker) if speaker is not None else None
        label = (
            _text(record.get("output_label"), "speaker output_label")
            if record is not None
            else fallback_label(speaker or "<none>")
        )
        result.append(
            {
                "id": segment_id,
                "start": start,
                "end": end,
                "text": text,
                "speaker_label": speaker,
                "output_label": label,
            }
        )
    return result


def _segment_times(bundle: Mapping[str, Any]) -> dict[str, float]:
    return {segment["id"]: segment["start"] for segment in _labelled_segments(bundle)}


def render_transcript(bundle: Mapping[str, Any]) -> str:
    """Render every segment exactly once; consecutive turns of one speaker merge."""

    data = _mapping(bundle, "bundle")
    segments = _labelled_segments(data)
    duration = _number(data.get("duration_seconds"), "duration_seconds")
    speakers = _speaker_records(data)
    participants: list[str] = []
    roles: dict[str, str] = {}
    for segment in segments:
        label = segment["output_label"]
        if label not in participants:
            participants.append(label)
            record = speakers.get(segment["speaker_label"] or "")
            role = record.get("role") if record is not None else None
            if record is not None and record.get("status") == "identified" and isinstance(role, str):
                roles[label] = role

    lines = [
        f"# Transcript: {_title(data)}",
        f"**Date:** {_event_date(data, weekday=True)}",
        f"**Duration:** {_hms(duration)}",
        "**Speakers:**",
        *(
            f"- {_markdown_inline(name, 'speaker')}"
            + (f" ({_markdown_inline(roles[name], 'role')})" if name in roles else "")
            for name in participants
        ),
        "",
        "---",
    ]
    block_label: str | None = None
    block_texts: list[str] = []
    block_start = 0.0

    def flush() -> None:
        if block_label is None:
            return
        lines.extend(
            [
                "",
                f"**[{_hms(block_start)}] {_markdown_inline(block_label, 'speaker')}:**",
                " ".join(block_texts),
            ]
        )

    for segment in segments:
        if segment["output_label"] != block_label:
            flush()
            block_label = segment["output_label"]
            block_texts = []
            block_start = segment["start"]
        block_texts.append(str(segment["text"]))
    flush()
    return "\n".join(lines) + "\n"


def _source_times(
    references: Sequence[Mapping[str, Any]], times: Mapping[str, float]
) -> list[str]:
    result: list[str] = []
    for reference in references:
        segment_id = reference.get("segment_id")
        if isinstance(segment_id, str) and segment_id in times:
            stamp = _hms(times[segment_id])
            if stamp not in result:
                result.append(stamp)
    return result


def _qa_questions(bundle: Mapping[str, Any]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    evidence = _mapping(bundle.get("qa_evidence"), "qa_evidence")
    topics = _records(evidence.get("topics"), "qa_evidence.topics")
    seen_topics: set[str] = set()
    seen_questions: set[str] = set()
    result: list[tuple[str, list[Mapping[str, Any]]]] = []
    for topic_index, topic in enumerate(topics):
        base = f"qa_evidence.topics[{topic_index}]"
        topic_id = _text(topic.get("id"), f"{base}.id")
        if topic_id in seen_topics:
            raise RenderValidationError(f"duplicate Q&A topic ID {topic_id}")
        seen_topics.add(topic_id)
        questions = _records(topic.get("questions"), f"{base}.questions")
        for question_index, question in enumerate(questions):
            question_id = _text(question.get("id"), f"{base}.questions[{question_index}].id")
            if question_id in seen_questions:
                raise RenderValidationError(f"duplicate Q&A question ID {question_id}")
            seen_questions.add(question_id)
        result.append((_text(topic.get("title"), f"{base}.title"), questions))
    return result


def render_qa(bundle: Mapping[str, Any]) -> str:
    """Render the Q&A summary with transcript timestamps for every answer."""

    data = _mapping(bundle, "bundle")
    speakers = _speaker_records(data)
    times = _segment_times(data)
    topics = _qa_questions(data)
    counts = {"answered": 0, "partially_answered": 0, "unanswered": 0}
    for _title_text, questions in topics:
        for question in questions:
            status = question.get("status")
            if status not in counts:
                raise RenderValidationError("question status is unsupported")
            counts[str(status)] += 1

    def speaker_name(label: object) -> str | None:
        if not isinstance(label, str):
            return None
        record = speakers.get(label)
        return str(record.get("output_label")) if record is not None else None

    lines = [
        f"# Q&A: {_title(data)}",
        f"**Date:** {_event_date(data, weekday=False)}",
        (
            f"**Questions:** {sum(counts.values())} "
            f"(answered {counts['answered']}, partially answered "
            f"{counts['partially_answered']}, unanswered {counts['unanswered']})"
        ),
    ]
    for topic_title, questions in topics:
        if not questions:
            continue
        lines.extend(["", f"## {_markdown_inline(topic_title, 'topic title')}"])
        for index, question in enumerate(questions):
            question_text = _single_line(question.get("question"), "question")
            asked = _source_times(question.get("asked_sources", []), times)
            asker = speaker_name(question.get("asked_by"))
            context = ", ".join(
                part
                for part in (
                    f"asked by {_markdown_inline(asker, 'asker')}" if asker else "",
                    f"at {asked[0]}" if asked else "",
                )
                if part
            )
            status = question.get("status")
            if status == "unanswered":
                answer_line = "**A:** *Not answered in the recording.*"
                sources_line = None
            else:
                combined = _single_line(question.get("combined_answer"), "combined_answer")
                label = "**A (partial):**" if status == "partially_answered" else "**A:**"
                answer_line = f"{label} {combined}"
                answer_times: list[str] = []
                for answer in _records(question.get("answers"), "answers"):
                    for stamp in _source_times(answer.get("sources", []), times):
                        if stamp not in answer_times:
                            answer_times.append(stamp)
                sources_line = (
                    f"*Answer at: {', '.join(sorted(answer_times))}*" if answer_times else None
                )
            if index:
                lines.append("")
            lines.append(f"**Q: {question_text}**" + (f" *({context})*" if context else ""))
            lines.append(answer_line)
            if sources_line:
                lines.append(sources_line)
    return "\n".join(lines) + "\n"


def _item(code: str, message: str, *, blocking: bool, details: object = None) -> dict[str, Any]:
    record: dict[str, Any] = {"code": code, "message": message, "blocking": blocking}
    if details not in (None, {}, []):
        record["details"] = details
    return record


def collect_review_items(bundle: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Derive blocking and informational review items from the sources."""

    data = _mapping(bundle, "bundle")
    items: list[dict[str, Any]] = []
    validation = data.get("artifact_validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        issues = validation.get("issues", []) if isinstance(validation, Mapping) else []
        items.append(
            _item(
                "artifact_validation_failed",
                "run validation failed; outputs must not be treated as final",
                blocking=True,
                details=[
                    {key: issue.get(key) for key in ("code", "path", "message")}
                    for issue in issues
                    if isinstance(issue, Mapping)
                ],
            )
        )
    speakers = _speaker_records(data)
    segments = _labelled_segments(data)
    transcript_speakers = sorted(
        {segment["speaker_label"] for segment in segments if segment["speaker_label"]}
    )
    for label in transcript_speakers:
        record = speakers.get(label)
        if record is None:
            items.append(
                _item("speaker_record_missing", f"{label} has no speaker record", blocking=True)
            )
        elif record.get("status") == "manual_review":
            items.append(
                _item(
                    "speaker_manual_review",
                    f"{label} needs a human decision before a name is published",
                    blocking=True,
                    details={
                        "candidate": record.get("display_name"),
                        "contradictions": record.get("contradictions", []),
                    },
                )
            )
        elif record.get("status") == "unidentified":
            items.append(
                _item(
                    "speaker_unidentified",
                    f"{label} is published as {record.get('output_label')}",
                    blocking=False,
                )
            )
    unlabelled = sum(1 for segment in segments if not segment["speaker_label"])
    if unlabelled:
        items.append(
            _item(
                "segments_without_speaker",
                f"{unlabelled} segment(s) have no diarized speaker",
                blocking=False,
            )
        )
    for _topic, questions in _qa_questions(data):
        for question in questions:
            if question.get("status") == "unanswered":
                items.append(
                    _item(
                        "question_unanswered",
                        f"question {question.get('id')} is marked unanswered",
                        blocking=False,
                    )
                )
    event = data.get("event")
    if not isinstance(event, Mapping) or event.get("status") != "resolved":
        items.append(
            _item("date_not_established", "the recording date is not established", blocking=False)
        )
    return tuple(items)


def _reference_text(reference: Mapping[str, Any], times: Mapping[str, float]) -> str:
    parts: list[str] = []
    segment_id = reference.get("segment_id")
    if isinstance(segment_id, str):
        stamp = f" [{_hms(times[segment_id])}]" if segment_id in times else " [unknown segment]"
        parts.append(f"{segment_id}{stamp}")
    for key in ("visual_frame_id", "visual_observation_id"):
        value = reference.get(key)
        if isinstance(value, str):
            parts.append(f"{key.removeprefix('visual_')}:{value}")
    quote = reference.get("quote")
    if isinstance(quote, str):
        parts.append(f'"{" ".join(quote.split())}"')
    return " ".join(parts)


def _references(value: object, times: Mapping[str, float]) -> str:
    if not isinstance(value, list):
        return "—"
    rendered = [
        _reference_text(reference, times)
        for reference in value
        if isinstance(reference, Mapping)
    ]
    return "; ".join(rendered) or "—"


def render_review(bundle: Mapping[str, Any]) -> str:
    """Render the private review: status, blocking items, and full evidence."""

    data = _mapping(bundle, "bundle")
    items = collect_review_items(data)
    blocking = [item for item in items if item["blocking"]]
    notes = [item for item in items if not item["blocking"]]
    speakers = _speaker_records(data)
    times = _segment_times(data)
    status_line = (
        f"{NOT_READY_STATUS} — resolve the blocking items below before using these outputs."
        if blocking
        else f"{READY_STATUS} — no blocking review items."
    )
    lines = [
        f"# Review: {_title(data)}",
        status_line,
        f"**Date:** {_event_date(data, weekday=False)}",
        "",
        "This file is private working material: it contains raw evidence quotes.",
        "",
        "## Blocking items",
    ]
    if blocking:
        for item in blocking:
            suffix = f" Details: {_table_cell(item.get('details'))}" if "details" in item else ""
            lines.append(f"- **{item['code']}:** {item['message']}.{suffix}")
    else:
        lines.append("None.")
    lines.extend(["", "## Notes"])
    lines.extend(
        [f"- **{item['code']}:** {item['message']}." for item in notes] or ["None."]
    )

    lines.extend(
        [
            "",
            "## Speakers",
            "| Label | Status | Published as | Confidence | Candidate name | Evidence | Contradictions |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for label in sorted(speakers):
        record = speakers[label]
        bindings = record.get("binding_evidence", [])
        evidence = "; ".join(
            f"{binding.get('type')}: {_references(binding.get('sources'), times)}"
            for binding in bindings
            if isinstance(binding, Mapping)
        )
        lines.append(
            "| "
            + " | ".join(
                _table_cell(value)
                for value in (
                    label,
                    record.get("status"),
                    record.get("output_label"),
                    record.get("confidence"),
                    record.get("display_name"),
                    evidence,
                    "; ".join(str(item) for item in record.get("contradictions", [])),
                )
            )
            + " |"
        )

    decisions = data.get("operator_decisions")
    if isinstance(decisions, list) and decisions:
        lines.extend(["", "## Operator decisions"])
        for decision in decisions:
            if isinstance(decision, Mapping):
                lines.append(
                    f"- {_table_cell(decision.get('recorded_at'))}: "
                    f"{_table_cell(decision.get('decision'))} {_table_cell(decision.get('details'))}"
                )

    lines.extend(["", "## Question evidence"])
    for topic_title, questions in _qa_questions(data):
        for question in questions:
            lines.extend(
                [
                    "",
                    f"### {_table_cell(question.get('id'))} ({_table_cell(question.get('status'))})",
                    f"- **Topic:** {_markdown_inline(topic_title, 'topic')}",
                    f"- **Question:** {_single_line(question.get('question'), 'question')}",
                    f"- **Asked:** {_references(question.get('asked_sources'), times)}",
                ]
            )
            for index, answer in enumerate(_records(question.get("answers"), "answers"), start=1):
                lines.append(f"- **Answer {index}:** {_references(answer.get('sources'), times)}")
            for claim in _records(question.get("claims"), "claims"):
                lines.append(
                    f"- **Claim ({_table_cell(claim.get('kind'))}):** "
                    f"{_single_line(claim.get('text'), 'claim')} — "
                    f"{_references(claim.get('sources'), times)}"
                )
    return "\n".join(lines) + "\n"


def review_is_ready(text: str) -> bool:
    """Return whether a rendered review declares itself ready."""

    lines = text.splitlines()
    return len(lines) > 1 and lines[1].startswith(f"{READY_STATUS} ")


def write_text_atomic(path: str | os.PathLike[str], text: str) -> None:
    """Write UTF-8 text with fsync and atomic replacement of one exact path."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
        _fsync_directory(output.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_text(directory: Path, name: str, content: str) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=directory,
            prefix=f".{name}.render-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _backup_link(path: Path) -> Path:
    descriptor, raw_backup = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.render-backup-",
        suffix=".tmp",
    )
    os.close(descriptor)
    backup = Path(raw_backup)
    try:
        backup.unlink()
        os.link(path, backup)
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _replace_artifact_set(directory: Path, contents: Mapping[str, str]) -> None:
    """Stage a render set and roll every target back if replacement is interrupted."""

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    changed: set[Path] = set()
    try:
        for name, content in contents.items():
            staged[directory / name] = _stage_text(directory, name, content)
        _fsync_directory(directory)
        for target in staged:
            if target.exists():
                backups[target] = _backup_link(target)
        _fsync_directory(directory)
        for target, temporary in staged.items():
            os.replace(temporary, target)
            changed.add(target)
        _fsync_directory(directory)
    except BaseException:
        # Hard-link backups keep the previous inode alive, so os.rename can
        # restore it even if the replacement primitive itself failed.
        for target in reversed(list(staged)):
            if target not in changed:
                continue
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.rename(backup, target)
            else:
                target.unlink(missing_ok=True)
        try:
            _fsync_directory(directory)
        except OSError:
            pass
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def render_run_artifacts(
    run_dir: str | os.PathLike[str], bundle: Mapping[str, Any]
) -> dict[str, Path]:
    """Render transcript, Q&A, and review, then replace them as one set."""

    # Every pure render finishes before the filesystem is touched at all.
    contents = {
        TRANSCRIPT_FILENAME: render_transcript(bundle),
        QA_FILENAME: render_qa(bundle),
        REVIEW_FILENAME: render_review(bundle),
    }
    directory = Path(run_dir)
    if directory.exists() and not directory.is_dir():
        raise ValueError("run_dir must be a directory")
    directory.mkdir(parents=True, exist_ok=True)
    _replace_artifact_set(directory, contents)
    return {name: directory / name for name in contents}


__all__ = [
    "NOT_READY_STATUS",
    "QA_FILENAME",
    "READY_STATUS",
    "REVIEW_FILENAME",
    "RenderValidationError",
    "TRANSCRIPT_FILENAME",
    "collect_review_items",
    "render_qa",
    "render_review",
    "render_run_artifacts",
    "render_transcript",
    "review_is_ready",
    "write_text_atomic",
]
