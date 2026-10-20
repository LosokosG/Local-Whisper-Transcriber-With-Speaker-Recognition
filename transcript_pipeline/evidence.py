"""Strict, cross-referenced contracts for agent- or human-proposed evidence.

Two evidence documents are validated here:

* speaker evidence, which maps diarized ``SPEAKER_xx`` labels to public names
  or to ``Unidentified Speaker N``; and
* Q&A evidence, which lists questions, answers, and checkable claims.

Both are first checked against packaged JSON Schemas. They are then checked
against the run's own sources: every cited segment, frame, and observation
must exist, every quote must occur in the cited segment text, speaker labels
must match the segments they cite, and a public name is accepted only when a
definitive binding can be re-derived from the sources. The validator never
trusts a self-reported confidence flag on its own.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Literal, TypeAlias

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

EVIDENCE_SCHEMA_VERSION = 1
SPEAKER_ALIGNMENT_TOLERANCE_SECONDS = 2.0
INTRODUCTION_WINDOW_SECONDS = 60.0
ANSWER_ORDER_TOLERANCE_SECONDS = 1.0
EvidenceKind: TypeAlias = Literal["speaker", "qa"]
ValidationIssue: TypeAlias = dict[str, str]
ValidationResult: TypeAlias = dict[str, Any] | list[ValidationIssue]

_SCHEMA_FILES: dict[str, str] = {
    "speaker": "speaker-evidence.schema.json",
    "qa": "qa-evidence.schema.json",
}
_EXTRACTED_FRAME_BASENAME = re.compile(r"^frame_[0-9]{9,}\.png$")
_UNIDENTIFIED_LABEL = re.compile(r"^Unidentified Speaker ([1-9][0-9]*)$")
# Bindings whose cited transcript segments must all belong to the named speaker.
_OWN_SPEECH_BINDINGS = frozenset(
    {
        "self_identification",
        "visible_speaking_indicator",
        "voice_cluster_propagation",
        "timing_correlation",
        "operator_confirmation",
    }
)


class EvidenceValidationError(ValueError):
    """One or more sanitized evidence problems with exact JSON paths."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        copied = tuple(dict(issue) for issue in issues)
        if not copied:
            raise ValueError("EvidenceValidationError requires at least one issue")
        self.issues = copied
        summary = "; ".join(
            f"{issue['path']}: {issue['message']}" for issue in copied
        )
        super().__init__(f"evidence validation failed: {summary}")


@cache
def _cached_schema(kind: str) -> dict[str, Any]:
    filename = _SCHEMA_FILES[kind]
    try:
        text = (
            resources.files("transcript_pipeline")
            .joinpath("schemas", filename)
            .read_text(encoding="utf-8")
        )
        schema = json.loads(text)
        if not isinstance(schema, dict):
            raise TypeError("schema root is not an object")
        Draft202012Validator.check_schema(schema)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"packaged {kind} evidence schema cannot be loaded") from exc
    return schema


def load_evidence_schema(kind: str) -> dict[str, Any]:
    """Load a packaged evidence schema without exposing the cached object."""

    if kind not in _SCHEMA_FILES:
        raise ValueError("unknown evidence schema kind")
    return copy.deepcopy(_cached_schema(kind))


def normalize_text(value: str) -> str:
    """Normalize text for quote matching: case, punctuation, and whitespace.

    Letters and digits are kept; every other character becomes a word break.
    This tolerates punctuation and capitalization differences introduced by
    ASR or by a light transcript edit, but not changed words.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        character
        if unicodedata.category(character)[0] in {"L", "N"}
        else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def _contains_phrase(haystack: str, needle: str) -> bool:
    normalized_needle = normalize_text(needle)
    if not normalized_needle:
        return False
    return f" {normalized_needle} " in f" {normalize_text(haystack)} "


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(value: str) -> set[str]:
    """Digit groups in text, with thousands separators removed."""

    return {match.replace(",", "") for match in _NUMBER.findall(unicodedata.normalize("NFKC", value))}


def _issue(path: str, code: str, message: str) -> ValidationIssue:
    return {"path": path, "code": code, "message": message}


def _child_path(base: str, part: object) -> str:
    if isinstance(part, int):
        return f"{base}[{part}]"
    name = str(part)
    if name.replace("_", "a").isalnum() and not name[:1].isdigit():
        return f"{base}.{name}"
    return f"{base}[{json.dumps(name)}]"


def _json_path(parts: Sequence[object], *, base: str = "$") -> str:
    result = base
    for part in parts:
        result = _child_path(result, part)
    return result


def _sanitized_schema_message(error: ValidationError) -> str:
    messages = {
        "additionalProperties": "contains unsupported field(s)",
        "anyOf": "source reference requires a supported source ID",
        "const": "uses an unsupported schema version",
        "dependentRequired": "a quote requires a segment_id",
        "enum": "contains an unsupported value",
        "maxLength": "exceeds the allowed length",
        "maximum": "exceeds the allowed maximum",
        "minItems": "requires at least one item",
        "minLength": "must not be empty",
        "minimum": "is below the allowed minimum",
        "oneOf": "has an invalid type or format",
        "pattern": "does not match the required ID or label format",
        "required": "is missing a required field",
        "type": "has an invalid type",
        "uniqueItems": "must contain unique items",
    }
    return messages.get(error.validator, "does not satisfy the evidence contract")


def _schema_issues(kind: EvidenceKind, document: object) -> list[ValidationIssue]:
    if not isinstance(document, Mapping):
        return [_issue("$", "schema", "evidence document must be an object")]
    validator = Draft202012Validator(_cached_schema(kind))
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: (list(error.absolute_path), error.validator or ""),
    )
    return [
        _issue(
            _json_path(list(error.absolute_path)),
            "schema",
            _sanitized_schema_message(error),
        )
        for error in errors
    ]


def _normalized_copy(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(document, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError(
            [_issue("$", "json", "evidence document must be strict JSON")]
        ) from exc


def _json_issues(document: object) -> list[ValidationIssue]:
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError):
        return [_issue("$", "json", "evidence document must be strict JSON")]
    return []


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class _SourceIndex:
    segments: dict[str, Mapping[str, Any]]
    frames: dict[str, Mapping[str, Any]]
    observations: dict[str, Mapping[str, Any]]
    confirmed_names: Mapping[str, str]

    @property
    def speakers(self) -> frozenset[str]:
        return frozenset(
            speaker
            for segment in self.segments.values()
            if isinstance((speaker := segment.get("speaker")), str) and speaker
        )

    def segment_speaker(self, segment_id: str) -> str | None:
        speaker = self.segments.get(segment_id, {}).get("speaker")
        return speaker if isinstance(speaker, str) and speaker else None

    def segment_bounds(self, segment_id: str) -> tuple[float, float] | None:
        segment = self.segments.get(segment_id)
        if segment is None:
            return None
        start = _finite(segment.get("start"))
        end = _finite(segment.get("end"))
        if start is None or end is None or end < start:
            return None
        return start, end

    def segment_texts(self, segment_id: str) -> tuple[str, ...]:
        # Quotes are checked against the canonical ASR text only. Reviewed
        # edits change what is displayed, never what counts as evidence.
        text = self.segments.get(segment_id, {}).get("text")
        return (text,) if isinstance(text, str) and text else ()

    def next_turn_after(self, segment_id: str) -> str | None:
        """Return the first later segment spoken by someone else."""

        speaker = self.segment_speaker(segment_id)
        identifiers = list(self.segments)
        try:
            position = identifiers.index(segment_id)
        except ValueError:
            return None
        for candidate in identifiers[position + 1 :]:
            other = self.segment_speaker(candidate)
            if other is not None and other != speaker:
                return candidate
        return None

    def reference_time(self, reference: Mapping[str, Any]) -> float | None:
        segment_id = reference.get("segment_id")
        if isinstance(segment_id, str):
            bounds = self.segment_bounds(segment_id)
            return bounds[0] if bounds is not None else None
        for key, records in (
            ("visual_frame_id", self.frames),
            ("visual_observation_id", self.observations),
        ):
            identifier = reference.get(key)
            if isinstance(identifier, str) and identifier in records:
                return _finite(records[identifier].get("timestamp_seconds"))
        return None


def _record_alias_index(
    records: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        for name in names:
            value = record.get(name)
            if isinstance(value, str) and value:
                result.setdefault(value, record)
    return result


def _extracted_frame_alias(record: Mapping[str, Any]) -> str | None:
    raw_path = record.get("raw_path")
    timestamp = _finite(record.get("timestamp_seconds"))
    if not isinstance(raw_path, str) or not raw_path or timestamp is None or timestamp < 0:
        return None
    basename = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
    if _EXTRACTED_FRAME_BASENAME.fullmatch(basename) is None:
        return None
    if basename != f"frame_{round(timestamp * 1000):09d}.png":
        return None
    return basename.removesuffix(".png")


def _frame_source_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    usable = [record for record in records if record.get("status") == "ok"]
    result = _record_alias_index(usable, ("id", "frame_id", "source_ref"))
    aliases: dict[str, list[Mapping[str, Any]]] = {}
    for record in usable:
        alias = _extracted_frame_alias(record)
        if alias is not None:
            aliases.setdefault(alias, []).append(record)
    for alias, matches in aliases.items():
        if len(matches) == 1:
            result.setdefault(alias, matches[0])
    return result


def _source_index(
    diarized_document: object,
    visual_evidence: object | None,
    confirmed_names: Mapping[str, str] | None = None,
) -> tuple[_SourceIndex | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if not isinstance(diarized_document, Mapping):
        issues.append(
            _issue("$sources.diarized", "source_document", "must be an object")
        )
        diarized_document = {}
    if visual_evidence is None:
        visual_evidence = {}
    elif not isinstance(visual_evidence, Mapping):
        issues.append(
            _issue("$sources.visual", "source_document", "must be an object")
        )
        visual_evidence = {}

    segments: dict[str, Mapping[str, Any]] = {}
    for index, segment in enumerate(_mapping_sequence(diarized_document.get("segments"))):
        identifier = segment.get("id")
        path = f"$sources.diarized.segments[{index}].id"
        if not isinstance(identifier, str) or not identifier:
            issues.append(_issue(path, "source_document", "requires a stable ID"))
        elif identifier in segments:
            issues.append(_issue(path, "source_document", "ID must be unique"))
        else:
            segments[identifier] = segment

    frames = _frame_source_index(_mapping_sequence(visual_evidence.get("sources")))
    observations = _record_alias_index(
        _mapping_sequence(visual_evidence.get("presence_observations")),
        ("id", "observation_id", "source_ref"),
    )
    for index, observation in enumerate(
        _mapping_sequence(visual_evidence.get("presence_observations"))
    ):
        frame_id = observation.get("frame_id")
        if frame_id is None:
            continue
        frame = frames.get(frame_id) if isinstance(frame_id, str) else None
        observed_at = _finite(observation.get("timestamp_seconds"))
        frame_at = _finite(frame.get("timestamp_seconds")) if frame else None
        if frame is None or observed_at is None or frame_at is None:
            issues.append(
                _issue(
                    f"$sources.visual.presence_observations[{index}].frame_id",
                    "source_document",
                    "observation must reference an extracted frame",
                )
            )
        elif abs(observed_at - frame_at) > 0.001:
            issues.append(
                _issue(
                    f"$sources.visual.presence_observations[{index}].timestamp_seconds",
                    "source_document",
                    "observation timestamp contradicts its frame",
                )
            )

    index = _SourceIndex(
        segments=segments,
        frames=frames,
        observations=observations,
        confirmed_names=dict(confirmed_names or {}),
    )
    return (index if not issues else None), issues


def _reference_issues(
    references: Sequence[Mapping[str, Any]],
    base_path: str,
    index: _SourceIndex,
) -> list[ValidationIssue]:
    """Unknown source IDs and quotes that do not occur in their segment."""

    issues: list[ValidationIssue] = []
    known: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("segment_id", index.segments),
        ("visual_frame_id", index.frames),
        ("visual_observation_id", index.observations),
    )
    for reference_index, reference in enumerate(references):
        path = f"{base_path}[{reference_index}]"
        for field, records in known:
            identifier = reference.get(field)
            if isinstance(identifier, str) and identifier not in records:
                issues.append(
                    _issue(
                        f"{path}.{field}",
                        "unknown_source",
                        f"references unknown {field} {identifier}",
                    )
                )
        segment_id = reference.get("segment_id")
        quote = reference.get("quote")
        if (
            isinstance(quote, str)
            and isinstance(segment_id, str)
            and segment_id in index.segments
            and not any(
                _contains_phrase(text, quote)
                for text in index.segment_texts(segment_id)
            )
        ):
            issues.append(
                _issue(
                    f"{path}.quote",
                    "quote_not_found",
                    f"quote does not occur in segment {segment_id}",
                )
            )
    return issues


def _segment_ids(references: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        identifier
        for reference in references
        if isinstance((identifier := reference.get("segment_id")), str)
    ]


def _verified_quotes(
    references: Sequence[Mapping[str, Any]], index: _SourceIndex
) -> list[tuple[str, str]]:
    """Return ``(segment_id, quote)`` pairs whose quote occurs in the segment."""

    result: list[tuple[str, str]] = []
    for reference in references:
        segment_id = reference.get("segment_id")
        quote = reference.get("quote")
        if (
            isinstance(segment_id, str)
            and isinstance(quote, str)
            and any(
                _contains_phrase(text, quote)
                for text in index.segment_texts(segment_id)
            )
        ):
            result.append((segment_id, quote))
    return result


def _self_identification_holds(
    references: Sequence[Mapping[str, Any]],
    speaker: str,
    name: str,
    index: _SourceIndex,
) -> bool:
    return any(
        index.segment_speaker(segment_id) == speaker and _contains_phrase(quote, name)
        for segment_id, quote in _verified_quotes(references, index)
    )


def _introduction_holds(
    references: Sequence[Mapping[str, Any]],
    speaker: str,
    name: str,
    index: _SourceIndex,
) -> bool:
    cited = set(_segment_ids(references))
    for segment_id, quote in _verified_quotes(references, index):
        introducer = index.segment_speaker(segment_id)
        if introducer is None or introducer == speaker or not _contains_phrase(quote, name):
            continue
        # The named person must take the very next turn, and must be cited.
        answer = index.next_turn_after(segment_id)
        if answer is None or answer not in cited or index.segment_speaker(answer) != speaker:
            continue
        intro = index.segment_bounds(segment_id)
        turn = index.segment_bounds(answer)
        if intro is not None and turn is not None and turn[0] <= intro[1] + INTRODUCTION_WINDOW_SECONDS:
            return True
    return False


def _speaking_indicator_holds(
    references: Sequence[Mapping[str, Any]],
    speaker: str,
    name: str,
    index: _SourceIndex,
) -> bool:
    turns = [
        bounds
        for segment_id in _segment_ids(references)
        if index.segment_speaker(segment_id) == speaker
        and (bounds := index.segment_bounds(segment_id)) is not None
    ]
    wanted = normalize_text(name)
    for reference in references:
        identifier = reference.get("visual_observation_id")
        if not isinstance(identifier, str):
            continue
        observation = index.observations.get(identifier)
        if observation is None:
            continue
        visible = observation.get("exact_visible_name")
        timestamp = _finite(observation.get("timestamp_seconds"))
        if (
            observation.get("evidence_type") != "speaking_indicator"
            or observation.get("truncated") is True
            or not isinstance(visible, str)
            or normalize_text(visible) != wanted
            or timestamp is None
        ):
            continue
        if any(
            start - SPEAKER_ALIGNMENT_TOLERANCE_SECONDS
            <= timestamp
            <= end + SPEAKER_ALIGNMENT_TOLERANCE_SECONDS
            for start, end in turns
        ):
            return True
    return False


def _operator_confirmation_holds(
    references: Sequence[Mapping[str, Any]],
    speaker: str,
    name: str,
    index: _SourceIndex,
) -> bool:
    confirmed = index.confirmed_names.get(speaker)
    return (
        isinstance(confirmed, str)
        and normalize_text(confirmed) == normalize_text(name)
        and any(index.segment_speaker(segment_id) == speaker for segment_id in _segment_ids(references))
    )


def _definitive_binding(
    bindings: Sequence[Mapping[str, Any]],
    speaker: str,
    name: str,
    index: _SourceIndex,
) -> bool:
    checks = {
        "self_identification": _self_identification_holds,
        "introduced_by_other": _introduction_holds,
        "visible_speaking_indicator": _speaking_indicator_holds,
        "operator_confirmation": _operator_confirmation_holds,
    }
    for binding in bindings:
        check = checks.get(str(binding.get("type")))
        if check is not None and check(
            _mapping_sequence(binding.get("sources")), speaker, name, index
        ):
            return True
    return False


def _speaker_semantic_issues(
    document: Mapping[str, Any], index: _SourceIndex
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_labels: set[str] = set()
    seen_outputs: dict[str, str] = {}
    identified_names: dict[str, str] = {}
    transcript_speakers = index.speakers

    for record_index, record in enumerate(_mapping_sequence(document.get("speakers"))):
        base = f"$.speakers[{record_index}]"
        speaker = record.get("speaker_label")
        status = record.get("status")
        confidence = record.get("confidence")
        output_label = record.get("output_label")
        display_name = record.get("display_name")
        contradictions = record.get("contradictions")
        bindings = _mapping_sequence(record.get("binding_evidence"))

        if isinstance(speaker, str):
            if speaker in seen_labels:
                issues.append(
                    _issue(f"{base}.speaker_label", "duplicate_speaker", "speaker_label values must be unique")
                )
            seen_labels.add(speaker)
            if speaker not in transcript_speakers:
                issues.append(
                    _issue(
                        f"{base}.speaker_label",
                        "unknown_speaker",
                        f"references unknown speaker label {speaker}",
                    )
                )

        if isinstance(output_label, str):
            key = output_label.casefold()
            if key in seen_outputs:
                issues.append(
                    _issue(
                        f"{base}.output_label",
                        "duplicate_output_label",
                        f"output label is already used by {seen_outputs[key]}",
                    )
                )
            else:
                seen_outputs[key] = str(speaker)

        for binding_index, binding in enumerate(bindings):
            references = _mapping_sequence(binding.get("sources"))
            source_path = f"{base}.binding_evidence[{binding_index}].sources"
            issues.extend(_reference_issues(references, source_path, index))
            if binding.get("type") not in _OWN_SPEECH_BINDINGS or not isinstance(speaker, str):
                continue
            for source_index, reference in enumerate(references):
                segment_id = reference.get("segment_id")
                if (
                    isinstance(segment_id, str)
                    and segment_id in index.segments
                    and index.segment_speaker(segment_id) != speaker
                ):
                    issues.append(
                        _issue(
                            f"{source_path}[{source_index}].segment_id",
                            "speaker_mismatch",
                            f"segment {segment_id} belongs to a different speaker label",
                        )
                    )

        if isinstance(contradictions, list) and contradictions and status != "manual_review":
            issues.append(
                _issue(
                    f"{base}.status",
                    "contradiction",
                    "recorded contradictions require manual_review",
                )
            )
        if status == "identified" and confidence != "high":
            issues.append(
                _issue(
                    f"{base}.confidence",
                    "confidence",
                    "only a high-confidence speaker may be published by name",
                )
            )

        if status == "identified":
            if not isinstance(display_name, str):
                issues.append(
                    _issue(f"{base}.display_name", "display_name", "identified speaker requires a display_name")
                )
            else:
                if output_label != display_name:
                    issues.append(
                        _issue(
                            f"{base}.output_label",
                            "output_label",
                            "identified speaker must publish exactly its display_name",
                        )
                    )
                name_key = normalize_text(display_name)
                if name_key in identified_names:
                    issues.append(
                        _issue(
                            f"{base}.display_name",
                            "conflicting_identity",
                            f"display name is already bound to {identified_names[name_key]}",
                        )
                    )
                else:
                    identified_names[name_key] = str(speaker)
                if not isinstance(speaker, str) or not _definitive_binding(
                    bindings, speaker, display_name, index
                ):
                    issues.append(
                        _issue(
                            f"{base}.binding_evidence",
                            "binding_not_definitive",
                            "a published name needs a self-identification or introduction "
                            "quote containing the name, an aligned speaking-indicator "
                            "observation of the exact name, or a recorded operator confirmation",
                        )
                    )
        else:
            if not isinstance(output_label, str) or _UNIDENTIFIED_LABEL.fullmatch(output_label) is None:
                issues.append(
                    _issue(
                        f"{base}.output_label",
                        "output_label",
                        "a speaker that is not identified must use 'Unidentified Speaker N'",
                    )
                )
            if status == "unidentified" and display_name is not None:
                issues.append(
                    _issue(
                        f"{base}.display_name",
                        "unidentified",
                        "an unidentified speaker cannot carry a display_name",
                    )
                )

    for speaker in sorted(transcript_speakers - seen_labels):
        issues.append(
            _issue(
                "$.speakers",
                "missing_speaker",
                f"missing record for speaker label {speaker}",
            )
        )
    return issues


def _qa_semantic_issues(
    document: Mapping[str, Any], index: _SourceIndex
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    topic_ids: set[str] = set()
    question_ids: set[str] = set()
    transcript_speakers = index.speakers

    def speaker_issue(path: str, label: object, references: Sequence[Mapping[str, Any]]) -> None:
        if not isinstance(label, str):
            return
        if label not in transcript_speakers:
            issues.append(_issue(path, "unknown_speaker", f"references unknown speaker label {label}"))
            return
        cited = [segment_id for segment_id in _segment_ids(references) if segment_id in index.segments]
        if cited and not any(index.segment_speaker(segment_id) == label for segment_id in cited):
            issues.append(
                _issue(path, "speaker_mismatch", f"no cited segment belongs to {label}")
            )

    for topic_index, topic in enumerate(_mapping_sequence(document.get("topics"))):
        topic_base = f"$.topics[{topic_index}]"
        topic_id = topic.get("id")
        if isinstance(topic_id, str):
            if topic_id in topic_ids:
                issues.append(_issue(f"{topic_base}.id", "duplicate_topic", "topic ids must be unique"))
            topic_ids.add(topic_id)
        for question_index, question in enumerate(_mapping_sequence(topic.get("questions"))):
            base = f"{topic_base}.questions[{question_index}]"
            question_id = question.get("id")
            if isinstance(question_id, str):
                if question_id in question_ids:
                    issues.append(_issue(f"{base}.id", "duplicate_question", "question ids must be unique"))
                question_ids.add(question_id)

            asked_sources = _mapping_sequence(question.get("asked_sources"))
            issues.extend(_reference_issues(asked_sources, f"{base}.asked_sources", index))
            speaker_issue(f"{base}.asked_by", question.get("asked_by"), asked_sources)

            answers = _mapping_sequence(question.get("answers"))
            asked_times = [
                time
                for reference in asked_sources
                if (time := index.reference_time(reference)) is not None
            ]
            quoted_numbers: set[str] = set()
            for answer_index, answer in enumerate(answers):
                answer_base = f"{base}.answers[{answer_index}]"
                answer_sources = _mapping_sequence(answer.get("sources"))
                issues.extend(_reference_issues(answer_sources, f"{answer_base}.sources", index))
                segment_ids = _segment_ids(answer_sources)
                if not any(isinstance(reference.get("quote"), str) for reference in answer_sources if reference.get("segment_id")):
                    issues.append(
                        _issue(
                            f"{answer_base}.sources",
                            "quote_required",
                            "an answer must cite a transcript segment with a verbatim quote",
                        )
                    )
                speaker_issue(f"{answer_base}.answered_by", answer.get("answered_by"), answer_sources)
                for _segment_id, quote in _verified_quotes(answer_sources, index):
                    quoted_numbers.update(_numbers(quote))
                answer_ends = [
                    bounds[1]
                    for segment_id in segment_ids
                    if (bounds := index.segment_bounds(segment_id)) is not None
                ]
                if asked_times and answer_ends and max(answer_ends) < (
                    min(asked_times) - ANSWER_ORDER_TOLERANCE_SECONDS
                ):
                    issues.append(
                        _issue(
                            f"{answer_base}.sources",
                            "answer_precedes_question",
                            "the cited answer ends before the question was asked",
                        )
                    )

            for claim_index, claim in enumerate(_mapping_sequence(question.get("claims"))):
                claim_base = f"{base}.claims[{claim_index}]"
                claim_sources = _mapping_sequence(claim.get("sources"))
                issues.extend(_reference_issues(claim_sources, f"{claim_base}.sources", index))
                if not any(
                    isinstance(reference.get("segment_id"), str)
                    and isinstance(reference.get("quote"), str)
                    for reference in claim_sources
                ):
                    issues.append(
                        _issue(
                            f"{claim_base}.sources",
                            "quote_required",
                            "a claim must cite a transcript segment with a verbatim quote",
                        )
                    )
                claim_numbers: set[str] = set()
                for _segment_id, quote in _verified_quotes(claim_sources, index):
                    claim_numbers.update(_numbers(quote))
                quoted_numbers.update(claim_numbers)
                text = claim.get("text")
                missing = sorted(_numbers(text) - claim_numbers) if isinstance(text, str) else []
                if missing:
                    issues.append(
                        _issue(
                            f"{claim_base}.text",
                            "number_not_quoted",
                            f"claim number(s) {', '.join(missing)} do not appear in its quotes",
                        )
                    )

            status = question.get("status")
            combined = question.get("combined_answer")
            claims = question.get("claims")
            if status == "unanswered":
                if answers or combined is not None:
                    issues.append(
                        _issue(
                            f"{base}.answers",
                            "unanswered",
                            "an unanswered question cannot carry an answer",
                        )
                    )
                if isinstance(claims, list) and claims:
                    issues.append(
                        _issue(
                            f"{base}.claims",
                            "unanswered",
                            "an unanswered question cannot carry answer claims",
                        )
                    )
            elif status in {"answered", "partially_answered"}:
                if not answers or not isinstance(combined, str) or not combined.strip():
                    issues.append(
                        _issue(
                            f"{base}.answers",
                            "answer_required",
                            "an answered question requires answer evidence and a combined answer",
                        )
                    )

            if isinstance(combined, str):
                missing = sorted(_numbers(combined) - quoted_numbers)
                if missing:
                    issues.append(
                        _issue(
                            f"{base}.combined_answer",
                            "number_not_quoted",
                            f"combined answer number(s) {', '.join(missing)} appear in no answer or claim quote",
                        )
                    )
    return issues


def _finalize(
    document: Mapping[str, Any],
    issues: list[ValidationIssue],
    *,
    raise_on_error: bool,
) -> ValidationResult:
    if issues:
        if raise_on_error:
            raise EvidenceValidationError(issues)
        return issues
    return _normalized_copy(document)


def _validate_document(
    kind: EvidenceKind,
    document: object,
    diarized_document: object,
    visual_evidence: object | None,
    *,
    raise_on_error: bool,
    confirmed_names: Mapping[str, str] | None = None,
) -> ValidationResult:
    issues = _json_issues(document)
    issues.extend(_schema_issues(kind, document))
    index, source_issues = _source_index(diarized_document, visual_evidence, confirmed_names)
    issues.extend(source_issues)
    if not issues and isinstance(document, Mapping) and index is not None:
        semantic = {
            "speaker": _speaker_semantic_issues,
            "qa": _qa_semantic_issues,
        }[kind]
        issues.extend(semantic(document, index))
    if not isinstance(document, Mapping):
        if raise_on_error:
            raise EvidenceValidationError(issues)
        return issues
    return _finalize(document, issues, raise_on_error=raise_on_error)


def validate_speaker_evidence(
    document: object,
    diarized_document: object,
    visual_evidence: object | None = None,
    *,
    raise_on_error: bool = True,
    confirmed_names: Mapping[str, str] | None = None,
) -> ValidationResult:
    """Validate speaker naming against schema, transcript, and visual sources.

    ``confirmed_names`` maps speaker labels to names that a human operator
    confirmed through the CLI; they are read from the run manifest, never
    from the evidence document itself.
    """

    return _validate_document(
        "speaker",
        document,
        diarized_document,
        visual_evidence,
        raise_on_error=raise_on_error,
        confirmed_names=confirmed_names,
    )


def validate_qa_evidence(
    document: object,
    diarized_document: object,
    visual_evidence: object | None = None,
    *,
    raise_on_error: bool = True,
) -> ValidationResult:
    """Validate questions, answers, claims, quotes, and their ordering."""

    return _validate_document(
        "qa",
        document,
        diarized_document,
        visual_evidence,
        raise_on_error=raise_on_error,
    )


def validate_evidence_bundle(
    speaker_document: object,
    qa_document: object,
    diarized_document: object,
    visual_evidence: object | None = None,
    *,
    raise_on_error: bool = True,
    confirmed_names: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]] | list[ValidationIssue]:
    """Validate both evidence documents and return them normalized, or issues."""

    issues: list[ValidationIssue] = []
    normalized: dict[str, dict[str, Any]] = {}
    for name, kind, document in (
        ("speaker", "speaker", speaker_document),
        ("qa", "qa", qa_document),
    ):
        result = _validate_document(
            kind,  # type: ignore[arg-type]
            document,
            diarized_document,
            visual_evidence,
            raise_on_error=False,
            confirmed_names=confirmed_names,
        )
        if isinstance(result, list):
            for issue in result:
                path = issue["path"]
                prefixed = f"$.{name}{path[1:]}" if path.startswith("$.") or path == "$" else path
                issues.append({**issue, "path": prefixed})
        else:
            normalized[name] = result
    if issues:
        if raise_on_error:
            raise EvidenceValidationError(issues)
        return issues
    return normalized


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceValidationError",
    "load_evidence_schema",
    "normalize_text",
    "validate_evidence_bundle",
    "validate_qa_evidence",
    "validate_speaker_evidence",
]
