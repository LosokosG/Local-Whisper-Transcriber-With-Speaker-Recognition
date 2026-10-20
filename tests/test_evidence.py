from __future__ import annotations

import copy
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from transcript_pipeline.evidence import (
    EvidenceValidationError,
    load_evidence_schema,
    normalize_text,
    validate_evidence_bundle,
    validate_qa_evidence,
    validate_speaker_evidence,
)


def _codes(result: object) -> set[str]:
    assert isinstance(result, list), "expected validation issues"
    return {issue["code"] for issue in result}


def _speaker(document: dict[str, Any], label: str) -> dict[str, Any]:
    return next(record for record in document["speakers"] if record["speaker_label"] == label)


def _question(document: dict[str, Any], question_id: str) -> dict[str, Any]:
    return next(
        question
        for topic in document["topics"]
        for question in topic["questions"]
        if question["id"] == question_id
    )


def _visual(observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sources": [
            {
                "frame_id": "frame_000003000",
                "timestamp_seconds": 3.0,
                "status": "ok",
                "raw_path": "/run/.work/frames/frame_000003000.png",
            },
            {
                "frame_id": "frame_000040000",
                "timestamp_seconds": 40.0,
                "status": "ok",
                "raw_path": "/run/.work/frames/frame_000040000.png",
            },
            {"frame_id": "frame_000050000", "timestamp_seconds": 50.0, "status": "error"},
        ],
        "presence_observations": observations or [],
    }


def _speaking_observation(timestamp: float, frame_id: str, **changes: Any) -> dict[str, Any]:
    observation = {
        "id": f"obs-{frame_id}",
        "frame_id": frame_id,
        "timestamp_seconds": timestamp,
        "region": "speakers",
        "evidence_type": "speaking_indicator",
        "confidence": 0.9,
        "exact_visible_name": "Ada Example",
        "truncated": False,
    }
    observation.update(changes)
    return observation


@pytest.mark.parametrize("kind", ["speaker", "qa"])
def test_packaged_schemas_are_draft_2020_12_strict_and_detached(kind: str) -> None:
    schema = load_evidence_schema(kind)

    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    schema["title"] = "mutated"
    assert load_evidence_schema(kind)["title"] != "mutated"


def test_schema_loader_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown evidence schema kind"):
        load_evidence_schema("../../etc/passwd")


def test_valid_bundle_returns_detached_normalized_documents(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    result = validate_evidence_bundle(speaker_evidence, qa_evidence, diarized)

    assert isinstance(result, dict)
    assert result["speaker"] == speaker_evidence
    assert result["qa"] == qa_evidence
    result["qa"]["topics"].clear()
    assert qa_evidence["topics"]


def test_unknown_fields_are_rejected_at_exact_json_paths(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    qa_evidence["unexpected"] = True
    qa_evidence["topics"][0]["questions"][0]["answers"][0]["confidence"] = "high"

    issues = validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)

    assert isinstance(issues, list)
    assert {issue["path"] for issue in issues} == {
        "$",
        "$.topics[0].questions[0].answers[0]",
    }
    assert all(issue["code"] == "schema" for issue in issues)


def test_quote_requires_a_segment_reference(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    _question(qa_evidence, "q-mobile")["asked_sources"] = [
        {"visual_frame_id": "frame_000003000", "quote": "mobile version"}
    ]

    issues = validate_qa_evidence(qa_evidence, diarized, _visual(), raise_on_error=False)

    assert any(issue["message"] == "a quote requires a segment_id" for issue in issues)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "quote",
    [
        "about 45 minutes on one worker",
        "finishes in about 40 minutes on two workers",
        "it usually finishes",
        "usually",
    ],
)
def test_quote_must_occur_in_the_cited_segment(
    diarized: dict[str, Any], qa_evidence: dict[str, Any], quote: str
) -> None:
    question = _question(qa_evidence, "q-import-duration")
    question["claims"][0]["sources"][0]["quote"] = quote
    # A quote copied from another segment does not count either.
    question["answers"][0]["sources"][0] = {"segment_id": "seg-000003", "quote": "it usually finishes"}

    codes = _codes(validate_qa_evidence(qa_evidence, diarized, raise_on_error=False))

    assert "quote_not_found" in codes


def test_quote_matching_ignores_case_punctuation_and_spacing(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    _question(qa_evidence, "q-import-resume")["claims"][0]["sources"][0]["quote"] = (
        "YES —  every   stage writes a checkpoint!"
    )

    assert isinstance(validate_qa_evidence(qa_evidence, diarized), dict)
    assert normalize_text("I'm  Ada-Example.") == "i m ada example"


def test_quote_must_not_match_inside_a_longer_word(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    _question(qa_evidence, "q-import-resume")["claims"][0]["sources"][0]["quote"] = "heck"

    assert "quote_not_found" in _codes(
        validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)
    )


def test_quotes_must_match_canonical_text_not_reviewed_edits(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    diarized["segments"][3]["finalized_text"] = (
        "For a 10 GB archive it usually finishes in about 40 minutes on one worker."
    )
    _question(qa_evidence, "q-import-duration")["claims"][0]["sources"][0]["quote"] = "a 10 GB archive"

    assert "quote_not_found" in _codes(
        validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)
    )


def test_a_name_inserted_by_a_transcript_edit_cannot_bind_a_speaker(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    diarized["segments"][2]["finalized_text"] = "I'm Mallory Invented. How long does the import job take?"
    guest = _speaker(speaker_evidence, "SPEAKER_01")
    guest.update(
        status="identified",
        confidence="high",
        display_name="Mallory Invented",
        output_label="Mallory Invented",
        binding_evidence=[
            {"type": "self_identification", "sources": [{"segment_id": "seg-000003", "quote": "I'm Mallory Invented"}]}
        ],
    )

    assert {"quote_not_found", "binding_not_definitive"} <= _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_claims_need_a_verbatim_quote(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    del _question(qa_evidence, "q-import-duration")["claims"][0]["sources"][0]["quote"]

    assert "quote_required" in _codes(
        validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)
    )


def test_answers_need_a_quoted_transcript_segment(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    _question(qa_evidence, "q-import-duration")["answers"][0]["sources"] = [
        {"visual_frame_id": "frame_000040000"}
    ]

    assert "quote_required" in _codes(
        validate_qa_evidence(qa_evidence, diarized, _visual(), raise_on_error=False)
    )


def test_fabricated_source_ids_are_rejected(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    question = _question(qa_evidence, "q-import-resume")
    question["asked_sources"] = [{"segment_id": "seg-999999"}]
    question["answers"][0]["sources"].append({"visual_frame_id": "frame_000099000"})

    issues = validate_qa_evidence(qa_evidence, diarized, _visual(), raise_on_error=False)

    assert isinstance(issues, list)
    assert {issue["code"] for issue in issues} == {"unknown_source"}
    assert any("seg-999999" in issue["message"] for issue in issues)


def test_failed_frame_extraction_cannot_be_cited(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    _question(qa_evidence, "q-mobile")["asked_sources"].append({"visual_frame_id": "frame_000050000"})

    assert "unknown_source" in _codes(
        validate_qa_evidence(qa_evidence, diarized, _visual(), raise_on_error=False)
    )


def test_frame_can_be_cited_by_unique_extracted_basename(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    visual = _visual()
    for source in visual["sources"]:
        source.pop("frame_id")
    _question(qa_evidence, "q-mobile")["asked_sources"].append({"visual_frame_id": "frame_000040000"})

    assert isinstance(validate_qa_evidence(qa_evidence, diarized, visual), dict)


def test_asked_by_and_answered_by_must_match_cited_segments(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    question = _question(qa_evidence, "q-import-duration")
    question["asked_by"] = "SPEAKER_00"
    question["answers"][0]["answered_by"] = "SPEAKER_09"

    codes = _codes(validate_qa_evidence(qa_evidence, diarized, raise_on_error=False))

    assert codes == {"speaker_mismatch", "unknown_speaker"}


def test_unanswered_question_cannot_carry_answers_or_claims(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    question = _question(qa_evidence, "q-mobile")
    question["combined_answer"] = "Probably next spring."
    question["claims"] = [
        {"kind": "date", "text": "next spring", "sources": [{"segment_id": "seg-000007", "quote": "next year"}]}
    ]

    assert _codes(validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)) == {"unanswered"}


@pytest.mark.parametrize("status", ["answered", "partially_answered"])
def test_answered_question_requires_answers_and_a_combined_answer(
    diarized: dict[str, Any], qa_evidence: dict[str, Any], status: str
) -> None:
    question = _question(qa_evidence, "q-mobile")
    question["status"] = status

    assert _codes(validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)) == {"answer_required"}


def test_answer_that_ends_before_the_question_is_rejected(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    question = _question(qa_evidence, "q-import-resume")
    early = {
        "text": "An earlier remark.",
        "answered_by": "SPEAKER_00",
        "sources": [{"segment_id": "seg-000002", "quote": "first question"}],
    }
    # One legitimate answer does not excuse another that precedes the question.
    question["answers"].append(early)

    issues = validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)

    assert isinstance(issues, list)
    assert [(issue["code"], issue["path"]) for issue in issues] == [
        ("answer_precedes_question", "$.topics[0].questions[1].answers[1].sources")
    ]


def test_numbers_in_claims_and_combined_answers_must_appear_in_quotes(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    question = _question(qa_evidence, "q-import-duration")
    question["claims"][0]["text"] = "About 400 minutes on 12 workers"
    question["combined_answer"] = "It takes about 45 minutes."

    issues = validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)

    assert isinstance(issues, list)
    assert {issue["path"]: issue["code"] for issue in issues} == {
        "$.topics[0].questions[0].claims[0].text": "number_not_quoted",
        "$.topics[0].questions[0].combined_answer": "number_not_quoted",
    }


def test_duplicate_topic_and_question_ids_are_rejected(
    diarized: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    qa_evidence["topics"].append(copy.deepcopy(qa_evidence["topics"][1]))

    assert _codes(validate_qa_evidence(qa_evidence, diarized, raise_on_error=False)) == {
        "duplicate_topic",
        "duplicate_question",
    }


def test_identified_speaker_requires_high_confidence(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    _speaker(speaker_evidence, "SPEAKER_00")["confidence"] = "medium"

    assert _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)) == {
        "confidence"
    }


def test_self_identification_quote_must_contain_the_published_name(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    record = _speaker(speaker_evidence, "SPEAKER_00")
    record["display_name"] = record["output_label"] = "Ada Lovelace"

    assert _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)) == {
        "binding_not_definitive"
    }


def test_self_identification_must_be_spoken_by_that_speaker(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    diarized["segments"][0]["speaker"] = "SPEAKER_01"
    diarized["segments"][2]["speaker"] = "SPEAKER_00"

    codes = _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False))

    assert {"speaker_mismatch", "binding_not_definitive"} <= codes


@pytest.mark.parametrize(
    "binding_type",
    ["verbal_mention", "visible_name_label", "ocr_visible_name", "timing_correlation", "voice_cluster_propagation"],
)
def test_supporting_bindings_alone_cannot_publish_a_name(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any], binding_type: str
) -> None:
    _speaker(speaker_evidence, "SPEAKER_00")["binding_evidence"][0]["type"] = binding_type

    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_a_mention_in_the_speakers_own_turn_is_not_an_introduction(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    diarized["segments"][2]["text"] = "Bo Sample asked me to read this: how long does the import job take?"
    guest = _speaker(speaker_evidence, "SPEAKER_01")
    guest.update(
        status="identified",
        confidence="high",
        display_name="Bo Sample",
        output_label="Bo Sample",
        binding_evidence=[
            {
                "type": "introduced_by_other",
                "sources": [
                    {"segment_id": "seg-000003", "quote": "Bo Sample asked me"},
                    {"segment_id": "seg-000005"},
                ],
            }
        ],
    )

    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_introduction_must_be_followed_directly_by_the_named_speaker(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    diarized["segments"][0]["text"] = "Welcome everyone, our guest Bo Sample joins later."
    _speaker(speaker_evidence, "SPEAKER_00").update(
        status="unidentified", display_name=None, output_label="Unidentified Speaker 3",
        confidence="low", binding_evidence=[],
    )
    guest = _speaker(speaker_evidence, "SPEAKER_01")
    guest.update(
        status="identified",
        confidence="high",
        display_name="Bo Sample",
        output_label="Bo Sample",
        binding_evidence=[
            {
                "type": "introduced_by_other",
                "sources": [
                    {"segment_id": "seg-000001", "quote": "our guest Bo Sample"},
                    {"segment_id": "seg-000003"},
                ],
            }
        ],
    )

    # seg-000002 (same host) does not break the turn, so seg-000003 is next: accepted.
    assert isinstance(validate_speaker_evidence(speaker_evidence, diarized), dict)

    # A different speaker answering first means the mention did not hand over.
    diarized["segments"][1]["speaker"] = "SPEAKER_02"
    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_introduction_by_another_speaker_needs_a_following_turn(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    diarized["segments"][1]["text"] = "Our first guest is Bo Sample, go ahead."
    guest = _speaker(speaker_evidence, "SPEAKER_01")
    guest.update(
        status="identified",
        confidence="high",
        display_name="Bo Sample",
        output_label="Bo Sample",
        binding_evidence=[
            {
                "type": "introduced_by_other",
                "sources": [
                    {"segment_id": "seg-000002", "quote": "our first guest is Bo Sample"},
                    {"segment_id": "seg-000003"},
                ],
            }
        ],
    )
    assert isinstance(validate_speaker_evidence(speaker_evidence, diarized), dict)

    # A turn that starts long after the introduction does not bind the name.
    guest["binding_evidence"][0]["sources"][1] = {"segment_id": "seg-000005"}
    diarized["segments"][4]["start"] = 90.0
    diarized["segments"][4]["end"] = 92.0
    for later in diarized["segments"][5:]:
        later["start"] += 60.0
        later["end"] += 60.0
    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_speaking_indicator_must_be_aligned_with_a_cited_turn(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    record = _speaker(speaker_evidence, "SPEAKER_00")
    record["binding_evidence"] = [
        {
            "type": "visible_speaking_indicator",
            "sources": [
                {"segment_id": "seg-000001"},
                {"visual_observation_id": "obs-frame_000003000"},
            ],
        }
    ]
    aligned = _visual([_speaking_observation(3.0, "frame_000003000")])
    assert isinstance(validate_speaker_evidence(speaker_evidence, diarized, aligned), dict)

    record["binding_evidence"][0]["sources"] = [
        {"segment_id": "seg-000001"},
        {"visual_observation_id": "obs-frame_000040000"},
    ]
    far = _visual([_speaking_observation(40.0, "frame_000040000")])
    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, far, raise_on_error=False)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"truncated": True},
        {"exact_visible_name": "Ada Ex"},
        {"evidence_type": "visible_label"},
    ],
)
def test_weak_visual_observations_do_not_bind_a_name(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any], changes: dict[str, Any]
) -> None:
    _speaker(speaker_evidence, "SPEAKER_00")["binding_evidence"] = [
        {
            "type": "visible_speaking_indicator",
            "sources": [{"segment_id": "seg-000001"}, {"visual_observation_id": "obs-frame_000003000"}],
        }
    ]
    visual = _visual([_speaking_observation(3.0, "frame_000003000", **changes)])

    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, visual, raise_on_error=False)
    )


def test_observation_must_agree_with_its_frame(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    visual = _visual([_speaking_observation(4.0, "frame_000003000")])

    assert "source_document" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, visual, raise_on_error=False)
    )


def test_contradictions_force_manual_review(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    _speaker(speaker_evidence, "SPEAKER_00")["contradictions"] = [
        "a different name appears on the slide at 00:00:40"
    ]

    assert _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)) == {
        "contradiction"
    }


def test_manual_review_keeps_candidate_private_and_unidentified_label_public(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    record = _speaker(speaker_evidence, "SPEAKER_01")
    record.update(status="manual_review", display_name="Bo Sample", contradictions=["voice differs later"])
    assert isinstance(validate_speaker_evidence(speaker_evidence, diarized), dict)

    record["output_label"] = "Bo Sample"
    assert "output_label" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_unidentified_speaker_cannot_carry_a_name(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    _speaker(speaker_evidence, "SPEAKER_02")["display_name"] = "Cy Placeholder"

    assert _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)) == {
        "unidentified"
    }


def test_every_transcript_speaker_needs_exactly_one_record(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    speaker_evidence["speakers"].pop()
    speaker_evidence["speakers"].append(copy.deepcopy(speaker_evidence["speakers"][1]))
    speaker_evidence["speakers"][-1]["output_label"] = "Unidentified Speaker 3"

    codes = _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False))

    assert codes == {"duplicate_speaker", "missing_speaker"}


def test_unknown_speaker_label_is_rejected(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    extra = copy.deepcopy(_speaker(speaker_evidence, "SPEAKER_02"))
    extra.update(speaker_label="SPEAKER_07", output_label="Unidentified Speaker 7")
    speaker_evidence["speakers"].append(extra)

    assert _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)) == {
        "unknown_speaker"
    }


def test_one_name_cannot_be_published_for_two_voices(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    diarized["segments"][2]["text"] = "Hi, I'm Ada Example too, how long does the import job take?"
    twin = _speaker(speaker_evidence, "SPEAKER_01")
    twin.update(
        status="identified",
        confidence="high",
        display_name="Ada Example",
        output_label="Ada Example",
        binding_evidence=[
            {"type": "self_identification", "sources": [{"segment_id": "seg-000003", "quote": "I'm Ada Example"}]}
        ],
    )

    codes = _codes(validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False))

    assert codes == {"conflicting_identity", "duplicate_output_label"}


def test_raise_mode_carries_every_sanitized_issue(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any], qa_evidence: dict[str, Any]
) -> None:
    _speaker(speaker_evidence, "SPEAKER_00")["confidence"] = "low"
    _question(qa_evidence, "q-mobile")["asked_sources"] = [{"segment_id": "seg-123456"}]

    with pytest.raises(EvidenceValidationError) as raised:
        validate_evidence_bundle(speaker_evidence, qa_evidence, diarized)

    paths = {issue["path"] for issue in raised.value.issues}
    assert "$.speaker.speakers[0].confidence" in paths
    assert "$.qa.topics[1].questions[0].asked_sources[0].segment_id" in paths


def test_non_finite_numbers_are_rejected_as_strict_json(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    speaker_evidence["speakers"][0]["confidence_score"] = math.nan

    assert "json" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_malformed_source_transcript_is_reported(
    speaker_evidence: dict[str, Any], diarized: dict[str, Any]
) -> None:
    diarized["segments"][1]["id"] = diarized["segments"][0]["id"]

    assert "source_document" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )


def test_wheel_contains_every_evidence_schema(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=False
    )
    uv = __import__("shutil").which("uv")
    if uv is None and completed.returncode != 0:
        pytest.skip("neither uv nor pip is available to build a wheel")
    command = (
        [uv, "build", "--wheel", "--out-dir", str(tmp_path), str(root)]
        if uv is not None
        else [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(tmp_path), str(root)]
    )
    built = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    assert built.returncode == 0, built.stderr[-2000:]
    wheel = next(tmp_path.glob("*.whl"))
    names = set(zipfile.ZipFile(wheel).namelist())
    for schema in ("speaker-evidence", "qa-evidence", "event-evidence"):
        assert f"transcript_pipeline/schemas/{schema}.schema.json" in names
    assert json.loads(
        zipfile.ZipFile(wheel).read("transcript_pipeline/schemas/qa-evidence.schema.json")
    )["$id"].startswith("urn:transcript-pipeline:")


def test_operator_confirmation_binds_only_the_confirmed_name_for_that_speaker(
    diarized: dict[str, Any], speaker_evidence: dict[str, Any]
) -> None:
    guest = _speaker(speaker_evidence, "SPEAKER_01")
    guest.update(
        status="identified",
        confidence="high",
        display_name="Bo Sample",
        output_label="Bo Sample",
        binding_evidence=[{"type": "operator_confirmation", "sources": [{"segment_id": "seg-000003"}]}],
    )

    # The document alone cannot claim an operator confirmation.
    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(speaker_evidence, diarized, raise_on_error=False)
    )
    assert "binding_not_definitive" in _codes(
        validate_speaker_evidence(
            speaker_evidence, diarized, raise_on_error=False, confirmed_names={"SPEAKER_01": "Someone Else"}
        )
    )
    assert isinstance(
        validate_speaker_evidence(speaker_evidence, diarized, confirmed_names={"SPEAKER_01": "bo  sample"}),
        dict,
    )
