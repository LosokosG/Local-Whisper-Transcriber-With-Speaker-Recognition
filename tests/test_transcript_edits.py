from __future__ import annotations

import hashlib
import json

import pytest


def canonical() -> dict[str, object]:
    return {
        "segments": [
            {
                "id": "seg-000001",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "text": "Original text",
            }
        ]
    }


def digest(document: dict[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def test_transcript_overlay_requires_exact_canonical_hash_binding() -> None:
    from transcript_pipeline.transcript_edits import TranscriptEditError, apply_transcript_edits

    document = canonical()
    edits = {
        "schema_version": 2,
        "diarized_sha256": "0" * 64,
        "edits": [
            {"segment_id": "seg-000001", "finalized_text": "Reviewed text"}
        ],
    }

    with pytest.raises(TranscriptEditError, match="stale"):
        apply_transcript_edits(document, edits, canonical_sha256=digest(document))


def test_bound_transcript_overlay_applies_only_to_a_copy() -> None:
    from transcript_pipeline.transcript_edits import apply_transcript_edits

    document = canonical()
    edits = {
        "schema_version": 2,
        "diarized_sha256": digest(document),
        "edits": [
            {"segment_id": "seg-000001", "finalized_text": "Reviewed text"}
        ],
    }

    rendered = apply_transcript_edits(
        document, edits, canonical_sha256=digest(document)
    )

    assert "finalized_text" not in document["segments"][0]  # type: ignore[index]
    assert rendered["segments"][0]["finalized_text"] == "Reviewed text"


def test_unbound_legacy_transcript_overlay_is_rejected() -> None:
    from transcript_pipeline.transcript_edits import TranscriptEditError, apply_transcript_edits

    document = canonical()
    legacy = {
        "schema_version": 1,
        "edits": [
            {"segment_id": "seg-000001", "finalized_text": "Reviewed text"}
        ],
    }

    with pytest.raises(TranscriptEditError, match="schema_version"):
        apply_transcript_edits(document, legacy, canonical_sha256=digest(document))
