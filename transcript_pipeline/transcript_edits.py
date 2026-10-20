"""Reviewed transcript-text overlays that preserve canonical diarization output."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any


class TranscriptEditError(ValueError):
    """A reviewed edit document cannot be applied to its canonical transcript."""


def apply_transcript_edits(
    diarized: Mapping[str, Any],
    edits_document: Mapping[str, Any],
    *,
    canonical_sha256: str,
) -> dict[str, Any]:
    """Return a copied transcript with reviewed text edits applied by segment ID."""

    if edits_document.get("schema_version") != 2:
        raise TranscriptEditError("transcript edits schema_version must be 2")
    if set(edits_document) != {"schema_version", "diarized_sha256", "edits"}:
        raise TranscriptEditError("transcript edits contain unsupported fields")
    if not isinstance(canonical_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", canonical_sha256
    ) is None:
        raise TranscriptEditError("canonical transcript hash is invalid")
    bound_sha256 = edits_document.get("diarized_sha256")
    if (
        not isinstance(bound_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", bound_sha256) is None
    ):
        raise TranscriptEditError("transcript edits diarized_sha256 is invalid")
    if bound_sha256 != canonical_sha256:
        raise TranscriptEditError("transcript edits are stale for the canonical transcript")
    raw_edits = edits_document.get("edits")
    if not isinstance(raw_edits, list):
        raise TranscriptEditError("transcript edits must be an array")
    raw_segments = diarized.get("segments")
    if not isinstance(raw_segments, list):
        raise TranscriptEditError("canonical transcript segments are unavailable")

    segments_by_id: dict[str, dict[str, Any]] = {}
    copied = copy.deepcopy(dict(diarized))
    copied_segments = copied.get("segments")
    assert isinstance(copied_segments, list)
    for segment in copied_segments:
        if not isinstance(segment, dict):
            raise TranscriptEditError("canonical transcript segment is malformed")
        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id or segment_id in segments_by_id:
            raise TranscriptEditError("canonical transcript segment IDs must be unique")
        segments_by_id[segment_id] = segment

    seen: set[str] = set()
    for edit in raw_edits:
        if not isinstance(edit, Mapping) or set(edit) != {
            "segment_id",
            "finalized_text",
        }:
            raise TranscriptEditError("each transcript edit requires only segment_id and finalized_text")
        segment_id = edit.get("segment_id")
        finalized_text = edit.get("finalized_text")
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen:
            raise TranscriptEditError("transcript edit segment IDs must be unique")
        if segment_id not in segments_by_id:
            raise TranscriptEditError("transcript edit references an unknown segment")
        if not isinstance(finalized_text, str) or not finalized_text.strip():
            raise TranscriptEditError("reviewed transcript text must be non-empty")
        if "\x00" in finalized_text:
            raise TranscriptEditError("reviewed transcript text contains an invalid character")
        seen.add(segment_id)
        segments_by_id[segment_id]["finalized_text"] = finalized_text.strip()
    return copied


__all__ = ["TranscriptEditError", "apply_transcript_edits"]
