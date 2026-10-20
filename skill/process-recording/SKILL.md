---
name: process-recording
description: Turn a local audio or video recording (talk, meeting, interview, Q&A session) into a speaker-attributed transcript and an evidence-checked Q&A summary using this repository's `transcript-pipeline` CLI. Use when the user hands over a recording and wants a transcript, a list of questions and answers, named speakers, or wants to resume or review an existing run.
---

# Process a recording

You drive a deterministic pipeline. You read the transcript and propose structured evidence; the validator decides whether that evidence is acceptable. You never write the public Markdown yourself and never mark anything as final by hand.

## Ground rules

- Work only through `uv run transcript-pipeline ...`. Run `status --run <dir>` first whenever you resume.
- Never print, copy, or pass `HF_TOKEN` (or any other credential) on a command line or into a file. The transcription step reads it from the environment on its own.
- Never edit files in `<run>/.work/` by hand, including `manifest.json`. Write your proposals to a scratch directory and hand them to `validate`, which checks them before copying them in.
- Treat every validation issue as information about your proposal. Fix the proposal; do not argue with the validator or route around it.
- A blocked stage (exit code 3) is a checkpoint for a human decision. Report it with the exact items from `review.md`; do not guess past it.

## Workflow

1. **Prepare.** `uv run transcript-pipeline prepare <absolute path> --title "<title>"` (add `--language`, `--num-speakers`, or `--timezone` when the user knows them). This probes the file, writes `audio.flac`, transcribes with WhisperX and diarizes with pyannote. Note the run directory it prints. If it blocks with `hf_token_unavailable`, ask the user to export `HF_TOKEN` and accept the pyannote model terms; do not try to work around diarization.
2. **Date (optional).** `infer-event --run <dir>` uses the container timestamp. One signal alone is not enough and will block; add corroborating evidence (`--evidence event.json`: a date said aloud, a clock on a slide) or an explicit `--event-start` that the user confirmed.
3. **Frames (optional, video only).** `extract-frames --run <dir>` samples frames around speaker turns and runs Tesseract OCR. Look at the contact sheets in `.work/frames/contact_sheets/`. If you see a name next to an active-speaker highlight, record it as an observation (`id`, `frame_id`, `timestamp_seconds`, `region`, `evidence_type: "speaking_indicator"`, `exact_visible_name`, `confidence`) and re-run with `--observations obs.json`. An observation must point at a real extracted frame.
4. **Read `.work/diarized_segments.json`** and draft two documents against the schemas in `transcript_pipeline/schemas/`:
   - `speakers.json`: one record per diarized `SPEAKER_xx` label. Use `status: "identified"` only with high confidence and a definitive binding: a self-introduction quote containing the name, an introduction by someone else followed by the person's own turn within a minute, or an aligned speaking-indicator observation of the exact name. Everyone else is `unidentified` (published as `Unidentified Speaker N`) or `manual_review` when you have a candidate name plus a contradiction or doubt.
   - `qa.json`: topics, questions, answers, and claims. Cite segment IDs. Quote the exact words (case and punctuation may differ, wording may not). Every claim with a name, date, number, price, or commitment needs a quoted segment. Mark a question `unanswered` or `partially_answered` instead of stretching an answer. Do not add facts that are not in the recording.
5. **Validate.** `validate --run <dir> --phase evidence --speaker-evidence speakers.json --qa-evidence qa.json`. Exit code 5 means at least one issue; each one has a JSON path and a code such as `quote_not_found`, `speaker_mismatch`, `binding_not_definitive`, or `answer_precedes_question`. Fix the proposal and run it again. Nothing is copied into the run until the whole bundle passes.
6. **Transcript corrections (optional).** For obvious ASR errors or a segment that must not be published, write a `transcript_edits.json` overlay (`schema_version: 2`, the SHA-256 of `diarized_segments.json`, and `{segment_id, finalized_text}` items) and pass it with `--transcript-edits`. The canonical transcript is never modified; your quotes may match either the original or the corrected text.
7. **Render.** `render --run <dir>` writes `transcript.md`, `qa.md`, and the private `review.md`, then runs final validation. Exit 0 means the review says READY. Exit 3 means `review.md` lists blocking items for the user.
8. **Hand over** the paths to `transcript.md`, `qa.md`, and `review.md`, plus every blocking item and every `Unidentified Speaker` that the user might be able to name.

## Resuming

Use `status` to get `next_action`, then `resume` for deterministic work. If the transcript or visual evidence changes after you validated, the run blocks with `reviewed_evidence_stale`: re-check your proposals against the new sources and supply both documents again. `--force-stage` is for re-running a stage after a real change, not for clearing a blocker you do not understand.

## Example: ambiguous speaker

A guest says "thanks for having me" and a frame shows the label "Bo S." near the speaking highlight, but the label is truncated and nobody says the full name. That is not a definitive binding. Record the guest as `manual_review` with `display_name: "Bo S."` as a candidate, add the truncated label as a `visible_name_label` binding, and explain the doubt in `contradictions`. The public files will say `Unidentified Speaker N` and `review.md` will ask the user to confirm the name.

If the user confirms the full name, change the record to `identified` with an `operator_confirmation` binding that cites the guest's own segments, and validate with `--confirm-speaker SPEAKER_xx "Full Name"`. The validator accepts that binding only when the same name was confirmed through the flag, and the decision is stored in the manifest and listed in `review.md`. Use the flag only to relay a confirmation the user actually gave; it is never a way to settle your own doubt.
