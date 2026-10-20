# Local Whisper Transcriber with Speaker Recognition

A local, resumable pipeline that turns a long recording (a talk, a meeting, an
interview, a Q&A session) into a speaker-attributed transcript and a Q&A
summary you can check: every answer and every claim points back to a
timestamped quote in the transcript.

## The problem

Speech-to-text is the easy part. The hard part is trusting what comes after it:
who said what, which questions were asked, and whether the summary of an answer
is something the speaker actually said. A language model can draft that
summary quickly, but it can also attach a quote to the wrong person, merge two
answers, or produce a confident number nobody mentioned.

This project keeps the model as a proposer and puts a deterministic validator
between its proposals and the published files.

## How it works

```text
recording (mp4, mkv, wav, m4a, ...)
  |
  | prepare          ffprobe + ffmpeg  -> audio.flac (16 kHz mono, decode-verified)
  |                  WhisperX (faster-whisper, wav2vec2 alignment) + pyannote diarization
  |                  -> .work/diarized_segments.json   seg-000001 ... with SPEAKER_xx labels
  |
  | infer-event      optional: recording date from corroborating signals
  | extract-frames   optional, video only: frames around speaker turns, Tesseract OCR
  |
  | validate         an agent or a human proposes speakers.json + qa.json
  |                  -> JSON Schema + cross-checks against the transcript and frames
  |
  | render           transcript.md, qa.md, review.md -> final validation
  v                  -> READY, or NOT READY with the exact items that need a decision
.work/manifest.json  stage state, SHA-256 of every input and output, blockers, decisions
```

## Design decisions

- **Resumable manifest with hashes.** Each stage records what it produced and
  what it consumed. If a file changes underneath (a re-run transcription, a
  hand-edited evidence file), the stages that depend on it are invalidated
  instead of silently mixing old and new data. Expensive steps never re-run
  just because a later step failed.
- **The agent proposes, the validator disposes.** Evidence is checked against
  the run's own sources, not just a schema: cited segments and frames must
  exist, quotes must occur in the cited segment, `asked_by`/`answered_by` must
  match the diarized speaker, an answer cannot end before its question was
  asked, and every name/date/number/commitment claim needs a verbatim quote.
- **No guessed names.** A speaker is published by name only when a definitive
  binding can be re-derived: a self-introduction quote containing the name, an
  introduction followed by that person's turn, an aligned "speaking" frame
  observation of the exact name, or a confirmation a human recorded through
  the CLI. Everyone else is `Unidentified Speaker N`.
- **Blockers instead of guesses.** Ambiguous dates, contradictory speaker
  evidence, and failed checks stop the run with exit code 3 and a list in
  `review.md`. Nothing is marked ready by hand.
- **Canonical transcript stays untouched.** Corrections and redactions of
  individual segments are an overlay (`transcript_edits.json`) bound to the
  transcript's hash.
- **Local processing, credentials kept out of files.** Audio and text are
  processed on your machine and the pipeline itself makes no network calls
  (the speech libraries download their models on first use). The Hugging Face
  token is read from `HF_TOKEN` when the diarization model is loaded (and by
  the validator, to check that its value appears in no run file); it is never
  accepted as an argument or written down. Error messages are redacted before
  they reach the manifest, unexpected CLI errors print only the exception
  type, and validation scans the whole run directory for credential patterns.
- **Light core.** Everything except the speech models runs, and is tested,
  without a GPU: WhisperX, torch and pyannote are an optional extra.

## Honest limits

- Real transcription needs the `ml` extra and, in practice, an NVIDIA GPU. CPU
  works but is slow.
- Diarization uses gated pyannote models: you need a Hugging Face account, to
  accept their terms, and `HF_TOKEN` in your environment.
- Frame sampling needs a video stream; OCR needs `tesseract` on the PATH.
- The validator checks grounding, not quality. It proves a quote exists and
  belongs to the right speaker; it cannot tell whether the agent missed a
  question or wrote a weak summary.
- Quote matching ignores case, punctuation and spacing, but not wording.
- Linux/macOS only (the manifest uses POSIX file locks).
- Used on the author's own recordings only; there are no accuracy numbers. The
  automated tests use synthetic fixtures and fakes for WhisperX and pyannote.

## Running it

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), FFmpeg, and for
the optional frame step, Tesseract.

```bash
uv sync --extra ml
uv run transcript-pipeline prepare /recordings/team-sync.mp4 --title "Team sync" --num-speakers 3
uv run transcript-pipeline status --run outputs/2026-03-14_team-sync
```

`prepare` prints the run directory (dated from the file's creation metadata
when it has any). Then, for the optional stages:

```bash
uv run transcript-pipeline infer-event --run outputs/2026-03-14_team-sync --event-start 2026-03-14T10:30:00+01:00
uv run transcript-pipeline extract-frames --run outputs/2026-03-14_team-sync
```

Write `speakers.json` and `qa.json` (schemas in `transcript_pipeline/schemas/`,
examples in `tests/fixtures/`), then validate and render:

```bash
uv run transcript-pipeline validate --run outputs/2026-03-14_team-sync --phase evidence --speaker-evidence speakers.json --qa-evidence qa.json
uv run transcript-pipeline render --run outputs/2026-03-14_team-sync
uv run transcript-pipeline resume --run outputs/2026-03-14_team-sync
```

Exit codes: `0` done, `2` usage error, `3` waiting for a human decision,
`4` stage failure, `5` validation failure.

`skill/process-recording/SKILL.md` describes how a coding agent should drive
these stages: draft the evidence, read the validator's issues, fix them, and
stop at blockers instead of guessing.

Tests (no GPU, models, token, or network needed; FFmpeg-based tests are
skipped when FFmpeg is missing):

```bash
uv sync --extra test
uv run pytest
```

## Origin

Started in May 2025 as a small Whisper script; rebuilt in 2026 (WhisperX,
diarization, evidence validation) as part of a private event-processing
pipeline. This repository contains the generic parts; client-specific stages
were removed. The original 2025 desktop script is still in the git history.

## License

MIT. See [LICENSE](LICENSE).
