# Review: Synthetic session
**Status:** READY — no blocking review items.
**Date:** March 14, 2026

This file is private working material: it contains raw evidence quotes.

## Blocking items
None.

## Notes
- **speaker_unidentified:** SPEAKER_01 is published as Unidentified Speaker 1.
- **speaker_unidentified:** SPEAKER_02 is published as Unidentified Speaker 2.
- **question_unanswered:** question q-mobile is marked unanswered.

## Speakers
| Label | Status | Published as | Confidence | Candidate name | Evidence | Contradictions |
| --- | --- | --- | --- | --- | --- | --- |
| SPEAKER_00 | identified | Ada Example | high | Ada Example | self_identification: seg-000001 [00:00:00] "I'm Ada Example and I'll host" | — |
| SPEAKER_01 | unidentified | Unidentified Speaker 1 | low | — | — | — |
| SPEAKER_02 | unidentified | Unidentified Speaker 2 | low | — | — | — |

## Question evidence

### q-import-duration (answered)
- **Topic:** Import jobs
- **Question:** How long does the import job take for a large archive?
- **Asked:** seg-000003 [00:00:12] "How long does the import job take"
- **Answer 1:** seg-000004 [00:00:18] "it usually finishes in about 40 minutes on one worker"
- **Claim (number):** About 40 minutes for a ten gigabyte archive on one worker — seg-000004 [00:00:18] "about 40 minutes on one worker"

### q-import-resume (answered)
- **Topic:** Import jobs
- **Question:** Does the import resume after a crash?
- **Asked:** seg-000005 [00:00:30]
- **Answer 1:** seg-000006 [00:00:34] "every stage writes a checkpoint"
- **Claim (commitment):** Every stage writes a checkpoint — seg-000006 [00:00:34] "every stage writes a checkpoint"

### q-mobile (unanswered)
- **Topic:** Roadmap
- **Question:** Will there be a mobile version next year?
- **Asked:** seg-000007 [00:00:45]
