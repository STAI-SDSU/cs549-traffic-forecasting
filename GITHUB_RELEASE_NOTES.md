# GitHub Release Notes (Instructor Reference)

This file is instructor-facing guidance for publishing this repository. It
contains no secrets, hidden labels, or private leaderboard information, so
it is safe to keep tracked in the student-facing repository. Nothing in
this file is configured automatically — every step below is a
recommendation for the instructor to apply manually when ready.

## Recommended Repository Settings

| Setting | Recommendation |
|---|---|
| Repository name | `cs549-traffic-forecasting` |
| Description | `CS549 Fall 2026 Final Project — METR-LA Traffic Forecasting Challenge` |
| Visibility | Public, unless the instructor later chooses otherwise |
| Default branch | `main` |
| Issues | Optional — recommend **disabling** if project questions should stay on Canvas |
| Discussions | Optional — recommend **enabling only** if the instructor wants public class discussion |
| Wiki | Unnecessary — not recommended |
| GitHub Pages | Not required for the first version |

## Recommended First Release

| | |
|---|---|
| **Tag** | `fall-2026-v1.0` |
| **Title** | `CS549 Fall 2026 Project Release v1.0` |

This tag/release should represent the official, frozen set of project files
that students clone and work from for the semester. Do not move the tag
after students begin working from it.

Suggested release notes body:

```markdown
Official Fall 2026 release of the CS549 Traffic Forecasting Challenge.

Included:
- Prepared METR-LA training, validation, and test input data
- Starter code for loading data, scoring validation predictions, and
  building a submission file
- Two worked baseline model implementations
- Submission format and sample_submission.csv
- Project rules, grading breakdown, and detailed assignment description

See README.md to get started.
```

## Data Size / Git Suitability

All released files under `data/` are well under GitHub's 100 MB hard
per-file limit (largest is `sample_submission.csv` at roughly 48 MB, under
the 50 MB soft-warning threshold too). Ordinary Git is acceptable for this
release — no Git LFS or external hosting is required. If a future data
revision pushes any file over roughly 50 MB, prefer attaching it to a
GitHub Release rather than committing it directly, and only configure Git
LFS if that later becomes insufficient.

## Publishing Commands (do not run automatically)

To be run manually by the instructor when ready to publish:

```bash
git remote add origin <GITHUB_REPOSITORY_URL>
git push -u origin main
```

To create the recommended tag/release after pushing:

```bash
git tag -a fall-2026-v1.0 -m "CS549 Fall 2026 Project Release v1.0"
git push origin fall-2026-v1.0
```
(or create the release through the GitHub web UI, which also lets you
attach the release-notes body above.)
