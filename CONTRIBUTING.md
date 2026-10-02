# Contributing to GEOScouter

Thank you for helping improve GEOScouter. This document covers the essentials; full project docs are in [README.md](README.md) and [doc/README.md](doc/README.md).

## Branch workflow

- **`dev`** is the integration branch (latest features after review).
- Create a **feature branch from `dev`**, open a **pull request into `dev`**, and delete your branch after merge.
- **`main`** is updated by the maintainer for **releases** only — please do not PR directly to `main` unless asked.

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-short-description
# … edit, commit …
git push -u origin feature/your-short-description
```

Keep each PR focused on **one feature or fix**.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Example input: [data/input/gds_result.txt](data/input/gds_result.txt).

## Before you open a PR

Run the test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

If you change behavior in `geoscouter/core/` or library code under `geoscouter/`, **add or update tests** in `tests/`.

## Where to change code

| App step | Main modules |
|----------|----------------|
| 1 — Run data scraping | `geoscouter/core/pipeline.py`, `gds_parse.py`, `tar_expand.py` |
| 2 — Filter datasets | `geoscouter/core/filters.py`, `platforms.py` |
| 3 — Visualize datasets | `geoscouter/viz/`, `geoscouter/core/similarity.py` |
| 4 — Select GSEs | `streamlit_app.py`, `similarity.py` |
| 5 — Metadata & curation | `geoscouter/core/metadata.py`, `curation.py`, `filters.py` |

UI wiring lives in [streamlit_app.py](streamlit_app.py); prefer putting reusable logic in the `geoscouter` package.

## Proposing changes

- For **non-trivial** work, open a [GitHub Issue](https://github.com/Mmaycon/GEOScouter/issues/new) first. Mention the **step (1–5)** in the title or description.
- Link the issue in your PR.
- Avoid unrelated refactors or large formatting-only diffs.

## Conduct and transparency

- Be respectful in issues and reviews.
- If AI tools generated a **substantial** part of your PR, say so in the PR description so maintainers can review carefully.

## Questions

Open an issue or tag [@Mmaycon](https://github.com/Mmaycon) on GitHub.
