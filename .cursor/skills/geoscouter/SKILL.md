---
name: geoscouter
description: >-
  Operate on the GEOScouter Streamlit app and geoscouter Python package (GEO
  dataset scraping, filtering, visualization, metadata curation). Use when
  editing GEOScouter, streamlit_app.py, geoscouter/, tests/, collaboration
  PRs to dev, or GEO supplementary-file / GDS workflow questions in this repo.
---

# GEOScouter agent skill

## Repository layout

- **`streamlit_app.py`** — Streamlit UI (steps 1–5), session state, downloads.
- **`geoscouter/`** — installable library; put reusable logic here.
- **`tests/`** — `unittest` suite; run before PRs.
- **`doc/`** — operator docs; code is source of truth for behavior.
- **`CONTRIBUTING.md`** — public collaborator workflow (branch `dev`, tests).
- **`archive/KIDS25-Team17/`** — hackathon provenance only; not the live app.

## Branch and release workflow

- Integrate on **`dev`** via feature branches and PRs to **`dev`**.
- **`main`** is for releases (maintainer merges `dev` → `main`, tags version).
- Keep PRs **one topic**; avoid drive-by refactors.

## App step → code map

| Step | Primary modules |
|------|-------------------|
| 1 Scrape | `geoscouter/core/pipeline.py`, `gds_parse.py`, `tar_expand.py`, `platforms.py` |
| 2 Filter | `geoscouter/core/filters.py`, `platforms.py` |
| 3 Viz | `geoscouter/viz/`, `geoscouter/core/similarity.py` |
| 4 GSE list | `streamlit_app.py`, `similarity.py` |
| 5 Metadata | `geoscouter/core/metadata.py`, `curation.py`, `filters.py` |

Prefer new logic in **`geoscouter/`**; wire UI in **`streamlit_app.py`**. Do not peel Streamlit out of core unless already editing that module for a feature.

## Commands

```bash
cd /path/to/GEOScouter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
streamlit run streamlit_app.py
```

Example input: `data/input/gds_result.txt`. CI runs the same unittest + `import geoscouter; import streamlit_app` on PRs to `dev`.

## Change guidelines

- **Minimal diffs** — match existing style; no large reformatting.
- **Tests** — if `geoscouter/core/` behavior changes, add or update tests in `tests/`.
- **No live GEO in unit tests** — mock HTTP or use fixtures; scraping E2E is manual.
- **Git commits** — plain `git commit -m "..."` only; **no** `Co-authored-by`, **no** `--trailer`, **no** Cursor attribution in messages.
- **Authorship** — humans only on papers and LICENSE; AI is a tool, not a co-author.

## When unsure

Read `doc/README.md` for the pipeline map. Point contributors to GitHub Issues (template `contribution.yml`) or `CONTRIBUTING.md`.
