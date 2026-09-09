# Visualization — steps 3 and 4

Explore the **active** (filtered) dataset: summary plots, file complexity, supplementary-file similarity networks, and GSE comparison lists.

**Source:** [streamlit_app.py](../../streamlit_app.py) (sections 3–4), [geoscouter/viz/](../../geoscouter/viz/), [similarity.py](../../geoscouter/core/similarity.py)

---

## Step 3 — Visualize datasets

Available after scrape-level filters produce a non-empty `df_active`.

### Dataset snapshot

**Show dataset snapshot** — bar/summary views of series in the working set ([snapshot.py](../../geoscouter/viz/snapshot.py)).

### File-per-sample complexity

**Show file-per-sample complexity** — how many supplementary files per sample (series-level heuristic) ([complexity.py](../../geoscouter/viz/complexity.py)).

### Supplementary file similarity network

Two modes (radio **Network type**):

#### Pairwise / reference

- Builds edges with **Jaccard similarity** on full supplementary filenames across series.
- Optional **Reference GSE**: star layout comparing every other series to one chosen layout.
- Threshold slider controls which edges are drawn.

See `SIMILARITY_HELP` in [similarity.py](../../geoscouter/core/similarity.py) for score interpretation.

#### Supervised structure

1. Select one or more **training GSEs** with a file layout you want to match.
2. Review/edit **match patterns** derived from their filenames (e.g. `transcripts.csv.gz`, `matrix.mtx.gz`, `experiment.xenium.txt.gz`).
3. **Apply signature & show network** — scores other GSEs by weighted rule coverage (not plain Jaccard).

Useful when technologies share structural suffixes but not identical prefixes.

Plots: [network.py](../../geoscouter/viz/network.py) (`file_similarity_network`, `supervised_file_similarity_network`).

---

## Step 4 — Select specific GSEs

**Build a comparison list** — multiselect GSEs from the active set into `gse_selection_list`.

| Action | Result |
|--------|--------|
| View selection table | Subset of `df_active` for chosen series |
| Download filtered CSV | Writes `filtered_geo_webscrap.csv` to `WORK_DIR` |

---

## Similarity inputs

Networks use the **`File type/resource`** column when present (full filename on `dev`); otherwise **`Supplementary file`**.

Filename normalization for patterns: [similarity.py](../../geoscouter/core/similarity.py) (`normalize_filename`, `filename_to_pattern`).

---

## Session behavior

- Changing filters or reference GSE resets hidden plot flags until you click show buttons again.
- Supervised pattern editor state is kept in session (`supervised_pattern_editor_df`, `supervised_applied_rules`).

---

## Related

- [filtering/README.md](../filtering/README.md) — how `df_active` is built
- [pipeline/02_web_scraping.md](../pipeline/02_web_scraping.md) — where supplementary filenames come from
- [reference/series_matrix_files.md](../reference/series_matrix_files.md) — alternative data source (not used by default)
