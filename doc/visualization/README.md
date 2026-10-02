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

### Supervised file-structure signature network

Choose a **signature source**:

| Mode | What you do |
|------|-------------|
| **From training GSE(s)** | Select one or more training GSEs; review deduplicated match patterns; apply signature |
| **Manual file types** | Type expected file-type tokens (one per line); no training GSE required |

Then **Apply signature & show network** — scores other GSEs by weighted rule coverage. The comparison table links each GSE to its [GEO accession page](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE325935).

See `SUPERVISED_SIMILARITY_HELP` in [similarity.py](../../geoscouter/core/similarity.py) for score interpretation.

Plot: [network.py](../../geoscouter/viz/network.py) (`supervised_file_similarity_network`).

---

## Step 4 — Select specific GSEs

**Build a comparison list** — add GSEs manually or bulk-add by **signature similarity** (requires an applied signature from step 3).

| Action | Result |
|--------|--------|
| Add GSEs matching signature | Adds GSEs whose supervised score meets the threshold |
| View selection table | Subset of `df_active` for chosen series |
| Download filtered CSV | Writes `filtered_geo_webscrap.csv` to `WORK_DIR` |

---

## Similarity inputs

Networks use the **`File type/resource`** column when present (full filename on `dev`); otherwise **`Supplementary file`**.

Filename normalization for patterns: [similarity.py](../../geoscouter/core/similarity.py) (`normalize_filename`, `filename_to_pattern`, `structural_match_for_filename`).

---

## Session behavior

- Changing filters resets hidden plot flags until you click show buttons again.
- Supervised pattern editor state is kept in session (`supervised_pattern_editor_df`, `supervised_applied_rules`, `supervised_signature_mode`).

---

## Related

- [filtering/README.md](../filtering/README.md) — how `df_active` is built
- [pipeline/02_web_scraping.md](../pipeline/02_web_scraping.md) — where supplementary filenames come from
- [reference/series_matrix_files.md](../reference/series_matrix_files.md) — alternative data source (not used by default)
