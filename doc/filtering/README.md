# Filtering — step 2

Narrow the scraped dataset before visualizations and exports. Filters apply to the **working set** (`df_active`), not the raw cache (`df_combined`).

**Source:** [streamlit_app.py](../../streamlit_app.py) (section 2), [filters.py](../../geoscouter/core/filters.py), [platforms.py](../../geoscouter/core/platforms.py), [metadata.py](../../geoscouter/core/metadata.py)

---

## Scrape-level filters

Click **Apply scrape-level filters** after choosing criteria.

| Filter | UI control | Logic |
|--------|------------|-------|
| Assay type | Multiselect "Assay type (gds Type)" | Matches `Platform_labels` or `Study_type` from `gds_result.txt` |
| Platform (GPL) | Multiselect "Platform (GPL)" | Regex match on `Platforms` column; labels from GEO platform browser |
| Sample count | Min / max (0 = off) | Filters on `Samples` per series |
| Reset | **Reset to full scrape** | Restores `df_active` from `df_combined` |

Implementation: `apply_series_filters()` in [filters.py](../../geoscouter/core/filters.py).

Changing filters **resets step-3 visualizations** (plots hidden until regenerated).

---

## Optional: GSM sample metadata

**Fetch GEO sample metadata** downloads SOFT via GEOparse for each GSE in scope:

- **Active filtered series** — only GSEs in the current working set
- **All scraped series** — full `df_combined`

Output: list of per-GSE DataFrames in session (`list_of_metadata_dfs`), cached under `/tmp/geoscouter/metadata/geo_soft_files/`.

This step can take several minutes for large lists.

### Metadata field filters

After fetch, multiselects appear for common GSM fields (`organism_ch1`, `tissue`, `library_source`, `assay_type`, etc. — see `METADATA_FILTER_FIELDS` in filters.py).

**Apply metadata filters to working set** keeps only GSEs where at least one sample matches all selected field constraints.

---

## Outputs

| File | When written |
|------|--------------|
| `filtered_geo_webscrap.csv` | Exported from step 4 when saving the comparison / filtered set |
| `metadata_GSE.xlsx` | Step 5 — export all fetched GSM tables |
| `metadata_filtered_by_word.xlsx` | Step 5 — keyword search export |

---

## Working set vs full scrape

| Session key | Meaning |
|-------------|---------|
| `df_combined` | Full pipeline output (cached scrape) |
| `df_active` | After scrape-level and/or metadata filters |

Step 3 visualizations always use `df_active`.

---

## Related

- [pipeline/02_web_scraping.md](../pipeline/02_web_scraping.md) — building `geo_webscrap.csv`
- [visualization/README.md](../visualization/README.md) — step 3 plots on the filtered set
