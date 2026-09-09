# Filtering — step 2

Narrow the scraped dataset before visualizations and exports. Filters apply to the **working set** (`df_active`), not the raw cache (`df_combined`).

**Source:** [streamlit_app.py](../../streamlit_app.py) (section 2), [filters.py](../../geoscouter/core/filters.py), [platforms.py](../../geoscouter/core/platforms.py)

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

Step 2 is **scrape-only**. GSM sample metadata fetch and filtering moved to [step 5](../metadata/README.md).

---

## Outputs

| File | When written |
|------|--------------|
| `filtered_geo_webscrap.csv` | Exported from step 4 when saving the comparison / filtered set |
| `curated_gse_catalog.csv` | Step 5 — curated GSE catalog after metadata filtering and health curation |

---

## Working set vs full scrape

| Session key | Meaning |
|-------------|---------|
| `df_combined` | Full pipeline output (cached scrape) |
| `df_active` | After scrape-level filters |

Step 3 visualizations always use `df_active`.

---

## Related

- [pipeline/02_web_scraping.md](../pipeline/02_web_scraping.md) — building `geo_webscrap.csv`
- [metadata/README.md](../metadata/README.md) — step 5 sample metadata curation
- [visualization/README.md](../visualization/README.md) — step 3 plots on the filtered set
