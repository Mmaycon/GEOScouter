# Pipeline — scrape GEO series

Two-step flow inside the app (step 1): parse `gds_result.txt`, then scrape each GSE into `geo_webscrap.csv`.

| Step | Doc | Source |
|------|-----|--------|
| 0. GEO DataSets input | [01_gds_input.md](01_gds_input.md) | [gds_parse.py](../../geoscouter/core/gds_parse.py) |
| 1. Web scraping | [02_web_scraping.md](02_web_scraping.md) | [pipeline.py](../../geoscouter/core/pipeline.py) |

---

## Outputs

All files are written under `WORK_DIR` ([`geoscouter/config.py`](../../geoscouter/config.py)) — default `/tmp/geoscouter/` on Streamlit Cloud and locally.

| File | Contents |
|------|----------|
| `gds_result.txt` | Uploaded GEO DataSets export (input) |
| `gds_processed.csv` | GSE list with proximity clusters and `Study_type` |
| `geo_webscrap.csv` | One row per supplementary file (or one metadata-only row if none found) |

---

## `geo_webscrap.csv` column contract

Columns come from scraped dict keys; not every GSE has every field.

| Column group | Examples | Source |
|--------------|----------|--------|
| Series metadata | `Title`, `Summary`, `Overall design`, contact fields | SOFT full-view (`!Series_*`) |
| Identifiers | `Platforms`, `Samples`, `Series`, `SuperSeries` | SOFT + scrape |
| From `gds_result.txt` | `Study_type`, `Assay_hint`, `Platform_labels` | [gds_parse.py](../../geoscouter/core/gds_parse.py) + [platforms.py](../../geoscouter/core/platforms.py) |
| Supplementary files | `Supplementary file`, `Size`, `File type/resource`, `Supplementary URL` | SOFT, XML custom list, tar expansion |

Downstream code expects at least `Platforms`, `Series`, `Samples`, and supplementary-file columns ([`normalize_scrape_df`](../../geoscouter/utils/summary.py)).

---

## Cache behavior

If `geo_webscrap.csv` already exists, **Run pipeline** loads the cache and **does not re-scrape**.

After code updates or to fix stale output:

1. Click **Delete all cached outputs** in the app, or
2. Remove files listed in `CACHE_FILES` from `/tmp/geoscouter/`

Then upload `gds_result.txt` again and **Run pipeline**.

---

## Directory roles

| Location | Role |
|----------|------|
| `data/input/gds_result.txt` | Example input (repo; not used automatically by the app) |
| `/tmp/geoscouter/` | Runtime working directory (upload, cache, exports) |
| `/tmp/geoscouter/metadata/` | GEOparse SOFT cache when fetching GSM metadata (step 2) |
