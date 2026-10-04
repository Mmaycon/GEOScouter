# Worked Example: End-to-End GEOScouter Run

This guide provides a reproducible end-to-end workflow so new collaborators can test GEOScouter using a fixed input and expected outputs. 

### Pinned Parameters

| Parameter | Value |
|-----------|-------|
| **Input file** | [`gds_result_skin_xenium.txt`](files/input/gds_result_skin_xenium.txt) |
| **Step 2 Filters** | Platform (GPL): `GPL33762` (Xenium Homo sapiens) AND `GPL33896` (Xenium Mus musculus) |
| **Signature Tokens** | `boundaries.parquet.gz`, `feature_matrix.h5`, `cells.parquet.gz`, `morphology.ome.tif.gz`, `transcripts.parquet.gz` |
| **Similarity Threshold** | `0.80` |

---

## Step 1 — Run data scraping

1. Go to Step 1 and upload the example search result file: `files/input/gds_result_skin_xenium.txt`. (For instructions on generating this file, see [01_gds_input.md](../pipeline/01_gds_input.md)).
2. Click **Run pipeline**. 
3. The app will scrape GEO for the 29 GSEs found in the input. Once complete, you can download the raw scrape results (e.g., `geo_webscrap.csv`).

## Step 2 — Filter datasets

Apply scrape-level filters to ensure we only compare relevant datasets.

1. Under **Platform (GPL)**, select `Xenium In Situ Analyzer: Homo sapiens (GPL33762)` and `Xenium In Situ Analyzer: Mus musculus (GPL33896)`.
2. Click **Apply scrape-level filters**. 
3. The active working set is now reduced to **26 series**[cite: 2].
*(See filter configuration: `img/2_Filter_dataset.png`)*

## Step 3 — Visualize datasets

Review the visual profiling of the filtered working set.

### Dataset Snapshot & Complexity
* **Snapshot bars:** Displays total size, number of samples, and number of file types per series.
  * ![Total Size](img/total_size_per_series.png)
  * ![Samples per Series](img/samples_per_series.png)
  * ![File Types](img/file_types_per_series.png)
* **File vs samples:** Compares sample count against unique supplementary files.
  * ![Complexity](img/unique_supplementary_files_vs_samples_pe.png)

### Supervised file-structure signature network
1. Select **Manual file types** as the signature source[cite: 4].
2. Enter the 5 signature tokens listed in the parameters table (one per line).
3. Click **Build signature from file types** to review the match patterns *(see `img/creating_signature.png`)*.
4. Click **Apply signature & show network**. Using a minimum edge similarity of `0.20`, the resulting network maps structural similarity across the datasets.

![Signature Network](img/similarity_plot.png)
*(The Comparison to signature table details which specific rules matched or missed for each GSE: `img/comparison_to_signature.png`)*

## Step 4 — Select specific GSEs

Build the final comparison list based on the applied signature.

1. Set the **Minimum signature similarity** threshold to `0.80`[cite: 6].
2. Click **Add GSEs matching signature (>= threshold)**. 
3. No manual GSEs are added for this run. 
4. The comparison list now contains **8 unique GSEs**[cite: 6] that meet the strict structural criteria.
5. Click **Export comparison list as filtered table** to download the subset.
   * **Export:** [`filtered_geo_web_scrap.csv`](files/output/filtered_geo_web_scrap.csv)

## Step 5 — Sample metadata filtering and curation

Fetch and review metadata strictly for the 8 selected GSEs.

1. Under **Sample metadata filtering**, we apply no further cell/organ/disease filters for this example.
2. The curated catalog summarizes the filtered GSEs.
   * **Export:** [`curated_gse_catalog.csv`](files/output/curated_gse_catalog.csv)
3. Use the **Keyword search across metadata** to explore specific traits. Searching for `Homo sapiens` yields 6 GSEs and 47 GSMs[cite: 7].
   *(See keyword results: `img/metadata_key_word.png`)*
4. Click **Export all fetched metadata to Excel** to generate a workbook containing full sample-level details (one sheet per GSE).
   * **Export:** [`metadata_GSE.xlsx`](files/output/metadata_GSE.xlsx)