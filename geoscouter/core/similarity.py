"""Supplementary-file similarity between GEO series."""

from itertools import combinations

import networkx as nx
import pandas as pd

FILE_RESOURCE_COL = "File type/resource"

SIMILARITY_HELP = """
**What is the similarity score?**

For each GSE, we collect every **full supplementary filename** (including extension)
from the `File type/resource` column.

Two series are compared with **Jaccard similarity** on those filename sets:

`similarity = |filenames in common| / |all unique filenames across both series|`

Scores range from **0** (no overlap) to **1** (identical supplementary file lists).

**How to interpret thresholds**

| Score | Typical meaning |
|-------|-----------------|
| 0.9+  | Very similar supplementary file layouts (often related studies or reprocessed data) |
| 0.7–0.9 | Substantial overlap; worth comparing side by side |
| 0.5–0.7 | Partial overlap; exploratory link |
| < 0.5 | Weak overlap; may still share biology but different file packaging |

Small changes (e.g. 0.80 vs 0.85) usually mean a few extra or missing files — use the network
plot and hover tooltips to inspect which series cluster together.

**Reference mode**

Optionally pick one GSE as a **reference layout**. The network still shows **general pairwise**
similarity (light gray edges, same Jaccard metric). **Reference edges** connect only the
reference GSE to every other series (star topology), drawn on top in a distinct style so you
can compare each series' supplementary filenames against your chosen favorite layout.
"""


def normalize_filename(name) -> str:
    if pd.isnull(name) or not str(name).strip():
        return ""
    return str(name).strip()


def _filename_column(df: pd.DataFrame) -> str:
    if FILE_RESOURCE_COL in df.columns:
        return FILE_RESOURCE_COL
    return "Supplementary file"


def series_filename_sets(df: pd.DataFrame) -> pd.Series:
    """Map each Series to the set of normalized supplementary filenames."""
    col = _filename_column(df)
    return df.groupby("Series")[col].apply(
        lambda files: {normalize_filename(f) for f in files if normalize_filename(f)}
    )


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def calculate_similarity_edges(df: pd.DataFrame):
    series_files = series_filename_sets(df)
    edges = [
        (s1, s2, _jaccard(series_files[s1], series_files[s2]))
        for s1, s2 in combinations(series_files.index, 2)
        if len(series_files[s1] | series_files[s2]) > 0
    ]
    graph = nx.Graph()
    graph.add_nodes_from(series_files.index)
    graph.add_weighted_edges_from(e for e in edges if e[2] > 0)
    return series_files, edges, graph


def calculate_reference_edges(
    series_files: pd.Series, reference_gse: str
) -> list[tuple[str, str, float]]:
    """Star edges from reference GSE to every other series (Jaccard on filename sets)."""
    ref = reference_gse.strip().upper()
    if ref not in series_files.index:
        return []
    ref_set = series_files[ref]
    return [
        (ref, other, _jaccard(ref_set, series_files[other]))
        for other in series_files.index
        if other != ref
    ]


def reference_vs_gse_filenames(
    series_files: pd.Series, reference_gse: str, other_gse: str
) -> tuple[float, list[str], list[str], list[str]]:
    """Jaccard score and sorted shared / reference-only / other-only filenames."""
    ref = reference_gse.strip().upper()
    other = other_gse.strip().upper()
    ref_set = series_files.get(ref, set())
    other_set = series_files.get(other, set())
    shared = sorted(ref_set & other_set)
    only_ref = sorted(ref_set - other_set)
    only_other = sorted(other_set - ref_set)
    return _jaccard(ref_set, other_set), shared, only_ref, only_other


def reference_comparison_table(
    series_files: pd.Series, reference_gse: str
) -> pd.DataFrame:
    """Rows for each non-reference GSE, sorted by similarity descending."""
    ref = reference_gse.strip().upper()
    rows = []
    for other in series_files.index:
        if other == ref:
            continue
        score, shared, only_ref, only_other = reference_vs_gse_filenames(
            series_files, ref, other
        )
        rows.append(
            {
                "GSE": other,
                "Similarity to reference": score,
                "Shared files": ", ".join(shared) if shared else "",
                "Only in reference": ", ".join(only_ref) if only_ref else "",
                "Only in GSE": ", ".join(only_other) if only_other else "",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "GSE",
                "Similarity to reference",
                "Shared files",
                "Only in reference",
                "Only in GSE",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Similarity to reference", ascending=False)
        .reset_index(drop=True)
    )
