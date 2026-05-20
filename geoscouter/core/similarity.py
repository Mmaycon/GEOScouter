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
"""


def normalize_filename(name) -> str:
    if pd.isnull(name) or not str(name).strip():
        return ""
    return str(name).strip()


def _filename_column(df: pd.DataFrame) -> str:
    if FILE_RESOURCE_COL in df.columns:
        return FILE_RESOURCE_COL
    return "Supplementary file"


def calculate_similarity_edges(df: pd.DataFrame):
    col = _filename_column(df)
    series_files = df.groupby("Series")[col].apply(
        lambda files: {normalize_filename(f) for f in files if normalize_filename(f)}
    )
    edges = [
        (
            s1,
            s2,
            len(series_files[s1] & series_files[s2]) / len(series_files[s1] | series_files[s2]),
        )
        for s1, s2 in combinations(series_files.index, 2)
        if len(series_files[s1] | series_files[s2]) > 0
    ]
    graph = nx.Graph()
    graph.add_nodes_from(series_files.index)
    graph.add_weighted_edges_from(e for e in edges if e[2] > 0)
    return series_files, edges, graph
