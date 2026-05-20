"""Supplementary-file similarity between GEO series."""

import re
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

SIMILARITY_HELP = """
**What is the similarity score?**

For each GSE, we collect a *token* from every supplementary filename. The token is the
meaningful part of the filename **after** the `GSE` accession (not just the file extension).

Two series are compared with **Jaccard similarity** on those token sets:

`similarity = |tokens in common| / |all unique tokens across both series|`

Scores range from **0** (no overlap) to **1** (identical file naming patterns).

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


def supplementary_file_token(filename: str) -> str:
    """
    Extract a comparable token from a supplementary filename.

    Prefer the substring after GSE\\d+ so extensions alone do not dominate.
    """
    if pd.isnull(filename) or not str(filename).strip():
        return ""
    name = str(filename).strip()
    match = re.search(r"GSE\d+[_\-.]?(.*)", name, flags=re.IGNORECASE)
    if match and match.group(1).strip():
        token = match.group(1).strip()
    else:
        token = Path(name).stem
    token = re.sub(r"\.(tar|gz|zip|bz2)+$", "", token, flags=re.IGNORECASE)
    token = token.strip("._- ")
    return token.lower() if token else Path(name).suffix.lower()


def calculate_similarity_edges(df: pd.DataFrame):
    series_files = df.groupby("Series")["Supplementary file"].apply(
        lambda files: {supplementary_file_token(f) for f in files if supplementary_file_token(f)}
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
