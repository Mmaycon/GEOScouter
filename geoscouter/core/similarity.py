"""Supplementary-file similarity between GEO series."""

from __future__ import annotations

import math
import re
from itertools import combinations

import networkx as nx
import pandas as pd

FILE_RESOURCE_COL = "File type/resource"
SIGNATURE_NODE = "File structure signature"

_GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)
_GSE_RE = re.compile(r"GSE\d+", re.IGNORECASE)
_SRR_RE = re.compile(r"SRR\d+", re.IGNORECASE)
_SRX_RE = re.compile(r"SRX\d+", re.IGNORECASE)

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

SUPERVISED_SIMILARITY_HELP = """
**Supervised file structure signature**

Pick one or more **training GSEs** whose supplementary file layout you want to use as a template.
Each filename is converted to a **structure pattern** by replacing study-specific IDs
(`GSM…`, `GSE…`, `SRR…`, `SRX…`) with placeholders, keeping extensions and layout tokens
(`_R1`, `_R2`, `barcodes`, etc.).

A **signature pattern** is kept when it appears in at least the chosen fraction of training GSEs
(default 80%). Each pattern is weighted by how often it appears across the training set.

Every **non-training** GSE is scored with **weighted signature coverage**:

`similarity = sum(pattern weights matched) / sum(all signature pattern weights)`

Scores range from **0** (none of the taught layout) to **1** (matches the full signature).

Training GSEs define the signature but are not ranked against it — the network links them
to the central signature node to show which series you used as examples.
"""


def normalize_filename(name) -> str:
    if pd.isnull(name) or not str(name).strip():
        return ""
    return str(name).strip()


def filename_to_pattern(name) -> str:
    """Normalize a filename to a reusable structure token."""
    raw = normalize_filename(name)
    if not raw:
        return ""
    base = raw.replace("\\", "/").split("/")[-1]
    pattern = _GSM_RE.sub("{sample}", base)
    pattern = _GSE_RE.sub("{series}", pattern)
    pattern = _SRR_RE.sub("{run}", pattern)
    pattern = _SRX_RE.sub("{run}", pattern)
    return pattern.lower()


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


def series_pattern_sets(series_files: pd.Series) -> pd.Series:
    """Map each Series to the set of normalized filename structure patterns."""
    return series_files.apply(
        lambda files: {filename_to_pattern(f) for f in files if filename_to_pattern(f)}
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


def build_structure_signature(
    series_files: pd.Series,
    training_gses: list[str],
    min_support_ratio: float = 0.8,
) -> tuple[dict[str, float], set[str]]:
    """
    Build weighted pattern signature from training GSEs.

    Returns (pattern -> weight, normalized training GSE ids).
    """
    training = []
    seen: set[str] = set()
    for gse in training_gses:
        g = str(gse).strip().upper()
        if not g or g in seen or g not in series_files.index:
            continue
        seen.add(g)
        training.append(g)

    if not training:
        return {}, set()

    n = len(training)
    min_support = max(1, math.ceil(min_support_ratio * n))
    pattern_freq: dict[str, int] = {}

    for gse in training:
        for fname in series_files[gse]:
            pat = filename_to_pattern(fname)
            if pat:
                pattern_freq[pat] = pattern_freq.get(pat, 0) + 1

    signature = {
        pat: count / n
        for pat, count in pattern_freq.items()
        if count >= min_support
    }
    return signature, set(training)


def supervised_similarity_score(
    signature: dict[str, float], candidate_patterns: set[str]
) -> float:
    """Weighted coverage of signature patterns in a candidate GSE."""
    if not signature:
        return 0.0
    matched = sum(weight for pat, weight in signature.items() if pat in candidate_patterns)
    total = sum(signature.values())
    return matched / total if total else 0.0


def pattern_jaccard(a: set[str], b: set[str]) -> float:
    return _jaccard(a, b)


def calculate_supervised_edges(
    series_files: pd.Series,
    training_gses: list[str],
    min_support_ratio: float = 0.8,
) -> tuple[dict[str, float], set[str], pd.Series, list[tuple[str, str, float]]]:
    """
    Star edges from virtual signature node to non-training GSEs.

    Returns (signature, training_set, pattern_sets, edges).
    """
    signature, training = build_structure_signature(
        series_files, training_gses, min_support_ratio=min_support_ratio
    )
    pattern_sets = series_pattern_sets(series_files)
    edges: list[tuple[str, str, float]] = []

    for gse in series_files.index:
        if gse in training:
            continue
        score = supervised_similarity_score(signature, pattern_sets[gse])
        if score > 0:
            edges.append((SIGNATURE_NODE, gse, score))

    return signature, training, pattern_sets, edges


def supervised_comparison_table(
    series_files: pd.Series,
    training_gses: list[str],
    min_support_ratio: float = 0.8,
) -> pd.DataFrame:
    """Rows for each non-training GSE sorted by supervised similarity."""
    signature, training, pattern_sets, _ = calculate_supervised_edges(
        series_files, training_gses, min_support_ratio=min_support_ratio
    )
    sig_patterns = set(signature)
    rows = []

    for gse in series_files.index:
        if gse in training:
            continue
        patterns = pattern_sets[gse]
        score = supervised_similarity_score(signature, patterns)
        matched = sorted(sig_patterns & patterns)
        missing = sorted(sig_patterns - patterns)
        extra = sorted(patterns - sig_patterns)
        rows.append(
            {
                "GSE": gse,
                "Similarity to signature": score,
                "Matching patterns": ", ".join(matched) if matched else "",
                "Missing signature patterns": ", ".join(missing) if missing else "",
                "Extra patterns": ", ".join(extra) if extra else "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "GSE",
                "Similarity to signature",
                "Matching patterns",
                "Missing signature patterns",
                "Extra patterns",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Similarity to signature", ascending=False)
        .reset_index(drop=True)
    )


def signature_patterns_table(signature: dict[str, float]) -> pd.DataFrame:
    """Display signature patterns and their training-set weights."""
    if not signature:
        return pd.DataFrame(columns=["Pattern", "Weight in training set"])
    rows = [
        {"Pattern": pat, "Weight in training set": weight}
        for pat, weight in sorted(signature.items(), key=lambda x: (-x[1], x[0]))
    ]
    return pd.DataFrame(rows)
