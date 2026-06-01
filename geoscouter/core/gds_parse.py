"""Parse GEO DataSets export (gds_result.txt) into scrape targets."""

import re

import pandas as pd

# Block split: numbered entries ("1. Title", "2. Title", ...)
_ENTRY_SPLIT = re.compile(r"\n(?=\d+\.\s)")
_ACCESSION_RE = re.compile(r"Series\s+Accession:\s+(GSE\d+)", re.IGNORECASE)
_TYPE_RE = re.compile(r"^Type:\s*(.+)$", re.MULTILINE)
_ORGANISM_RE = re.compile(r"^Organism:\s*(.+)$", re.MULTILINE)
_SUPER_COMPOSED_RE = re.compile(
    r"This\s+SuperSeries\s+is\s+composed",
    re.IGNORECASE,
)
# SuperSeries entries often use "Platforms:" (plural); subseries use "Platform:"
_PLATFORMS_PLURAL_RE = re.compile(r"^Platforms:\s", re.MULTILINE | re.IGNORECASE)


def _is_superseries_block(block: str) -> bool:
    if _SUPER_COMPOSED_RE.search(block):
        return True
    if _PLATFORMS_PLURAL_RE.search(block):
        return True
    first_line = block.split("\n", 1)[0]
    if re.search(r"\bSuperSeries\b", first_line, re.IGNORECASE):
        return True
    return False


def parse_gds_result(text: str) -> tuple[list[str], list[str]]:
    """
    Parse gds_result.txt into GSE accessions to scrape.

    Returns
    -------
    scrape_gses : list[str]
        Subseries and standalone series (SuperSeries parents excluded).
    excluded_superseries : list[str]
        Parent SuperSeries accessions that were dropped.
    """
    blocks = _ENTRY_SPLIT.split(text.strip())
    scrape_gses: list[str] = []
    excluded: list[str] = []

    for block in blocks:
        if not block.strip():
            continue
        match = _ACCESSION_RE.search(block)
        if not match:
            continue
        gse = match.group(1).upper()
        if _is_superseries_block(block):
            excluded.append(gse)
        else:
            scrape_gses.append(gse)

    # Stable order, first occurrence in file
    seen: set[str] = set()
    ordered_scrape: list[str] = []
    for gse in scrape_gses:
        if gse not in seen:
            seen.add(gse)
            ordered_scrape.append(gse)

    excluded_unique = list(dict.fromkeys(excluded))
    return ordered_scrape, excluded_unique


def _parse_assay_hint(block: str) -> str:
    """Extract assay tag from the series title, e.g. [scRNA-Seq] or (CROP-Seq, ...)."""
    first_line = block.split("\n", 1)[0]
    first_line = re.sub(r"^\d+\.\s*", "", first_line).strip()
    bracket = re.search(r"\[([^\]]+)\]", first_line)
    if bracket:
        return bracket.group(1).strip()
    paren = re.search(r"\(([^)]+)\)\s*$", first_line)
    if paren:
        return paren.group(1).strip()
    return ""


def parse_gds_series_metadata(text: str) -> dict[str, dict[str, str]]:
    """Return per-GSE fields parsed from gds_result.txt (Type, Organism, assay hint)."""
    blocks = _ENTRY_SPLIT.split(text.strip())
    metadata: dict[str, dict[str, str]] = {}

    for block in blocks:
        if not block.strip():
            continue
        match = _ACCESSION_RE.search(block)
        if not match:
            continue
        gse = match.group(1).upper()
        type_match = _TYPE_RE.search(block)
        organism_match = _ORGANISM_RE.search(block)
        metadata[gse] = {
            "study_type": type_match.group(1).strip() if type_match else "",
            "organism": organism_match.group(1).strip() if organism_match else "",
            "assay_hint": _parse_assay_hint(block),
        }

    return metadata


def build_gds_processed_df(
    scrape_gses: list[str],
    proximity_window: int = 10,
    series_metadata: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Cluster numeric GSE IDs by proximity (legacy behavior on filtered list)."""
    gse_nums = sorted(
        {int(re.search(r"GSE(\d+)", g, re.I).group(1)) for g in scrape_gses if re.search(r"GSE(\d+)", g, re.I)}
    )

    clusters: list[tuple[int, list[int]]] = []
    cluster_index = 0
    current_cluster: list[int] = []
    prev_num = None

    for gse_num in gse_nums:
        if prev_num is None:
            current_cluster = [gse_num]
            cluster_index = 1
        elif gse_num - prev_num <= proximity_window:
            current_cluster.append(gse_num)
        else:
            clusters.append((cluster_index, current_cluster))
            cluster_index += 1
            current_cluster = [gse_num]
        prev_num = gse_num

    if current_cluster:
        clusters.append((cluster_index, current_cluster))

    gse_to_cluster: dict[int, int] = {}
    for cluster_id, gse_list in clusters:
        for val in gse_list:
            gse_to_cluster[val] = cluster_id

    rows = []
    for gse in scrape_gses:
        m = re.search(r"GSE(\d+)", gse, re.I)
        if not m:
            continue
        num = int(m.group(1))
        rows.append({
            "GSE": f"GSE{m.group(1)}",
            "Cluster": f"Cluster{gse_to_cluster.get(num, 1)}",
            "Study_type": (series_metadata or {}).get(gse, {}).get("study_type", ""),
        })

    return pd.DataFrame(rows).drop_duplicates(subset=["GSE"]).reset_index(drop=True)
