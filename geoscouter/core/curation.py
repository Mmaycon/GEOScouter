"""GSE curation: platform labels, panel size, health status, export tables."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from geoscouter.core.filters import METADATA_FILTER_GROUPS
from geoscouter.core.similarity import geo_series_url
from geoscouter.utils.summary import normalize_scrape_df

HealthStatus = Literal["healthy", "cancer", "other"]

PLATFORM_ALIASES: list[tuple[str, str]] = [
    (r"xenium", "Xenium"),
    (r"cosmx|cos-mx|cos mx", "CosMx"),
    (r"visium", "Visium"),
    (r"merfish", "MERFISH"),
    (r"seqfish", "seqFISH"),
    (r"merscope", "MERSCOPE"),
    (r"slide[\-\s]?seq", "Slide-seq"),
    (r"scrnaseq|scrna[\-\s]?seq|single[\-\s]?cell rna", "scRNA-seq"),
    (r"snrnaseq|snrna[\-\s]?seq", "snRNA-seq"),
    (r"spatial transcript", "Spatial transcriptomics"),
]

CANCER_KEYWORDS = (
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "malignant",
    "metastatic",
    "neoplasm",
    "adenocarcinoma",
    "sarcoma",
    "lymphoma",
    "leukemia",
    "leukaemia",
    "melanoma",
    "oncolog",
)

HEALTHY_KEYWORDS = (
    "healthy",
    "normal",
    "control",
    "non-tumor",
    "non tumour",
    "nontumor",
    "non tumor",
    "benign",
    "adjacent normal",
    "non-cancer",
    "non cancer",
    "noncancer",
)

METADATA_GROUP_FIELDS = METADATA_FILTER_GROUPS

_PANEL_TITLE_RE = re.compile(
    r"(\d+\s*k|\d+\s*mer|human\s+\d+|mouse\s+\d+|panel\s*[\w\-]+)",
    re.IGNORECASE,
)


def unified_platform_name(
    platform_title: str,
    platform_technology: str = "",
    platforms: str = "",
) -> str:
    """Map GPL titles/technologies to short labels (Xenium, CosMx, etc.)."""
    combined = " ".join(
        str(x).strip()
        for x in (platform_title, platform_technology, platforms)
        if x and str(x).strip()
    ).lower()
    if not combined:
        return "Unknown"
    for pattern, label in PLATFORM_ALIASES:
        if re.search(pattern, combined, re.IGNORECASE):
            return label
    title = str(platform_title or "").strip()
    if title and not re.fullmatch(r"GPL\d+", title, re.IGNORECASE):
        return title
    tech = str(platform_technology or "").strip()
    if tech and tech.lower() not in {"other", "unknown"}:
        return tech
    return platforms.split(",")[0].strip() if platforms else "Unknown"


def _metadata_text_blob(metadata_df: pd.DataFrame | None) -> str:
    if metadata_df is None or metadata_df.empty:
        return ""
    parts: list[str] = []
    for col in metadata_df.columns:
        if col in {"gse_id", "gsm_id"}:
            continue
        parts.extend(metadata_df[col].dropna().astype(str).tolist())
    return " ".join(parts)


def _column_values_matching(metadata_df: pd.DataFrame, field_names: list[str]) -> set[str]:
    values: set[str] = set()
    if metadata_df is None or metadata_df.empty:
        return values
    lower_cols = {str(c).lower(): c for c in metadata_df.columns}
    for name in field_names:
        col = lower_cols.get(name.lower())
        if col is None:
            continue
        for val in metadata_df[col].dropna().astype(str):
            val = val.strip()
            if val:
                values.add(val)
    return values


def extract_panel_size(
    series_row: pd.Series,
    metadata_df: pd.DataFrame | None = None,
) -> str:
    """Best-effort panel size from GSM metadata or series title."""
    if metadata_df is not None and not metadata_df.empty:
        for col in metadata_df.columns:
            if "panel" not in str(col).lower():
                continue
            for val in metadata_df[col].dropna().astype(str):
                val = val.strip()
                if val:
                    return val
        for _, row in metadata_df.iterrows():
            for col, val in row.items():
                if pd.isna(val):
                    continue
                text = str(val)
                if "panel" in text.lower():
                    if ":" in text:
                        parts = text.split(":", 1)
                        if len(parts) == 2 and parts[1].strip():
                            return parts[1].strip()
                    return text.strip()

    title = str(series_row.get("Title", "") or "")
    match = _PANEL_TITLE_RE.search(title)
    if match:
        return match.group(1).strip()
    return ""


def classify_health_status(text: str) -> HealthStatus:
    """Keyword rules on lowercase text; cancer beats healthy."""
    blob = str(text or "").lower()
    if not blob.strip():
        return "other"
    for kw in CANCER_KEYWORDS:
        if kw in blob:
            return "cancer"
    for kw in HEALTHY_KEYWORDS:
        if kw in blob:
            return "healthy"
    return "other"


def gse_health_status(
    series_row: pd.Series,
    metadata_df: pd.DataFrame | None = None,
    override: str | None = None,
) -> tuple[HealthStatus, str]:
    """Return (status, source) where source is auto or manual."""
    if override and str(override).strip().lower() in {"healthy", "cancer", "other"}:
        return str(override).strip().lower(), "manual"  # type: ignore[return-value]

    title = str(series_row.get("Title", "") or "")
    meta_text = _metadata_text_blob(metadata_df)
    status = classify_health_status(f"{title} {meta_text}")
    return status, "auto"


def aggregate_metadata_summary(
    metadata_df: pd.DataFrame | None,
    field_names: list[str],
    limit: int = 5,
) -> str:
    values = sorted(_column_values_matching(metadata_df, field_names))
    if not values:
        return ""
    if len(values) <= limit:
        return "; ".join(values)
    return "; ".join(values[:limit]) + f" (+{len(values) - limit} more)"


def count_matching_samples(
    metadata_df: pd.DataFrame | None,
    group_filters: dict[str, list[str]] | None = None,
    advanced_filters: dict[str, list[str]] | None = None,
) -> int:
    """Count samples matching all active metadata filters."""
    from geoscouter.core.filters import sample_matches_metadata_filters

    if metadata_df is None or metadata_df.empty:
        return 0
    if not group_filters and not advanced_filters:
        return len(metadata_df)
    return int(
        metadata_df.apply(
            lambda row: sample_matches_metadata_filters(
                row, group_filters or {}, advanced_filters or {}
            ),
            axis=1,
        ).sum()
    )


def build_curated_gse_table(
    df_active: pd.DataFrame,
    list_of_metadata_dfs: list[pd.DataFrame],
    gse_selection: list[str],
    health_overrides: dict[str, str] | None = None,
    metadata_matched_gses: list[str] | None = None,
    group_filters: dict[str, list[str]] | None = None,
    advanced_filters: dict[str, list[str]] | None = None,
    health_filter: list[str] | None = None,
) -> pd.DataFrame:
    """One row per GSE with curation columns for export."""
    df_active = normalize_scrape_df(df_active)
    series_level = df_active.drop_duplicates(subset=["Series"]).set_index("Series")
    health_overrides = health_overrides or {}

    requested = {g.strip().upper() for g in gse_selection if g}
    if requested:
        gse_ids = sorted(requested & set(series_level.index))
    else:
        gse_ids = sorted(series_level.index)

    if metadata_matched_gses is not None:
        matched_set = {g.strip().upper() for g in metadata_matched_gses}
        gse_ids = [g for g in gse_ids if g in matched_set]

    meta_by_gse = {
        str(df["gse_id"].iloc[0]).strip().upper(): df
        for df in list_of_metadata_dfs
        if "gse_id" in df.columns and not df.empty
    }

    rows = []
    for gse in gse_ids:
        if gse not in series_level.index:
            continue
        row = series_level.loc[gse]
        meta_df = meta_by_gse.get(gse)
        override = health_overrides.get(gse)
        status, source = gse_health_status(row, meta_df, override=override)

        if health_filter and status not in health_filter:
            continue

        rows.append(
            {
                "GSE": gse,
                "GEO link": geo_series_url(gse),
                "Title": row.get("Title", ""),
                "Unified_platform": unified_platform_name(
                    str(row.get("Platform_title", "") or ""),
                    str(row.get("Platform_technology", "") or ""),
                    str(row.get("Platforms", "") or ""),
                ),
                "Platform_title": row.get("Platform_title", ""),
                "Panel_size": extract_panel_size(row, meta_df),
                "num_samples": row.get("Samples", 0),
                "Health_status": status,
                "Health_status_source": source,
                "Matching_samples": count_matching_samples(
                    meta_df, group_filters, advanced_filters
                ),
                "Cell_types_found": aggregate_metadata_summary(
                    meta_df, METADATA_GROUP_FIELDS["Cell type"]
                ),
                "Organs_found": aggregate_metadata_summary(
                    meta_df, METADATA_GROUP_FIELDS["Organ / tissue"]
                ),
                "Diseases_found": aggregate_metadata_summary(
                    meta_df, METADATA_GROUP_FIELDS["Disease"]
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "GSE",
                "GEO link",
                "Title",
                "Unified_platform",
                "Platform_title",
                "Panel_size",
                "num_samples",
                "Health_status",
                "Health_status_source",
                "Matching_samples",
                "Cell_types_found",
                "Organs_found",
                "Diseases_found",
            ]
        )
    return pd.DataFrame(rows)
