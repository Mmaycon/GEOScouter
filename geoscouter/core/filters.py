"""Filter helpers for series-level and sample-level metadata."""

import re
from typing import Iterable

import pandas as pd

from geoscouter.utils.summary import normalize_scrape_df

# GEO sample metadata fields commonly used for discovery
METADATA_FILTER_FIELDS = [
    "assay_type",
    "molecule_ch1",
    "library_source",
    "library_strategy",
    "organism_ch1",
    "source_name_ch1",
    "tissue",
    "cell type",
    "cell_type",
    "treatment_protocol_ch1",
    "growth_protocol_ch1",
    "platform_id",
    "disease",
    "disease state",
    "diagnosis",
    "condition",
    "organ",
    "organism part",
]

METADATA_FILTER_GROUPS = {
    "Cell type": ["cell type", "cell_type", "celltype"],
    "Organ / tissue": [
        "tissue",
        "organ",
        "organism part",
        "organism_part",
        "source_name_ch1",
    ],
    "Disease": ["disease", "disease state", "diagnosis", "condition"],
}


def _resolve_metadata_columns(
    df: pd.DataFrame, field_names: list[str]
) -> list[str]:
    """Return actual column names in df matching any of the field aliases."""
    lower_cols = {str(c).lower(): c for c in df.columns}
    resolved: list[str] = []
    seen: set[str] = set()
    for name in field_names:
        col = lower_cols.get(name.lower())
        if col and col not in seen:
            seen.add(col)
            resolved.append(col)
    return resolved


def metadata_group_options(
    list_of_metadata_dfs: list[pd.DataFrame],
) -> dict[str, list[str]]:
    """Collect unique values per metadata filter group across loaded GSE tables."""
    options: dict[str, set[str]] = {}
    for df in list_of_metadata_dfs:
        for group, aliases in METADATA_FILTER_GROUPS.items():
            for col in _resolve_metadata_columns(df, aliases):
                for val in df[col].dropna().astype(str).unique():
                    val = val.strip()
                    if val:
                        options.setdefault(group, set()).add(val)
    return {k: sorted(v) for k, v in options.items()}


def group_filters_to_field_filters(
    group_filters: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Expand UI group selections into alias field keys (for legacy callers)."""
    field_filters: dict[str, list[str]] = {}
    for group, selected_vals in group_filters.items():
        if not selected_vals:
            continue
        aliases = METADATA_FILTER_GROUPS.get(group, [group])
        for alias in aliases:
            field_filters[alias] = sorted(set(selected_vals))
    return field_filters


def merge_field_filters(
    group_filters: dict[str, list[str]],
    advanced_filters: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Combine grouped and advanced metadata filters for legacy field-based APIs."""
    merged = group_filters_to_field_filters(group_filters)
    for field, vals in advanced_filters.items():
        if not vals:
            continue
        merged[field] = sorted(set(merged.get(field, [])) | set(vals))
    return merged


def sample_matches_metadata_filters(
    row: pd.Series,
    group_filters: dict[str, list[str]],
    advanced_filters: dict[str, list[str]],
) -> bool:
    """True when a sample row satisfies all active group and advanced filters."""
    for group, selected_vals in group_filters.items():
        if not selected_vals:
            continue
        aliases = METADATA_FILTER_GROUPS.get(group, [])
        cols = _resolve_metadata_columns(row.to_frame().T, aliases)
        if not cols:
            return False
        matched = any(str(row.get(col, "")).strip() in selected_vals for col in cols)
        if not matched:
            return False

    for field, selected_vals in advanced_filters.items():
        if not selected_vals:
            continue
        cols = _resolve_metadata_columns(row.to_frame().T, [field])
        if not cols:
            return False
        matched = any(str(row.get(col, "")).strip() in selected_vals for col in cols)
        if not matched:
            return False
    return True


def filter_gses_by_metadata(
    list_of_metadata_dfs: list[pd.DataFrame],
    group_filters: dict[str, list[str]] | None = None,
    advanced_filters: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return GSE IDs whose samples match ALL active metadata filters."""
    group_filters = group_filters or {}
    advanced_filters = advanced_filters or {}
    if not group_filters and not advanced_filters:
        return [
            str(df["gse_id"].iloc[0]).strip().upper()
            for df in list_of_metadata_dfs
            if "gse_id" in df.columns and not df.empty
        ]

    matched: list[str] = []
    for df in list_of_metadata_dfs:
        if "gse_id" not in df.columns or df.empty:
            continue
        gse_id = str(df["gse_id"].iloc[0]).strip().upper()
        ok = True
        for group, selected_vals in group_filters.items():
            if not selected_vals:
                continue
            aliases = METADATA_FILTER_GROUPS.get(group, [])
            cols = _resolve_metadata_columns(df, aliases)
            if not cols:
                ok = False
                break
            group_match = any(
                df[col].astype(str).str.strip().isin(selected_vals).any() for col in cols
            )
            if not group_match:
                ok = False
                break
        if not ok:
            continue
        for field, selected_vals in advanced_filters.items():
            if not selected_vals:
                continue
            cols = _resolve_metadata_columns(df, [field])
            if not cols:
                ok = False
                break
            field_match = any(
                df[col].astype(str).str.strip().isin(selected_vals).any() for col in cols
            )
            if not field_match:
                ok = False
                break
        if ok:
            matched.append(gse_id)
    return matched


def apply_series_filters(
    df: pd.DataFrame,
    *,
    selected_platforms: list[str] | None = None,
    selected_technologies: list[str] | None = None,
    min_samples: int = 0,
    max_samples: int = 0,
    gse_selection: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return row-level scrape data restricted to matching series."""
    df_base = normalize_scrape_df(df)
    series_level = df_base.drop_duplicates(subset=["Series"]).copy()
    keep_series = set(series_level["Series"].unique())

    if selected_technologies:
        label_col = "Platform_labels" if "Platform_labels" in series_level.columns else "Study_type"
        if label_col in series_level.columns:
            keep_series &= set(
                series_level.loc[
                    series_level[label_col].isin(selected_technologies),
                    "Series",
                ]
            )

    if selected_platforms:
        platform_regex = "|".join(re.escape(p) for p in selected_platforms)
        keep_series &= set(
            series_level.loc[
                series_level["Platforms"].str.contains(platform_regex, na=False, regex=True),
                "Series",
            ]
        )

    if min_samples > 0:
        keep_series &= set(series_level.loc[series_level["Samples"] >= int(min_samples), "Series"])

    if max_samples > 0:
        keep_series &= set(series_level.loc[series_level["Samples"] <= int(max_samples), "Series"])

    if gse_selection:
        requested = {s.strip().upper() for s in gse_selection if s}
        keep_series &= requested

    return df_base[df_base["Series"].isin(keep_series)].copy()


def metadata_filter_options(list_of_metadata_dfs: list[pd.DataFrame]) -> dict[str, list[str]]:
    """Collect unique values per metadata field across loaded GSE tables."""
    options: dict[str, set[str]] = {}
    for df in list_of_metadata_dfs:
        lower_cols = {str(c).lower(): c for c in df.columns}
        for field in METADATA_FILTER_FIELDS:
            col = lower_cols.get(field.lower())
            if col is None:
                continue
            for val in df[col].dropna().astype(str).unique():
                val = val.strip()
                if val:
                    options.setdefault(field, set()).add(val)
    return {k: sorted(v) for k, v in options.items()}


def filter_metadata_tables(
    list_of_metadata_dfs: list[pd.DataFrame],
    field_filters: dict[str, list[str]],
) -> list[str]:
    """
    Return GSE IDs whose samples match ALL active metadata field filters (OR within field).
    """
    advanced = {k: v for k, v in field_filters.items() if v}
    return filter_gses_by_metadata(list_of_metadata_dfs, advanced_filters=advanced)
