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
]


def apply_series_filters(
    df: pd.DataFrame,
    *,
    selected_platforms: list[str] | None = None,
    min_samples: int = 0,
    max_samples: int = 0,
    gse_selection: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return row-level scrape data restricted to matching series."""
    df_base = normalize_scrape_df(df)
    series_level = df_base.drop_duplicates(subset=["Series"]).copy()
    keep_series = set(series_level["Series"].unique())

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
        for field in METADATA_FILTER_FIELDS:
            if field not in df.columns:
                continue
            for val in df[field].dropna().astype(str).unique():
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
    if not field_filters:
        return [df["gse_id"].iloc[0] for df in list_of_metadata_dfs if "gse_id" in df.columns]

    matched = []
    for df in list_of_metadata_dfs:
        gse_id = df["gse_id"].iloc[0] if "gse_id" in df.columns else None
        if gse_id is None:
            continue
        ok = True
        for field, selected_vals in field_filters.items():
            if not selected_vals or field not in df.columns:
                continue
            col_vals = df[field].astype(str).str.strip()
            if not col_vals.isin(selected_vals).any():
                ok = False
                break
        if ok:
            matched.append(gse_id)
    return matched
