"""Technology labels from gds_result.txt Type field (and title assay hints)."""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from geoscouter.config import GDS_INPUT_NAME, WORK_DIR
from geoscouter.core.gds_parse import parse_gds_series_metadata

logger = logging.getLogger(__name__)

GPL_RE = re.compile(r"GPL\d+")
PLATFORM_CACHE_PATH = WORK_DIR / "platform_cache.json"


def parse_gpl_ids(platforms: str | None) -> list[str]:
    if platforms is None or pd.isna(platforms) or not str(platforms).strip():
        return []
    return sorted(set(GPL_RE.findall(str(platforms))))


def build_technology_label(study_type: str, assay_hint: str = "") -> str:
    """
    Build a display label from gds_result.txt.

    Uses the Type field; when Type is only \"Other\", falls back to the assay
    tag parsed from the series title (e.g. [scRNA-Seq], (CROP-Seq, ...)).
    """
    study_type = (study_type or "").strip()
    assay_hint = (assay_hint or "").strip()
    if study_type and study_type.lower() != "other":
        return study_type
    if assay_hint:
        return assay_hint
    return study_type or "Unknown"


def apply_gds_technology_metadata(
    df: pd.DataFrame,
    gds_path: Path | str | None = None,
) -> pd.DataFrame:
    """Attach Study_type, Assay_hint, and Platform_labels from gds_result.txt."""
    if df is None or df.empty:
        return df

    path = Path(gds_path or WORK_DIR / GDS_INPUT_NAME)
    if not path.exists():
        logger.warning("gds_result.txt not found at %s; technology labels unavailable.", path)
        return df

    metadata = parse_gds_series_metadata(path.read_text(encoding="utf-8"))
    df = df.copy()

    def _meta(gse: str, field: str) -> str:
        return metadata.get(str(gse).upper(), {}).get(field, "")

    df["Study_type"] = df["Series"].astype(str).str.upper().map(lambda g: _meta(g, "study_type"))
    df["Assay_hint"] = df["Series"].astype(str).str.upper().map(lambda g: _meta(g, "assay_hint"))
    df["Platform_labels"] = df.apply(
        lambda row: build_technology_label(row["Study_type"], row["Assay_hint"]),
        axis=1,
    )
    return df


def ensure_platform_labels(
    df: pd.DataFrame,
    gds_path: Path | str | None = None,
) -> pd.DataFrame:
    """Ensure Platform_labels reflects gds_result.txt Type (no GPL name lookup)."""
    if df is None or df.empty:
        return df
    return apply_gds_technology_metadata(df, gds_path)


def technology_filter_options(df: pd.DataFrame) -> list[str]:
    """Unique technology labels for multiselect filters."""
    if df is None or df.empty:
        return []
    series = df.drop_duplicates("Series")
    labels = series.get("Platform_labels", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    labels = labels.loc[labels != ""]
    if labels.empty and "Study_type" in series.columns:
        labels = series["Study_type"].fillna("").astype(str).str.strip()
        labels = labels.loc[labels != ""]
    return sorted(labels.unique())


# Legacy helpers kept for imports elsewhere; GPL network lookup is unused by default.
def _load_cache(path: Path = PLATFORM_CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def platform_multiselect_label(
    gpl_id: str,
    cache: dict[str, dict] | None = None,
    df: pd.DataFrame | None = None,
) -> str:
    if df is not None and "Platform_labels" in df.columns:
        mask = df["Platforms"].fillna("").astype(str).str.contains(
            re.escape(gpl_id), na=False, regex=True
        )
        labels = (
            df.loc[mask, "Platform_labels"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        if labels:
            label = labels[0] if len(labels) == 1 else f"{labels[0]} (+{len(labels) - 1})"
            return f"{label} · {gpl_id}"
    return gpl_id
