"""Technology labels from gds_result.txt and GPL metadata from GEO platforms."""

import json
import logging
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from geoscouter.config import GDS_INPUT_NAME, WORK_DIR
from geoscouter.core.gds_parse import parse_gds_series_metadata
from geoscouter.utils.http import ncbi_get

logger = logging.getLogger(__name__)

GPL_RE = re.compile(r"GPL\d+")
PLATFORM_CACHE_PATH = WORK_DIR / "platform_cache.json"


def _is_generic_other(value: str) -> bool:
    return (value or "").strip().lower() == "other"


def resolve_gpl_technology(info: dict[str, str], gpl_id: str) -> str:
    """GEO Technology column, falling back to Title when technology is \"other\"."""
    title = str(info.get("title") or gpl_id).strip()
    technology = str(info.get("technology") or "").replace("_", " ").strip()
    if _is_generic_other(technology) or not technology:
        return title if title != gpl_id else gpl_id
    return technology


def parse_gpl_ids(platforms: str | None) -> list[str]:
    if platforms is None or pd.isna(platforms) or not str(platforms).strip():
        return []
    return sorted(set(GPL_RE.findall(str(platforms))))


def build_technology_label(study_type: str, assay_hint: str = "") -> str:
    """
    Build a series-level assay label from gds_result.txt.

    Uses the Type field; when Type is only \"Other\", falls back to the assay
    tag parsed from the series title (e.g. [scRNA-Seq], (CROP-Seq, ...)).
    """
    study_type = (study_type or "").strip()
    assay_hint = (assay_hint or "").strip()
    if study_type and study_type.lower() != "other":
        return study_type
    if assay_hint:
        return assay_hint
    return "Unknown"


def _parse_soft_field(soft_text: str, field: str) -> str:
    prefix = field + " = "
    for line in soft_text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip('"')
    return ""


def _parse_platform_html(gpl_id: str) -> dict[str, str]:
    """Fallback parser for GEO platform page Title and Technology fields."""
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl_id}"
    response = ncbi_get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    fields: dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 2:
            continue
        label = cells[0].get_text(" ", strip=True)
        value = cells[1].get_text(" ", strip=True)
        if label in {"Title", "Technology"}:
            fields[label.lower()] = value
    return fields


def _load_cache(path: Path = PLATFORM_CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read platform cache %s: %s", path, exc)
        return {}


def _save_cache(cache: dict[str, dict], path: Path = PLATFORM_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _cache_needs_refresh(gpl_id: str, info: dict[str, str]) -> bool:
    title = str(info.get("title") or "").strip()
    if not title or title.upper() == gpl_id.upper():
        return True
    return False


def fetch_platform_info(
    gpl_id: str,
    cache_path: Path = PLATFORM_CACHE_PATH,
    *,
    force: bool = False,
) -> dict[str, str]:
    """
    Fetch GEO platform Accession metadata (Title + Technology columns).

    Same fields shown in the GEO platform browser:
    https://www.ncbi.nlm.nih.gov/geo/browse/?view=platforms
    """
    cache = _load_cache(cache_path)
    if not force and gpl_id in cache and not _cache_needs_refresh(gpl_id, cache[gpl_id]):
        return cache[gpl_id]

    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl_id}&format=soft"
    title, technology = "", ""
    try:
        response = ncbi_get(url, timeout=30)
        response.raise_for_status()
        soft_text = response.text
        title = _parse_soft_field(soft_text, "!Platform_title")
        technology = _parse_soft_field(soft_text, "!Platform_technology")
    except Exception as exc:
        logger.warning("SOFT fetch failed for platform %s: %s", gpl_id, exc)

    if not title or title.upper() == gpl_id.upper():
        try:
            html_fields = _parse_platform_html(gpl_id)
            title = html_fields.get("title", title)
            technology = html_fields.get("technology", technology)
        except Exception as exc:
            logger.warning("HTML fetch failed for platform %s: %s", gpl_id, exc)
            if gpl_id in cache and not _cache_needs_refresh(gpl_id, cache[gpl_id]):
                return cache[gpl_id]

    info = {
        "title": title or gpl_id,
        "technology": (technology or "").replace("_", " "),
    }
    cache[gpl_id] = info
    _save_cache(cache, cache_path)
    return info


def warm_gpl_cache(gpl_ids: set[str], cache_path: Path = PLATFORM_CACHE_PATH) -> dict[str, dict]:
    cache = _load_cache(cache_path)
    for gpl_id in sorted(gpl_ids):
        if gpl_id not in cache or _cache_needs_refresh(gpl_id, cache[gpl_id]):
            fetch_platform_info(gpl_id, cache_path, force=True)
    return _load_cache(cache_path)


def gpl_platform_title(info: dict[str, str], gpl_id: str) -> str:
    """Primary platform label: always the GEO Title column."""
    title = str(info.get("title") or gpl_id).strip()
    return title if title else gpl_id


def platform_filter_label(gpl_id: str, cache: dict[str, dict] | None = None) -> str:
    """Label for GPL multiselect: Title (GPL...), never bare \"other\"."""
    if cache is None:
        cache = _load_cache()
    if gpl_id not in cache or _cache_needs_refresh(gpl_id, cache.get(gpl_id, {})):
        cache = warm_gpl_cache({gpl_id})
    info = cache.get(gpl_id, {})
    title = gpl_platform_title(info, gpl_id)
    if title.upper() == gpl_id.upper():
        return gpl_id
    return f"{title} ({gpl_id})"


def _join_gpl_field(platforms: str | None, cache: dict[str, dict], field: str) -> str:
    gpl_ids = parse_gpl_ids(platforms)
    if not gpl_ids:
        return ""
    values = []
    for gpl_id in gpl_ids:
        info = cache.get(gpl_id, {})
        if field == "title":
            val = gpl_platform_title(info, gpl_id)
        elif field == "technology":
            val = resolve_gpl_technology(info, gpl_id)
        else:
            val = str(info.get(field) or "").strip()
        values.append(val)
    return ", ".join(values)


def enrich_gpl_metadata(
    df: pd.DataFrame,
    cache_path: Path = PLATFORM_CACHE_PATH,
) -> pd.DataFrame:
    """Add Platform_title and Platform_technology from GEO GPL records."""
    if df is None or df.empty or "Platforms" not in df.columns:
        return df

    all_gpls = {
        gpl
        for platforms in df["Platforms"].dropna().unique()
        for gpl in parse_gpl_ids(platforms)
    }
    cache = warm_gpl_cache(all_gpls, cache_path)

    series_df = df.drop_duplicates("Series").set_index("Series")
    title_map = {
        series: _join_gpl_field(series_df.at[series, "Platforms"], cache, "title")
        for series in series_df.index
    }
    tech_map = {
        series: _join_gpl_field(series_df.at[series, "Platforms"], cache, "technology")
        for series in series_df.index
    }

    df = df.copy()
    df["Platform_title"] = df["Series"].map(title_map).fillna("")
    df["Platform_technology"] = df["Series"].map(tech_map).fillna("")
    return df


def _apply_platform_title_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """When gds Type is uninformative, use the GEO platform Title instead."""
    if "Platform_title" not in df.columns or "Platform_labels" not in df.columns:
        return df
    df = df.copy()
    labels = df["Platform_labels"].fillna("").astype(str).str.strip()
    titles = df["Platform_title"].fillna("").astype(str).str.strip()
    uninformative = labels.str.lower().isin(["", "unknown", "other"])
    has_title = titles.ne("") & ~titles.str.upper().str.match(r"^GPL\d+$", na=False)
    df.loc[uninformative & has_title, "Platform_labels"] = titles[uninformative & has_title]
    return df


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
    """Apply gds_result Type labels and GEO GPL Title/Technology metadata."""
    if df is None or df.empty:
        return df
    df = apply_gds_technology_metadata(df, gds_path)
    df = enrich_gpl_metadata(df)
    return _apply_platform_title_fallback(df)


def technology_filter_options(df: pd.DataFrame) -> list[str]:
    """Unique series assay labels (from gds_result Type), excluding bare \"Other\"."""
    if df is None or df.empty:
        return []
    series = df.drop_duplicates("Series")
    labels = series.get("Platform_labels", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    labels = labels.loc[(labels != "") & ~labels.str.lower().eq("other")]
    return sorted(labels.unique())


def gpl_filter_options(df: pd.DataFrame) -> list[str]:
    """Unique GPL accessions present in the working set."""
    if df is None or df.empty:
        return []
    gpls = {
        gpl
        for platforms in df["Platforms"].dropna()
        for gpl in parse_gpl_ids(platforms)
    }
    warm_gpl_cache(gpls)
    return sorted(gpls)
