"""Resolve GEO platform accessions (GPL) to human-readable names."""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from geoscouter.config import WORK_DIR
from geoscouter.utils.http import ncbi_get

logger = logging.getLogger(__name__)

GPL_RE = re.compile(r"GPL\d+")
PLATFORM_CACHE_PATH = WORK_DIR / "platform_cache.json"


def parse_gpl_ids(platforms: str | None) -> list[str]:
    if platforms is None or pd.isna(platforms) or not str(platforms).strip():
        return []
    return sorted(set(GPL_RE.findall(str(platforms))))


def _parse_soft_field(soft_text: str, field: str) -> str:
    prefix = field + " = "
    for line in soft_text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip('"')
    return ""


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


def fetch_platform_info(gpl_id: str, cache_path: Path = PLATFORM_CACHE_PATH) -> dict[str, str]:
    cache = _load_cache(cache_path)
    if gpl_id in cache:
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
        logger.warning("Failed to fetch platform %s: %s", gpl_id, exc)

    info = {
        "title": title or gpl_id,
        "technology": technology or "",
    }
    cache[gpl_id] = info
    _save_cache(cache, cache_path)
    return info


def platform_display_name(gpl_id: str, info: dict[str, str] | None = None) -> str:
    if info is None:
        info = fetch_platform_info(gpl_id)
    title = str(info.get("title") or "").strip()
    technology = str(info.get("technology") or "").strip()
    if title and title != gpl_id:
        return title
    if technology:
        return technology.replace("_", " ")
    return gpl_id


def platform_multiselect_label(gpl_id: str, cache: dict[str, dict] | None = None) -> str:
    if cache is None:
        cache = _load_cache()
    info = cache.get(gpl_id) or fetch_platform_info(gpl_id)
    name = platform_display_name(gpl_id, info)
    return name if name == gpl_id else f"{name} ({gpl_id})"


def labels_for_platforms(platforms: str | None, cache: dict[str, dict]) -> str:
    gpl_ids = parse_gpl_ids(platforms)
    if not gpl_ids:
        return ""
    return ", ".join(platform_display_name(gpl, cache.get(gpl)) for gpl in gpl_ids)


def ensure_platform_labels(
    df: pd.DataFrame,
    cache_path: Path = PLATFORM_CACHE_PATH,
) -> pd.DataFrame:
    """Add/update Platform_labels using a cached GPL lookup."""
    if df is None or df.empty or "Platforms" not in df.columns:
        return df

    df = df.copy()
    all_gpls: set[str] = set()
    for platforms in df["Platforms"].dropna().unique():
        all_gpls.update(parse_gpl_ids(platforms))

    cache = _load_cache(cache_path)
    for gpl_id in sorted(all_gpls):
        if gpl_id not in cache:
            fetch_platform_info(gpl_id, cache_path)
    cache = _load_cache(cache_path)

    series_labels = (
        df.drop_duplicates("Series")
        .set_index("Series")["Platforms"]
        .apply(lambda platforms: labels_for_platforms(platforms, cache))
    )
    df["Platform_labels"] = df["Series"].map(series_labels).fillna("")
    return df
