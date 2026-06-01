"""Paths and cache file names used by the app."""

from pathlib import Path
import tempfile

WORK_DIR = Path(tempfile.gettempdir()) / "geoscouter"
WORK_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILES = [
    "geo_webscrap.csv",
    "gds_processed.csv",
    "filtered_geo_webscrap.csv",
    "metadata_GSE.xlsx",
    "metadata_filtered_by_word.xlsx",
    "platform_cache.json",
]

GDS_INPUT_NAME = "gds_result.txt"
