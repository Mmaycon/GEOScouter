import pandas as pd

from geoscouter.utils.text import wrap_text_for_plotly


def normalize_scrape_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Platforms", "Series", "Samples"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["Platforms"] = (
        df["Platforms"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["Samples"] = pd.to_numeric(df["Samples"], errors="coerce").fillna(0).astype(int)
    df["Series"] = df["Series"].fillna("").astype(str).str.strip().str.upper()
    return df


def size_to_mb(size_str):
    if pd.isnull(size_str):
        return 0
    try:
        num, unit = str(size_str).strip().split()
        num = float(num)
        if "gb" in unit.lower():
            return num * 1024
        if "mb" in unit.lower():
            return num
        if "kb" in unit.lower():
            return num / 1024
        return num / (1024 * 1024)
    except Exception:
        return 0


def platform_group(p_str):
    if pd.isnull(p_str):
        return "Unknown"
    return "Multiple Platforms" if "," in str(p_str) else str(p_str)


def build_series_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scrape rows to one row per GSE series."""
    df = df.copy()
    df["Size_MB"] = df.get("Size", pd.Series([None] * len(df))).apply(size_to_mb)
    df["Filetype_unzipped"] = (
        df.get("File type/resource", pd.Series([""] * len(df)))
        .astype(str)
        .str.replace(".gz", "", regex=False)
    )
    summary = (
        df.groupby("Series")
        .agg(
            total_size_mb=("Size_MB", "sum"),
            num_samples=("Samples", "first"),
            num_file_types=("Filetype_unzipped", "nunique"),
            num_files=("Supplementary file", "nunique"),
            Platforms=("Platforms", "first"),
            Title=("Title", "first"),
        )
        .reset_index()
    )
    summary["Platform_Group"] = summary["Platforms"].apply(platform_group)
    summary["Platform_Label"] = summary["Platform_Group"]
    summary["Hover_Title"] = summary["Title"].apply(wrap_text_for_plotly)
    return summary
