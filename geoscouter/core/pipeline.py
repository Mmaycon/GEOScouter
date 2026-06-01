"""GEO web scraping pipeline (Streamlit Cloud compatible — no Selenium required)."""

import logging
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st
from geoscouter.utils.http import ncbi_get
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

SELENIUM_AVAILABLE = True
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
except Exception:
    SELENIUM_AVAILABLE = False


def get_headless_driver():
    if not SELENIUM_AVAILABLE:
        raise RuntimeError("Selenium is not available in this environment.")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2}
    )
    return webdriver.Chrome(options=chrome_options)


_ARCHIVE_BUNDLE_RE = re.compile(r"_RAW\.tar$|RAW\.tar$", re.IGNORECASE)
_CUSTOM_XML_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?format=xml&acc={gse_id}"


def is_archive_bundle(filename: str) -> bool:
    """True for bundled archives like GSE12345_RAW.tar where per-file list needs (custom)."""
    if not filename or pd.isna(filename):
        return False
    return bool(_ARCHIVE_BUNDLE_RE.search(str(filename).strip()))


def bytes_to_geo_size(size_bytes: str) -> str:
    try:
        n = int(size_bytes)
    except (TypeError, ValueError):
        return ""
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} Gb"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} Mb"
    if n >= 1024:
        return f"{n / 1024:.1f} Kb"
    return f"{n} b"


def _supp_row(data: dict, file_name: str, size: str = "") -> dict:
    row = data.copy()
    row.update({
        "Supplementary file": file_name,
        "Size": size,
        "File type/resource": file_name,
    })
    return row


def fetch_custom_supp_files_xml(gse_id: str, data: dict) -> list[dict]:
    """
    Fetch per-file supplementary listing from GEO's (custom) download XML.

    Same data shown after clicking (custom) on the series page:
    GET /geo/download/?format=xml&acc=GSE...
    """
    url = _CUSTOM_XML_URL.format(gse_id=gse_id)
    try:
        response = ncbi_get(url, timeout=45)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as e:
        logger.warning("Custom XML file list failed for %s: %s", gse_id, e)
        return []

    rows = []
    for file_el in root.findall("file"):
        name = (file_el.text or "").strip()
        if not name:
            continue
        size = bytes_to_geo_size(file_el.get("size", ""))
        rows.append(_supp_row(data, name, size))
    return rows


def has_custom_download_link(soup: BeautifulSoup) -> bool:
    return soup.find(id="customDl") is not None or soup.find("a", string="(custom)") is not None


def parse_custom_supp_files(custom_href: str, base_url: str, data: dict):
    custom_url = urljoin(base_url, custom_href)
    r = ncbi_get(custom_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    supp_data = []

    for row in soup.select("table tr"):
        cb = row.select_one("input[type='checkbox']")
        if not cb:
            continue
        tds = row.find_all("td")
        if len(tds) < 3:
            continue
        filename_td = tds[1]
        a = filename_td.find("a")
        file_name = a.get_text(strip=True) if a else filename_td.get_text(strip=True)
        if not file_name or file_name.lower() == "(all files)":
            continue
        size = tds[2].get_text(strip=True)
        supp_data.append(_supp_row(data, file_name, size))
    return supp_data


def supplementary_files_from_soft(soft_text: str):
    files = []
    for line in soft_text.splitlines():
        if line.startswith("!Series_supplementary_file"):
            parts = line.split("=", 1)
            if len(parts) != 2:
                continue
            url = parts[1].strip()
            if not url:
                continue
            filename = url.rstrip("/").split("/")[-1]
            if filename:
                files.append((filename, url))
    return files


def process_gse(gse_id, driver=None, super_series=None):
    if not super_series:
        super_series = gse_id

    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
    response = ncbi_get(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    desired_fields = [
        "Title", "Summary", "Overall design", "Contact name", "E-mail(s)",
        "Phone", "Organization name", "Department", "Lab", "City",
        "State/province", "Country",
    ]

    data = {}
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) == 2:
            label = cols[0].get_text(strip=True)
            value = cols[1].get_text(strip=True)
            if label in desired_fields:
                data[label] = value

    soft_url = f"{url}&format=soft"
    soft_response = ncbi_get(soft_url, timeout=30)
    soft_text = soft_response.text
    platforms = set(re.findall(r"(GPL\d+)", soft_text))
    samples = set(re.findall(r"(GSM\d+)", soft_text))

    data.update({
        "Platforms": ", ".join(sorted(platforms)),
        "Samples": len(samples),
        "Series": gse_id,
        "SuperSeries": super_series,
    })

    supp_data: list[dict] = []
    soft_supp_files = supplementary_files_from_soft(soft_text)

    # HTML supplementary table (may list only GSE*_RAW.tar)
    html_supp_rows: list[dict] = []
    supp_table = None
    for table in soup.find_all("table")[::-1]:
        header_row = table.find("tr")
        if not header_row:
            continue
        headers = [cell.get_text(strip=True) for cell in header_row.find_all(["td", "th"])]
        if "Supplementary file" in headers:
            supp_table = table
            break
    if supp_table:
        for row in supp_table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) >= 1:
                file_name = cells[0].get_text(strip=True)
                if not file_name or file_name.lower().startswith("sra"):
                    continue
                size = cells[1].get_text(strip=True) if len(cells) >= 2 else ""
                html_supp_rows.append(_supp_row(data, file_name, size))

    html_names = [r["Supplementary file"] for r in html_supp_rows]
    needs_custom = has_custom_download_link(soup) or (
        html_names and all(is_archive_bundle(n) for n in html_names)
    )

    if needs_custom:
        custom_rows = fetch_custom_supp_files_xml(gse_id, data)
        if custom_rows:
            supp_data = custom_rows
            logger.info(
                "Loaded %d file(s) from (custom) XML for %s", len(custom_rows), gse_id
            )
        elif driver is not None and SELENIUM_AVAILABLE:
            try:
                driver.get(url)
                wait = WebDriverWait(driver, 7)
                custom_link = wait.until(
                    EC.element_to_be_clickable((By.LINK_TEXT, "(custom)"))
                )
                custom_link.click()
                wait.until(
                    EC.presence_of_all_elements_located(
                        (By.XPATH, "//table//tr[td/input[@type='checkbox']]")
                    )
                )
                for row in driver.find_elements(
                    By.XPATH, "//table//tr[td/input[@type='checkbox']]"
                ):
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        file_name = cells[0].text.strip()
                        if file_name.lower() == "(all files)":
                            continue
                        size = cells[1].text.strip()
                        supp_data.append(_supp_row(data, file_name, size))
            except Exception as e:
                logger.warning("Selenium (custom) fallback failed for %s: %s", gse_id, e)

    if not supp_data and soft_supp_files:
        for file_name, file_url in soft_supp_files:
            if needs_custom and is_archive_bundle(file_name):
                continue
            row = _supp_row(data, file_name, "")
            row["Supplementary URL"] = file_url
            supp_data.append(row)

    if not supp_data and html_supp_rows:
        supp_data = html_supp_rows

    if not supp_data:
        supp_data.append(data)

    return supp_data


def run_geo_pipeline(dir_base, proximity_window=10):
    gds_file_path = os.path.join(dir_base, "gds_result.txt")
    if not os.path.exists(gds_file_path):
        st.error(f"Input file not found: {gds_file_path}.")
        return None

    with open(gds_file_path, encoding="utf-8") as f:
        text = f.read()

    from geoscouter.core.gds_parse import build_gds_processed_df, parse_gds_result

    scrape_gses, excluded_superseries = parse_gds_result(text)
    if excluded_superseries:
        st.info(
            f"Excluded {len(excluded_superseries)} SuperSeries parent(s): "
            + ", ".join(excluded_superseries)
        )
    if not scrape_gses:
        st.error("No subseries/standalone GSEs found in gds_result.txt after filtering.")
        return None

    df = build_gds_processed_df(scrape_gses, proximity_window=proximity_window)
    df.to_csv(os.path.join(dir_base, "gds_processed.csv"), index=False)

    driver = None
    if SELENIUM_AVAILABLE:
        try:
            driver = get_headless_driver()
        except Exception as e:
            logger.warning("Selenium unavailable; using requests only. %s", e)

    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        gse_list = df["GSE"]
        total = len(gse_list)
        for i, gse in enumerate(gse_list):
            status_text.text(f"Scraping {gse} ({i + 1}/{total})...")
            all_data.extend(process_gse(gse, driver))
            progress_bar.progress((i + 1) / total)
        status_text.success("Web scraping complete!")
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    from geoscouter.core.platforms import ensure_platform_labels

    df_combined = pd.DataFrame(all_data)
    df_combined = ensure_platform_labels(df_combined)
    df_combined.to_csv(os.path.join(dir_base, "geo_webscrap.csv"), index=False)
    return df_combined
