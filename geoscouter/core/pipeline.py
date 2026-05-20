"""GEO web scraping pipeline (Streamlit Cloud compatible — no Selenium required)."""

import logging
import os
import re

import pandas as pd
import requests
import streamlit as st
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


def parse_custom_supp_files(custom_href: str, base_url: str, data: dict):
    custom_url = urljoin(base_url, custom_href)
    r = requests.get(custom_url, timeout=30)
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
        file_type_resource = ""
        if len(tds) >= 5:
            file_type_resource = tds[4].get_text(strip=True)
        elif len(tds) >= 4:
            file_type_resource = tds[3].get_text(strip=True)
        parts = file_name.split(".")
        file_type = parts[1] if len(parts) > 1 else "unknown"
        row_dict = data.copy()
        row_dict.update({
            "Supplementary file": file_name,
            "Size": size,
            "File type/resource": file_type_resource or file_type,
        })
        supp_data.append(row_dict)
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
    response = requests.get(url, timeout=30)
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
    soft_response = requests.get(soft_url, timeout=30)
    soft_text = soft_response.text
    platforms = set(re.findall(r"(GPL\d+)", soft_text))
    samples = set(re.findall(r"(GSM\d+)", soft_text))

    data.update({
        "Platforms": ", ".join(sorted(platforms)),
        "Samples": len(samples),
        "Series": gse_id,
        "SuperSeries": super_series,
    })

    supp_data = []
    soft_supp_files = supplementary_files_from_soft(soft_text)
    if soft_supp_files:
        for file_name, file_url in soft_supp_files:
            parts = file_name.split(".")
            file_type = parts[1] if len(parts) > 1 else "unknown"
            row_dict = data.copy()
            row_dict.update({
                "Supplementary file": file_name,
                "Size": "",
                "File type/resource": file_type,
                "Supplementary URL": file_url,
            })
            supp_data.append(row_dict)

    custom_link_tag = soup.find("a", string="(custom)")
    if custom_link_tag:
        try:
            href = custom_link_tag.get("href")
            if href:
                parsed = parse_custom_supp_files(href, url, data)
                if parsed:
                    supp_data.extend(parsed)

            if not supp_data and driver is not None and SELENIUM_AVAILABLE:
                driver.get(url)
                wait = WebDriverWait(driver, 7)
                custom_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "(custom)")))
                custom_link.click()
                wait.until(
                    EC.presence_of_all_elements_located(
                        (By.XPATH, "//table//tr[td/input[@type='checkbox']]")
                    )
                )
                for row in driver.find_elements(By.XPATH, "//table//tr[td/input[@type='checkbox']]"):
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        file_name = cells[0].text.strip()
                        if file_name.lower() == "(all files)":
                            continue
                        size = cells[1].text.strip()
                        parts = file_name.split(".")
                        file_type = parts[1] if len(parts) > 1 else "unknown"
                        row_dict = data.copy()
                        row_dict.update({
                            "Supplementary file": file_name,
                            "Size": size,
                            "File type/resource": file_type,
                        })
                        supp_data.append(row_dict)
        except Exception as e:
            logger.warning("Error processing custom link for %s: %s", gse_id, e)

    if not supp_data:
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
                if len(cells) >= 4:
                    row_dict = data.copy()
                    row_dict.update({
                        "Supplementary file": cells[0].get_text(strip=True),
                        "Size": cells[1].get_text(strip=True),
                        "File type/resource": cells[3].get_text(strip=True),
                    })
                    supp_data.append(row_dict)
        else:
            supp_data.append(data)

    return supp_data


def run_geo_pipeline(dir_base, proximity_window=10):
    gds_file_path = os.path.join(dir_base, "gds_result.txt")
    if not os.path.exists(gds_file_path):
        st.error(f"Input file not found: {gds_file_path}.")
        return None

    with open(gds_file_path, encoding="utf-8") as f:
        text = f.read()

    gse_ids = sorted(set(re.findall(r"GSE(\d+)\b", text)), key=int)

    clusters = []
    cluster_index = 0
    current_cluster = []
    prev_num = None

    for gse_num_str in gse_ids:
        gse_num = int(gse_num_str)
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

    gse_to_cluster = {}
    for cluster_id, gse_list in clusters:
        for val in gse_list:
            gse_to_cluster[val] = cluster_id

    rows = []
    for gse_num_str in gse_ids:
        gse_num = int(gse_num_str)
        cluster_id = gse_to_cluster.get(gse_num)
        rows.append({"GSE": f"GSE{gse_num_str}", "Cluster": f"Cluster{cluster_id}"})

    df = pd.DataFrame(rows).drop_duplicates(subset=["GSE"]).reset_index(drop=True)
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

    df_combined = pd.DataFrame(all_data)
    df_combined.to_csv(os.path.join(dir_base, "geo_webscrap.csv"), index=False)
    return df_combined
