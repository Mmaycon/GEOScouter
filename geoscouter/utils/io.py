import hashlib
import os
import re

import pandas as pd
import streamlit as st


def offer_download(file_path: str, label: str | None = None):
    file_path = str(file_path)
    if not os.path.exists(file_path):
        return

    abs_path = os.path.abspath(file_path)
    st.session_state.setdefault("_rendered_download_keys", set())

    filename = os.path.basename(abs_path)
    label = label or f"Download {filename}"

    if filename.endswith(".csv"):
        mime = "text/csv"
    elif filename.endswith(".xlsx"):
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        mime = "application/octet-stream"

    key = "download_" + hashlib.md5(abs_path.encode("utf-8")).hexdigest()
    if key in st.session_state["_rendered_download_keys"]:
        return

    st.session_state["_rendered_download_keys"].add(key)

    with open(abs_path, "rb") as f:
        st.download_button(
            label=label,
            data=f,
            file_name=filename,
            mime=mime,
            key=key,
        )


def sanitize_sheet_name(name: str, used: set) -> str:
    name = str(name) if pd.notna(name) and str(name).strip() else "NA"
    bad = r"[:\\/?*\[\]]"
    safe = re.sub(bad, "_", name)
    safe = safe[:31] or "NA"
    base = safe
    i = 1
    while safe in used:
        suffix = f"_{i}"
        safe = (base[: 31 - len(suffix)] + suffix) if len(base) + len(suffix) > 31 else base + suffix
        i += 1
    used.add(safe)
    return safe
