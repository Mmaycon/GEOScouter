import os

import GEOparse
import pandas as pd
import streamlit as st


@st.cache_data
def get_gse_metadata(df_filtered: pd.DataFrame, metadata_dir: str):
    list_of_gse_dataframes = []
    gse_ids_to_process = df_filtered["Series"].unique()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, gse_id in enumerate(gse_ids_to_process):
        status_text.text(f"Getting metadata for {gse_id} ({i + 1}/{len(gse_ids_to_process)})...")
        try:
            gse = GEOparse.get_GEO(geo=gse_id, destdir=os.path.join(metadata_dir, "geo_soft_files"))
            all_samples_data = []
            for gsm_name, gsm_obj in gse.gsms.items():
                processed_metadata = {"gsm_id": gsm_name}
                for key, value in gsm_obj.metadata.items():
                    processed_metadata[key] = (
                        value[0] if isinstance(value, list) and len(value) > 0 else value
                    )
                if "characteristics_ch1" in gsm_obj.metadata:
                    for characteristic in gsm_obj.metadata["characteristics_ch1"]:
                        parts = characteristic.split(":", 1)
                        if len(parts) == 2:
                            processed_metadata[parts[0].strip()] = parts[1].strip()
                all_samples_data.append(processed_metadata)
            metadata_df = pd.DataFrame(all_samples_data)
            metadata_df["gse_id"] = gse_id
            list_of_gse_dataframes.append(metadata_df)
        except Exception as e:
            st.warning(f"Failed to get metadata for {gse_id}. Error: {e}")
        progress_bar.progress((i + 1) / len(gse_ids_to_process))

    status_text.success("Metadata extraction complete!")
    return list_of_gse_dataframes
