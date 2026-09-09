"""
GEOScouter – Streamlit entry point (deploy on Streamlit Community Cloud).

Run locally:
  streamlit run streamlit_app.py
"""

try:
    from bs4 import BeautifulSoup  # noqa: F401
except Exception as e:
    import streamlit as st

    st.set_page_config(layout="wide")
    st.title("GEOScouter – dependency error")
    st.error(
        "Missing package: beautifulsoup4. Add it to requirements.txt and redeploy.\n\n"
        f"Import error: {e}"
    )
    raise

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from geoscouter.config import CACHE_FILES, GDS_INPUT_NAME, WORK_DIR
from geoscouter.core.filters import (
    apply_series_filters,
    filter_metadata_tables,
    metadata_filter_options,
)
from geoscouter.core.metadata import get_gse_metadata
from geoscouter.core.pipeline import run_geo_pipeline
from geoscouter.core.platforms import (
    ensure_platform_labels,
    gpl_filter_options,
    platform_filter_label,
    technology_filter_options,
)
from geoscouter.core.similarity import (
    SUPERVISED_SIMILARITY_HELP,
    calculate_similarity_edges,
    discover_pattern_candidates,
    manual_pattern_editor_df,
    parse_manual_patterns,
    score_supervised_candidates,
    signature_rules_from_dataframe,
)
from geoscouter.utils.io import offer_download, sanitize_sheet_name
from geoscouter.utils.summary import build_series_summary, normalize_scrape_df
from geoscouter.viz.complexity import file_per_sample_complexity
from geoscouter.viz.network import supervised_file_similarity_network
from geoscouter.viz.snapshot import dataset_snapshot_plots

st.set_page_config(layout="wide", page_title="GEOScouter")
st.title("GEOScouter – curating datasets from GEO")

st.markdown(
    """
Explore GEO series from an exported `gds_result.txt`, filter by platform/samples/metadata,
then visualize and compare supplementary files. Deployed with [Streamlit Community Cloud](https://streamlit.io/cloud).
"""
)

# Session defaults
for key, default in {
    "df_combined": None,
    "df_active": None,
    "summary_df": None,
    "gse_selection_list": [],
    "gse_df_filtered": None,
    "list_of_metadata_dfs": None,
    "metadata_search_results": None,
    "metadata_matched_gses": None,
    "supervised_signature_mode": "training_gse",
    "supervised_manual_patterns_text": "",
    "supervised_training_gses": [],
    "supervised_pattern_training_key": None,
    "supervised_pattern_editor_df": None,
    "supervised_applied_rules": [],
    "supervised_signature_applied": False,
    "viz_show_snapshot": False,
    "viz_show_complexity": False,
    "viz_show_network": False,
}.items():
    st.session_state.setdefault(key, default)


def _reset_visualizations() -> None:
    """Hide step-3 plots after the working set changes."""
    st.session_state.viz_show_snapshot = False
    st.session_state.viz_show_complexity = False
    st.session_state.viz_show_network = False
    st.session_state.summary_df = None


def _file_similarity_network_ui(df: pd.DataFrame) -> None:
    """Supervised signature network for step 3."""
    series_opts = sorted(df["Series"].dropna().unique())
    series_files, _, _ = calculate_similarity_edges(df)

    signature_mode = st.radio(
        "Signature source",
        options=["From training GSE(s)", "Manual file types"],
        horizontal=True,
        index=0 if st.session_state.supervised_signature_mode == "training_gse" else 1,
        help="Learn patterns from training GSEs or type expected file-type tokens directly.",
    )
    st.session_state.supervised_signature_mode = (
        "training_gse"
        if signature_mode == "From training GSE(s)"
        else "manual"
    )

    training: list[str] = []
    editor_df = st.session_state.supervised_pattern_editor_df
    apply_signature = False

    if st.session_state.supervised_signature_mode == "manual":
        st.caption(
            "Enter one file-type token per line (e.g. `transcripts.parquet.gz`, "
            "`xenium.txt.gz`). No training GSE is required."
        )
        manual_text = st.text_area(
            "Expected file types",
            value=st.session_state.supervised_manual_patterns_text,
            height=120,
            placeholder="transcripts.parquet.gz\nxenium.txt.gz\ncell_matrix.mtx.gz",
            key="supervised_manual_patterns_input",
        )
        st.session_state.supervised_manual_patterns_text = manual_text

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            build_manual = st.button("Build signature from file types")
        with btn_col2:
            apply_signature = st.button("Apply signature & show network", type="primary")
        with btn_col3:
            if st.button("Reset manual signature"):
                st.session_state.supervised_manual_patterns_text = ""
                st.session_state.supervised_pattern_editor_df = None
                st.session_state.supervised_applied_rules = []
                st.session_state.supervised_signature_applied = False
                st.rerun()

        if build_manual:
            patterns = parse_manual_patterns(manual_text)
            if not patterns:
                st.error("Enter at least one file-type token.")
            else:
                st.session_state.supervised_pattern_editor_df = manual_pattern_editor_df(
                    patterns
                )
                st.session_state.supervised_applied_rules = []
                st.session_state.supervised_signature_applied = False
                st.rerun()

        editor_df = st.session_state.supervised_pattern_editor_df
        if editor_df is None or editor_df.empty:
            st.info("Enter file types above, then click **Build signature from file types**.")
            return

        st.markdown("#### Review signature patterns")
        edited_df = st.data_editor(
            editor_df,
            column_config={
                "Include": st.column_config.CheckboxColumn(
                    help="Include this rule when scoring other GSEs.",
                ),
                "Match pattern": st.column_config.TextColumn(
                    help="Structural file-type token searched within supplementary filenames.",
                    required=True,
                ),
                "Weight": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.3f",
                ),
                "rule_id": None,
            },
            hide_index=True,
            key="supervised_manual_pattern_editor",
            width="stretch",
        )
        st.session_state.supervised_pattern_editor_df = edited_df

    else:
        valid_training = [
            g for g in st.session_state.supervised_training_gses if g in series_opts
        ]
        st.session_state.supervised_training_gses = valid_training

        col_a, col_b = st.columns([3, 1])
        with col_a:
            training_choice = st.multiselect(
                "Training GSEs (define the file structure signature)",
                options=series_opts,
                default=valid_training,
                key="supervised_training_select",
                help="Select GSEs whose supplementary file layout should teach the signature.",
            )
            st.session_state.supervised_training_gses = [
                str(g).strip().upper() for g in training_choice
            ]
        with col_b:
            if st.button(
                "Use comparison list",
                help="Copy GSE IDs from step 4 comparison list into training GSEs.",
            ):
                from_list = sorted(
                    {
                        g.strip().upper()
                        for g in st.session_state.gse_selection_list
                        if g and g.strip().upper() in series_opts
                    }
                )
                st.session_state.supervised_training_gses = from_list
                st.session_state["supervised_training_select"] = from_list
                st.rerun()

        min_support = st.slider(
            "Minimum pattern support across training GSEs",
            0.5,
            1.0,
            0.8,
            0.05,
            help="Used when discovering patterns: a rule is pre-selected if it appears in "
            "at least this fraction of training GSEs. You can still toggle any rule manually.",
        )

        training = st.session_state.supervised_training_gses
        if not training:
            st.info("Select at least one training GSE to discover file-structure patterns.")
            return

        with st.expander("Supplementary files in training GSE(s)", expanded=False):
            for gse in sorted(training):
                files = sorted(series_files.get(gse, set()))
                st.markdown(f"**{gse}** — {len(files)} file(s)")
                if files:
                    st.code("\n".join(files), language=None)
                else:
                    st.caption("No supplementary filenames found for this GSE.")

        training_key = (tuple(sorted(training)), min_support)

        if training_key != st.session_state.supervised_pattern_training_key:
            st.session_state.supervised_pattern_training_key = training_key
            st.session_state.supervised_pattern_editor_df = discover_pattern_candidates(
                series_files, training, min_support_ratio=min_support
            )
            st.session_state.supervised_applied_rules = []
            st.session_state.supervised_signature_applied = False

        st.markdown("#### Review signature patterns")
        st.caption(
            "**Auto-detected** shows the full normalized filename (sample IDs replaced). "
            "**Match pattern** is the structural file-type token used for scoring "
            "(e.g. `transcripts.parquet.gz`, `cell_matrix.mtx.gz`). "
            "Duplicate match patterns are merged. Edit or uncheck **Include** as needed."
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Rediscover patterns"):
                st.session_state.supervised_pattern_editor_df = discover_pattern_candidates(
                    series_files, training, min_support_ratio=min_support
                )
                st.session_state.supervised_applied_rules = []
                st.session_state.supervised_signature_applied = False
                st.rerun()
        with btn_col2:
            apply_signature = st.button("Apply signature & show network", type="primary")

        editor_df = st.session_state.supervised_pattern_editor_df
        if editor_df is None or editor_df.empty:
            st.warning("No supplementary filenames found for the selected training GSEs.")
            return

        edited_df = st.data_editor(
            editor_df,
            column_config={
                "Include": st.column_config.CheckboxColumn(
                    help="Include this rule when scoring other GSEs.",
                ),
                "Match pattern": st.column_config.TextColumn(
                    help="Structural file-type token searched within supplementary filenames.",
                    required=True,
                ),
                "Auto-detected": st.column_config.TextColumn(
                    disabled=True,
                    help="Full normalized filename pattern (sample/study IDs replaced).",
                ),
                "Example filenames": st.column_config.TextColumn(
                    disabled=True,
                ),
                "Training GSEs": st.column_config.TextColumn(
                    disabled=True,
                ),
                "Weight": st.column_config.NumberColumn(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.3f",
                ),
                "rule_id": None,
            },
            disabled=["Auto-detected", "Example filenames", "Training GSEs"],
            hide_index=True,
            key="supervised_pattern_data_editor",
            width="stretch",
        )
        st.session_state.supervised_pattern_editor_df = edited_df

    if apply_signature:
        rules = signature_rules_from_dataframe(st.session_state.supervised_pattern_editor_df)
        if not rules:
            st.error("Enable at least one pattern with a non-empty Match pattern.")
        else:
            st.session_state.supervised_applied_rules = rules
            st.session_state.supervised_signature_applied = True

    if (
        st.session_state.supervised_signature_applied
        and st.session_state.supervised_applied_rules
    ):
        training = (
            st.session_state.supervised_training_gses
            if st.session_state.supervised_signature_mode == "training_gse"
            else []
        )
        supervised_file_similarity_network(
            df,
            training_gses=training,
            rules=st.session_state.supervised_applied_rules,
            signature_mode=st.session_state.supervised_signature_mode,
        )
    else:
        st.info("Edit patterns above, then click **Apply signature & show network**.")

st.session_state["_rendered_download_keys"] = set()
dir_base = str(WORK_DIR)

# --- 1. Scraping ---
st.header("1. Run data scraping")
st.caption(
    "Export `gds_result.txt` from [GEO DataSets](https://www.ncbi.nlm.nih.gov/gds/) after your search."
)

uploaded_gds = st.file_uploader("Upload gds_result.txt", type=["txt"])

if st.button("Delete all cached outputs"):
    deleted, missing, errors = [], [], []
    for fname in CACHE_FILES:
        fpath = os.path.join(dir_base, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                deleted.append(fname)
            except Exception as ex:
                errors.append(f"{fname}: {ex}")
        else:
            missing.append(fname)
    for key in [
        "df_combined", "df_active", "summary_df", "gse_df_filtered",
        "list_of_metadata_dfs", "metadata_search_results", "metadata_matched_gses",
    ]:
        st.session_state[key] = None
    st.session_state["gse_selection_list"] = []
    st.cache_data.clear()
    if deleted:
        st.success("Deleted: " + ", ".join(deleted))
    if errors:
        st.error("\n".join(errors))
    st.rerun()

if uploaded_gds is not None:
    (WORK_DIR / GDS_INPUT_NAME).write_bytes(uploaded_gds.getvalue())
    st.success(f"Saved upload to {WORK_DIR / GDS_INPUT_NAME}")

if st.button("Run pipeline", type="primary"):
    expected = WORK_DIR / GDS_INPUT_NAME
    if not expected.exists():
        st.error("Upload gds_result.txt first.")
    else:
        cache_path = os.path.join(dir_base, "geo_webscrap.csv")
        if os.path.exists(cache_path):
            st.info(f"Loading cached scrape: {cache_path}")
            st.session_state.df_combined = pd.read_csv(cache_path)
        else:
            st.info("Running web scrape (requests-first; Selenium only if available locally).")
            result = run_geo_pipeline(dir_base)
            if result is not None:
                st.session_state.df_combined = result
        if st.session_state.df_combined is not None:
            st.session_state.df_combined = ensure_platform_labels(
                st.session_state.df_combined,
                gds_path=WORK_DIR / GDS_INPUT_NAME,
            )
            st.session_state.df_active = st.session_state.df_combined.copy()
            st.success(
                f"Ready: {st.session_state.df_combined['Series'].nunique()} series, "
                f"{len(st.session_state.df_combined)} rows."
            )

st.subheader("Downloads")
found = False
for fname in CACHE_FILES:
    fp = os.path.join(dir_base, fname)
    if os.path.exists(fp):
        found = True
        offer_download(fp, f"Download {fname}")
if not found:
    st.info("No export files yet.")

# --- 2. Upstream filters (before plots) ---
if st.session_state.df_combined is not None:
    st.header("2. Filter datasets")
    st.caption("Apply filters here first so visualizations use the narrowed set.")

    st.session_state.df_combined = ensure_platform_labels(
        st.session_state.df_combined,
        gds_path=WORK_DIR / GDS_INPUT_NAME,
    )
    if st.session_state.df_active is not None:
        st.session_state.df_active = ensure_platform_labels(
            st.session_state.df_active,
            gds_path=WORK_DIR / GDS_INPUT_NAME,
        )
    df_base = normalize_scrape_df(st.session_state.df_combined)
    series_level = df_base.drop_duplicates(subset=["Series"])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        all_technologies = technology_filter_options(df_base)
        selected_technologies = st.multiselect(
            "Assay type (gds Type)",
            options=all_technologies,
            help="From the Type field in gds_result.txt. "
            "When Type is \"Other\", the assay tag from the series title is used.",
        )
    with col_b:
        all_gpls = gpl_filter_options(df_base)
        selected_platforms = st.multiselect(
            "Platform (GPL)",
            options=all_gpls,
            format_func=platform_filter_label,
            help="GEO platform Title from the platform browser (e.g. Xenium In Situ Analyzer).",
        )
    with col_c:
        min_samples = st.number_input("Min samples (0 = off)", min_value=0, value=0, step=10)
        max_samples = st.number_input("Max samples (0 = off)", min_value=0, value=0, step=10)

    if st.button("Apply scrape-level filters", type="primary"):
        _reset_visualizations()
        st.session_state.df_active = apply_series_filters(
            st.session_state.df_combined,
            selected_technologies=selected_technologies or None,
            selected_platforms=selected_platforms or None,
            min_samples=min_samples,
            max_samples=max_samples,
            gse_selection=None,
        )
        n = st.session_state.df_active["Series"].nunique()
        st.success(f"Active dataset: {n} series.")

    if st.button("Reset to full scrape"):
        _reset_visualizations()
        st.session_state.df_active = st.session_state.df_combined.copy()
        st.rerun()

    active = st.session_state.df_active
    if active is not None:
        st.info(f"Working set: **{active['Series'].nunique()}** series.")

    # Metadata fetch + upstream metadata filters
    st.subheader("Sample metadata (optional)")
    st.caption(
        "Fetch GSM-level metadata to filter by assay, organism, tissue, library source, etc. "
        "This step downloads SOFT files and can take several minutes."
    )

    meta_scope = st.radio(
        "Metadata scope",
        ["Active filtered series", "All scraped series"],
        horizontal=True,
    )

    if st.button("Fetch GEO sample metadata"):
        scope_df = active if meta_scope.startswith("Active") and active is not None else st.session_state.df_combined
        if scope_df is None:
            st.warning("No data to process.")
        else:
            metadata_dir = os.path.join(dir_base, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)
            st.session_state.list_of_metadata_dfs = get_gse_metadata(scope_df, metadata_dir)

    if st.session_state.list_of_metadata_dfs:
        options = metadata_filter_options(st.session_state.list_of_metadata_dfs)
        if not options:
            st.warning("No standard metadata fields found in downloaded tables.")
        else:
            field_filters = {}
            cols = st.columns(2)
            for i, (field, values) in enumerate(sorted(options.items())):
                with cols[i % 2]:
                    chosen = st.multiselect(
                        field.replace("_", " ").title(),
                        options=values,
                        key=f"meta_filter_{field}",
                    )
                    if chosen:
                        field_filters[field] = chosen

            if st.button("Apply metadata filters to working set"):
                matched = filter_metadata_tables(
                    st.session_state.list_of_metadata_dfs, field_filters
                )
                st.session_state.metadata_matched_gses = matched
                if active is not None and matched:
                    _reset_visualizations()
                    st.session_state.df_active = active[
                        active["Series"].isin({g.upper() for g in matched})
                    ].copy()
                    st.success(
                        f"Metadata filters matched {len(matched)} GSEs; working set updated."
                    )
                elif not matched:
                    st.warning("No GSE matched the selected metadata filters.")

# --- 3. Visualize (once, on active set) ---
if st.session_state.df_active is not None and not st.session_state.df_active.empty:
    st.header("3. Visualize datasets")
    st.caption(
        "Plots reflect the current working set from step 2. "
        "Click **Similarity network** to define a supervised file-structure signature."
    )

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        if st.button("Generate all visualizations", type="primary"):
            st.session_state.viz_show_snapshot = True
            st.session_state.viz_show_complexity = True
            st.session_state.viz_show_network = True
    with c1:
        if st.button("Snapshot bars"):
            st.session_state.viz_show_snapshot = True
    with c2:
        if st.button("File vs samples"):
            st.session_state.viz_show_complexity = True
    with c3:
        if st.button("Similarity network"):
            st.session_state.viz_show_network = True

    if any(
        (
            st.session_state.viz_show_snapshot,
            st.session_state.viz_show_complexity,
            st.session_state.viz_show_network,
        )
    ):
        if st.session_state.viz_show_snapshot or st.session_state.viz_show_complexity:
            summary = st.session_state.summary_df
            if summary is None:
                summary = build_series_summary(st.session_state.df_active)
                st.session_state.summary_df = summary
            if st.session_state.viz_show_snapshot:
                dataset_snapshot_plots(summary)
            if st.session_state.viz_show_complexity:
                file_per_sample_complexity(summary)
        if st.session_state.viz_show_network:
            _file_similarity_network_ui(st.session_state.df_active)

# --- 4. GSE selection ---
if st.session_state.df_active is not None:
    st.header("4. Select specific GSEs")
    with st.container(border=True):
        st.subheader("Build a comparison list")
        st.caption(
            "Use this list to restrict exports and downstream steps. "
            "Scrape-level filters in step 2 already narrow the working set; "
            "here you add GSEs by similarity or manual ID."
        )

        with st.expander("About signature similarity", expanded=False):
            st.markdown(SUPERVISED_SIMILARITY_HELP)

        col1, col2 = st.columns(2)
        with col1:
            sim_threshold = st.slider(
                "Minimum signature similarity",
                0.0,
                1.0,
                0.8,
                0.05,
                help="Used when bulk-adding GSEs that match the step 3 signature.",
            )
            if (
                st.session_state.supervised_signature_applied
                and st.session_state.supervised_applied_rules
            ):
                if st.button(
                    "Add GSEs matching signature (≥ threshold)",
                    help="Adds GSEs whose supervised signature score meets the threshold.",
                ):
                    series_files, _, _ = calculate_similarity_edges(
                        st.session_state.df_active
                    )
                    training = (
                        st.session_state.supervised_training_gses
                        if st.session_state.supervised_signature_mode == "training_gse"
                        else []
                    )
                    _, scores = score_supervised_candidates(
                        series_files,
                        training,
                        st.session_state.supervised_applied_rules,
                    )
                    added = {
                        gse for gse, score in scores.items() if score >= sim_threshold
                    }
                    before = len(set(st.session_state.gse_selection_list))
                    st.session_state.gse_selection_list.extend(sorted(added))
                    after = len(set(st.session_state.gse_selection_list))
                    st.success(
                        f"Added {after - before} GSEs matching the signature "
                        f"(≥{sim_threshold}, {len(added)} matched)."
                    )
            else:
                st.caption(
                    "Apply a signature in step 3 (**Apply signature & show network**) "
                    "to enable bulk add by signature similarity."
                )
        with col2:
            manual_gse = st.text_input("Manual GSE ID", placeholder="GSE12345")
            if st.button("Add manual GSE") and manual_gse:
                st.session_state.gse_selection_list.append(manual_gse.strip().upper())
                st.rerun()

        unique_n = len(set(st.session_state.gse_selection_list))
        st.write(f"**Comparison list:** {unique_n} unique GSEs")
        if st.button("Clear comparison list"):
            st.session_state.gse_selection_list = []
            st.rerun()

        if st.button("Export comparison list as filtered table", type="primary"):
            df_out = apply_series_filters(
                st.session_state.df_active,
                gse_selection=st.session_state.gse_selection_list or None,
            )
            if st.session_state.gse_selection_list:
                df_out = df_out[
                    df_out["Series"].isin(
                        {s.upper() for s in st.session_state.gse_selection_list}
                    )
                ]
            st.session_state.gse_df_filtered = df_out
            out_path = os.path.join(dir_base, "filtered_geo_webscrap.csv")
            df_out.to_csv(out_path, index=False)
            st.success(f"Saved {df_out['Series'].nunique()} series.")
            offer_download(out_path)

# --- 5. Metadata exploration ---
st.header("5. Metadata analysis")
if st.session_state.list_of_metadata_dfs:
    gse_options = [df["gse_id"].iloc[0] for df in st.session_state.list_of_metadata_dfs]
    selected_gse = st.selectbox("View samples for GSE", gse_options)
    for df in st.session_state.list_of_metadata_dfs:
        if df["gse_id"].iloc[0] == selected_gse:
            st.dataframe(df, width="stretch")
            break

    if st.button("Export all metadata to Excel"):
        excel_out = os.path.join(dir_base, "metadata_GSE.xlsx")
        used = set()
        with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
            for df in st.session_state.list_of_metadata_dfs:
                gse_id = df["gse_id"].iloc[0]
                sheet = sanitize_sheet_name(gse_id, used)
                df.to_excel(writer, sheet_name=sheet, index=False)
        st.success("Metadata workbook ready.")
        offer_download(excel_out)

    keyword = st.text_input("Keyword search across metadata")
    if st.button("Search metadata") and keyword:
        found_gses, found_gsms, counts = set(), set(), {}
        for df in st.session_state.list_of_metadata_dfs:
            mask = df.apply(
                lambda col: col.astype(str).str.contains(keyword, case=False, na=False)
            )
            if mask.values.any():
                gse_id = df["gse_id"].iloc[0]
                matching = df.loc[mask.any(axis=1), "gsm_id"].tolist()
                found_gses.add(gse_id)
                found_gsms.update(matching)
                counts[gse_id] = len(matching)
        st.session_state.metadata_search_results = {
            "gse_vector": sorted(found_gses),
            "gsm_vector": sorted(found_gsms),
            "counts": counts,
            "keyword": keyword,
        }

    if st.session_state.metadata_search_results:
        res = st.session_state.metadata_search_results
        st.info(
            f"Keyword '{res['keyword']}': {len(res['gse_vector'])} GSEs, "
            f"{len(res['gsm_vector'])} GSMs."
        )
        plot_df = pd.DataFrame(
            list(res["counts"].items()), columns=["GSE", "GSM Count"]
        ).sort_values("GSM Count", ascending=False)
        st.plotly_chart(
            px.bar(plot_df, x="GSE", y="GSM Count", title=f"Matches for '{res['keyword']}'"),
            width="stretch",
        )
        if st.button("Export keyword-filtered metadata"):
            excel_out = os.path.join(dir_base, "metadata_filtered_by_word.xlsx")
            used = set()
            with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
                for df in st.session_state.list_of_metadata_dfs:
                    gse_id = df["gse_id"].iloc[0]
                    if gse_id in res["gse_vector"]:
                        sub = df[df["gsm_id"].isin(res["gsm_vector"])]
                        if not sub.empty:
                            sheet = sanitize_sheet_name(gse_id, used)
                            sub.to_excel(writer, sheet_name=sheet, index=False)
            offer_download(excel_out)
else:
    st.info("Fetch metadata in step 2 to enable exploration here.")
