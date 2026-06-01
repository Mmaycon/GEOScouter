"""Dataset snapshot bar charts (Plotly with platform tooltips)."""

import plotly.express as px
import streamlit as st


def dataset_snapshot_plots(summary_df):
    st.subheader("Dataset Snapshot")
    metrics = [
        ("total_size_mb", "Total Size per Series", "Total Size (MB)"),
        ("num_samples", "Samples per Series", "Number of Samples"),
        ("num_file_types", "File Types per Series", "Number of File Types"),
    ]
    for y_col, title, ylabel in metrics:
        fig = px.bar(
            summary_df,
            x="Series",
            y=y_col,
            color="Platform_Label",
            hover_data={
                "Series": True,
                "Platform_Label": True,
                "Platform_labels": True,
                "Platforms": True,
                "Hover_Title": True,
                "num_samples": True,
                "total_size_mb": True,
            },
            title=title,
            labels={"Platform_Label": "Platform"},
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Platform: %{customdata[1]}<br>"
                "GPL: %{customdata[2]}<br>"
                "Title: %{customdata[3]}<br>"
                f"{ylabel}: %{{y}}<extra></extra>"
            )
        )
        fig.update_layout(
            height=500,
            xaxis_tickangle=-90,
            legend_title_text="Platform",
        )
        st.plotly_chart(fig, use_container_width=True)
