import plotly.express as px
import streamlit as st


def file_per_sample_complexity(summary_df):
    st.subheader("File vs. Sample Complexity")
    fig = px.scatter(
        summary_df,
        x="num_files",
        y="num_samples",
        color="Platform_Label",
        hover_name="Series",
        hover_data=[
            "Hover_Title",
            "Platform_labels",
            "Platform_technology",
            "Platform_title",
            "Platforms",
            "num_samples",
            "total_size_mb",
        ],
        title="Unique supplementary files vs samples per series",
        labels={"Platform_Label": "Technology"},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Title: %{customdata[0]}<br>"
            "Assay: %{customdata[1]}<br>"
            "Instrument: %{customdata[3]}<br>"
            "GPL: %{customdata[4]}<br>"
            "Samples: %{customdata[5]}<br>"
            "Total size (MB): %{customdata[6]}<extra></extra>"
        )
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")
