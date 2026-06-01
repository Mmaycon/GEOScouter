import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from geoscouter.core.similarity import SIMILARITY_HELP, calculate_similarity_edges


def file_similarity_network(df: pd.DataFrame):
    st.subheader("Supplementary file similarity network")
    with st.expander("How is similarity calculated?", expanded=False):
        st.markdown(SIMILARITY_HELP)

    _, _, graph = calculate_similarity_edges(df)
    if not graph.nodes:
        st.warning("Not enough data to create a network graph.")
        return

    import networkx as nx

    pos = nx.spring_layout(graph, seed=42, k=2.5)
    edge_traces = []
    all_weights = [d["weight"] for _, _, d in graph.edges(data=True)]

    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]["weight"]
        color = px.colors.sample_colorscale("viridis", weight)[0]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=2.5, color=color),
                hoverinfo="text",
                hovertext=f"Similarity (Jaccard): {weight:.3f}",
            )
        )

    node_info_df = df.drop_duplicates(subset="Series").set_index("Series")
    node_x, node_y, node_text, node_hover_text = [], [], [], []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        degree = graph.degree(node)
        if node in node_info_df.index:
            info = node_info_df.loc[node]
            title_val = info.get("Title", "N/A")
            title_str = "N/A" if pd.isna(title_val) else str(title_val)
            platform_val = info.get("Platform_labels") or info.get("Platforms", "N/A")
            if pd.isna(platform_val) or not str(platform_val).strip():
                platform_val = info.get("Platforms", "N/A")
            node_hover_text.append(
                f"<b>{node}</b><br>Title: {title_str[:80]}<br>"
                f"Platform: {platform_val}<br>"
                f"Samples: {info.get('Samples', 'N/A')}<br>"
                f"Connections: {degree}"
            )
        else:
            node_hover_text.append(f"<b>{node}</b><br>Connections: {degree}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover_text,
        marker=dict(
            size=15,
            color="#5a6c7d",
            line=dict(width=2, color="#ffffff"),
        ),
    )

    data = edge_traces + [node_trace]
    if all_weights:
        data.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale="viridis",
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(thickness=15, title="Similarity"),
                ),
                hoverinfo="none",
            )
        )

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title="Series linked by supplementary filename similarity",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
