import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from geoscouter.core.similarity import (
    SIMILARITY_HELP,
    calculate_reference_edges,
    calculate_similarity_edges,
    reference_comparison_table,
)


def file_similarity_network(df: pd.DataFrame, reference_gse: str | None = None):
    st.subheader("Supplementary file similarity network")
    with st.expander("How is similarity calculated?", expanded=False):
        st.markdown(SIMILARITY_HELP)

    series_files, edges, graph = calculate_similarity_edges(df)
    if not graph.nodes:
        st.warning("Not enough data to create a network graph.")
        return

    import networkx as nx

    ref = reference_gse.strip().upper() if reference_gse else None
    ref_edges: list[tuple[str, str, float]] = []
    use_reference = False

    if ref:
        if ref not in graph.nodes:
            st.warning(
                f"Reference GSE **{ref}** is not in the active dataset. "
                "Showing general pairwise similarity only."
            )
        elif not series_files.get(ref):
            st.warning(
                f"Reference GSE **{ref}** has no supplementary filenames. "
                "Reference edges and comparison table are disabled."
            )
        else:
            use_reference = True
            ref_edges = calculate_reference_edges(series_files, ref)

    pos = nx.spring_layout(graph, seed=42, k=2.5)
    if use_reference and ref in pos:
        ref_xy = pos[ref]
        pos = {n: (x - ref_xy[0], y - ref_xy[1]) for n, (x, y) in pos.items()}
        pos[ref] = (0.0, 0.0)

    ref_neighbors = {e[1] for e in ref_edges} if use_reference else set()
    edge_traces = []

    for s1, s2, weight in edges:
        if use_reference and ref and ref in (s1, s2):
            other = s2 if s1 == ref else s1
            if other in ref_neighbors:
                continue
        if weight <= 0:
            continue
        x0, y0 = pos[s1]
        x1, y1 = pos[s2]
        if use_reference:
            line_style = dict(width=1.5, color="rgba(180, 190, 200, 0.45)")
            hover = f"Pairwise similarity (Jaccard): {weight:.3f}"
        else:
            line_style = dict(
                width=2.5,
                color=px.colors.sample_colorscale("viridis", weight)[0],
            )
            hover = f"Similarity (Jaccard): {weight:.3f}"
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=line_style,
                hoverinfo="text",
                hovertext=hover,
                showlegend=False,
            )
        )

    if use_reference:
        for _, other, weight in ref_edges:
            if weight <= 0:
                continue
            x0, y0 = pos[ref]
            x1, y1 = pos[other]
            width = 2 + 4 * weight
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(
                        width=width,
                        color=px.colors.sample_colorscale(
                            [[0, "#e67e22"], [0.5, "#e74c3c"], [1, "#922b21"]], weight
                        )[0],
                    ),
                    hoverinfo="text",
                    hovertext=f"Similarity to reference {ref}: {weight:.3f}",
                    legendgroup="reference",
                    showlegend=False,
                )
            )

    node_info_df = df.drop_duplicates(subset="Series").set_index("Series")
    node_x, node_y, node_text, node_hover_text, node_sizes, node_colors, node_lines = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        degree = graph.degree(node)
        is_ref = use_reference and node == ref
        if node in node_info_df.index:
            info = node_info_df.loc[node]
            title_val = info.get("Title", "N/A")
            title_str = "N/A" if pd.isna(title_val) else str(title_val)
            assay_val = info.get("Platform_labels") or info.get("Study_type") or "N/A"
            instrument = info.get("Platform_title") or "N/A"
            gpl_val = info.get("Platforms", "N/A")
            ref_label = "<br><b>Reference layout</b>" if is_ref else ""
            node_hover_text.append(
                f"<b>{node}</b>{ref_label}<br>Title: {title_str[:80]}<br>"
                f"Assay: {assay_val}<br>"
                f"Instrument: {instrument}<br>"
                f"GPL: {gpl_val}<br>"
                f"Samples: {info.get('Samples', 'N/A')}<br>"
                f"Connections: {degree}"
            )
        else:
            ref_label = "<br><b>Reference layout</b>" if is_ref else ""
            node_hover_text.append(f"<b>{node}</b>{ref_label}<br>Connections: {degree}")

        if is_ref:
            node_sizes.append(22)
            node_colors.append("#d4a017")
            node_lines.append(dict(width=3, color="#8b6914"))
        else:
            node_sizes.append(15)
            node_colors.append("#5a6c7d")
            node_lines.append(dict(width=2, color="#ffffff"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=node_lines,
        ),
    )

    data = edge_traces + [node_trace]
    all_weights = [e[2] for e in edges if e[2] > 0]
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

    annotations = []
    if use_reference:
        annotations.extend(
            [
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text="Gray: pairwise similarity",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=11, color="#6b7280"),
                ),
                dict(
                    x=0.02,
                    y=0.94,
                    xref="paper",
                    yref="paper",
                    text=f"Orange/red: similarity to {ref}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=11, color="#c0392b"),
                ),
            ]
        )

    title = "Series linked by supplementary filename similarity"
    if use_reference:
        title += f" (reference: {ref})"

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
        ),
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, width="stretch")

    if use_reference:
        st.markdown("#### Comparison to reference GSE")
        st.dataframe(
            reference_comparison_table(series_files, ref),
            width="stretch",
            hide_index=True,
        )
