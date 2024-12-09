"""Graph rendering module."""
import time

import streamlit as st
from st_link_analysis import st_link_analysis
from streamlit.components.v1 import html

from agot_graph._models import EdgeData, MultiGraphData
from agot_graph._styles import (EDGE_STYLES, FINAL_ANSWER_TEMPLATE, ST_COLORS,
                                style_nodes)

GRAPH_LAYOUT = "cose"
RENDER_INTERVAL = 1  # seconds
FONT_COLOR = ST_COLORS["st-white"]


@st.dialog("✅ Final Answer", width="large")
def display_final_answer(content: str = "", long_content: str = ""):
    """Pop-up to display the final answer."""

    model = st.session_state.get("OPENAI-MODEL", "gpt-4o-mini")

    if content and long_content:
        st.markdown(
            FINAL_ANSWER_TEMPLATE.format(
                content=content,
                long_content=long_content,
                model=model,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.write("No final answer available. Try asking a question.")


def render_graphs(_graphs: MultiGraphData):
    """Graph rendering function. Process objects into link-graph data."""

    st.session_state.final_content = final_content = ""
    st.session_state.final_content_long = final_content_long = ""
    nodes, edges = [], []  # default
    if len(_graphs.graphs) == 0:
        elements = {"nodes": nodes, "edges": edges}  # default empty view

    else:
        depths_list = []
        for g in _graphs.graphs:
            depths_list.append((g.depth, g))

        if sum(x[0] == 0 for x in depths_list) != 1:
            raise RuntimeError("Data does not contain exactly one top graph")

        # Sort for ascending depth.
        graphs = [x[1] for x in sorted(depths_list, key=lambda x: x[0])]

        # Start with nodes in top graph.
        top_graph = graphs[0]
        elements = top_graph.export_data()

        nodes, edges = elements["nodes"], elements["edges"]

        # Add deeper subgraph nodes and connect to complex nodes at higher level.

        for graph in graphs[1:]:

            for n in graph.nodes:
                nodes.append({"data": n.model_dump_render()})
                if (
                    n.final
                    and (parent_node_id := graph.parent_node_id) is not None
                ):
                    # Subgraph to complex node edge.
                    e = EdgeData(
                        id=f"{n.id}-parent",
                        source=n.id,
                        target=parent_node_id,
                        label="SUBGRAPH"
                    )
                    edges.append({"data": e.model_dump()})
                    graph.tmp_edges = {}

            for e in graph.edges:
                # Subgraph internal edges.
                edges.append({"data": e.model_dump()})

            for e in graph.tmp_edges.values():
                # Subgraph to parent edges.
                edges.append({"data": e.model_dump()})

        for node in top_graph.nodes:
            if node.final:
                final_content = node.content
                final_content_long = node.long_content
                break

    # Call to render graph.
    color_cycle = st.session_state.border_color_cycle
    colors = st.session_state.dag_colors

    st_link_analysis(
        elements=elements,
        layout=GRAPH_LAYOUT,
        node_styles=style_nodes(colors, color_cycle, elements),
        edge_styles=EDGE_STYLES,
    )

    if final_content:
        st.session_state.final_content = final_content
        st.session_state.final_content_long = final_content_long
        time.sleep(RENDER_INTERVAL)

        for thread in st.session_state.threads:
            if thread.is_alive():
                thread.join()

        display_final_answer(final_content, final_content_long)


def render_graphs_text(_graphs: MultiGraphData):
    """Render the graph data in readable/text format."""
    if not _graphs.graphs:
        return

    lines = []

    for graph in _graphs.graphs:

        depth = graph.depth

        for n in graph.nodes:
            is_complex = " (complex)" if n.complex else ""
            is_final = ("✅ " if depth == 0 else "🟢 ") if n.final else "🔵 "
            content = n.long_content or n.content
            line = (
                f"<h4>{is_final}Node {n.id}: {n.name}{is_complex}</h4>"
                f"<div><font size=2>{content}</font></div>"
            )
            lines.append((depth, line))

    lines.sort(key=lambda x: -x[0])
    lines = [x[1] for x in lines]

    lines.insert(
        0, f"<font color='{FONT_COLOR}'>"
    )
    lines.append("</font>")

    html("\n".join(lines), scrolling=True, height=500)
