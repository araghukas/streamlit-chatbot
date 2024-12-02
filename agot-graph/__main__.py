"""AGoT graph visualizer"""

import itertools
import os
import threading
import time
from uuid import uuid4

import streamlit as st
from _events import add_agot_node
from _models import AnswerSchema, EdgeData, MultiGraphData
from _state import load_latest_state
from _styles import (BORDER_COLORS, EDGE_STYLES, FINAL_ANSWER_TEMPLATE,
                     style_nodes)
from agot import AGoT
from multi_agent_llm import OpenAILLM
from st_link_analysis import st_link_analysis

REFRESH_INTERVAL = 0.1  # seconds
RENDER_INTERVAL = 1  # seconds

GRAPHS = []

GRAPH_LAYOUT = "cose"


if "show_issues" not in st.session_state:
    st.session_state.show_issues = False

if "counter" not in st.session_state:
    st.session_state.counter = itertools.count(1)

if "state_counter" not in st.session_state:
    st.session_state.state_counter = itertools.count(1)

if "timestamp" not in st.session_state:
    st.session_state.timestamp = time.monotonic()

if "first_ever_run" not in st.session_state:
    st.session_state.first_ever_run = True

if "dag_colors" not in st.session_state:
    st.session_state.dag_colors = {}

if "border_color_cycle" not in st.session_state:
    st.session_state.border_color_cycle = itertools.cycle(BORDER_COLORS)

if "ai-input-box" not in st.session_state:
    st.session_state["ai-input-box"] = ""

if "agot-max-depth" not in st.session_state:
    st.session_state["agot-max-depth"] = 1

if "agot-num-layers" not in st.session_state:
    st.session_state["agot-num-layers"] = 3

if "agot-max-new-tasks" not in st.session_state:
    st.session_state["agot-max-new-tasks"] = 3

if "graph_hash" not in st.session_state:
    st.session_state.graph_hash = None

if "final_content" not in st.session_state:
    st.session_state.final_content = ""

if "final_content_long" not in st.session_state:
    st.session_state.final_content_long = ""

if "threads" not in st.session_state:
    st.session_state.threads = []


AGOT_SETTINGS = {
    "verbose": 0,
    "max_concurrent_tasks": 20,
}


def run_agot(
    *,
    _max_depth: int,
    _num_layers: int,
    _max_new_tasks: int,
    _graphs: MultiGraphData
):
    """AGoT initiator and runner"""

    for thread in st.session_state.threads:
        if thread.is_alive():
            thread.join()

    st.session_state.threads = []

    if not (_input := st.session_state.get("ai-input-box")):
        return

    _graphs.reset()
    st.session_state.colors = {}

    st.session_state.session_id = _session_id = str(uuid4())
    st.session_state.state_counter = _state_counter = itertools.count(1)


    openai_model = st.session_state.get("OPENAI-MODEL", "gpt-4o-mini")
    openai_api_key = st.session_state.get("OPENAI-API-KEY")

    os.environ["OPENAI_API_KEY"] = openai_api_key

    llm = OpenAILLM(openai_model)
    ai_runner = AGoT(
        llm=llm,
        log_callback=lambda *a, **kw: add_agot_node(
            *a,
            _graphs=_graphs,
            _state_counter=_state_counter,
            _session_id=_session_id,
            **kw,
        ),
        max_depth=_max_depth,
        max_num_layers=_num_layers,
        max_new_tasks=_max_new_tasks,
        **AGOT_SETTINGS,
    )
    runner_thread = threading.Thread(
        target=ai_runner.run,
        args=(_input,),
        kwargs={"schema": AnswerSchema},
    )
    runner_thread.start()  # start the thread and let it go
    st.session_state.threads.append(runner_thread)


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


_graphs = load_latest_state()
_graph_hash = hash(_graphs.model_dump_json())

st.set_page_config(
    layout="wide",
    page_title="AGoT Graph Visualizer",
)

with st.sidebar:

    st.text_input(
        "OpenAI API Key",
        key="OPENAI-API-KEY",
        placeholder="Enter your OpenAI API Key",
        type="password",
    )
    st.text_input(
        "OpenAI Model",
        key="OPENAI-MODEL",
        value="gpt-4o-mini",
    )

    st.markdown("## Graph Settings")

    st.slider(
        "limit depth",
        min_value=0,
        step=1,
        max_value=3,
        key="agot-max-depth"
    )
    st.slider(
        "limit layers",
        min_value=1,
        step=1,
        max_value=5,
        key="agot-num-layers"
    )
    st.slider(
        "limit tasks",
        min_value=1,
        step=1,
        max_value=10,
        key="agot-max-new-tasks"
    )
    st.button(
        label="show answer",
        on_click=lambda: display_final_answer(
            st.session_state.final_content,
            st.session_state.final_content_long,
        ),
        use_container_width=True,
    )

t = time.monotonic()

st.text_area(
    label="AI Input",
    label_visibility="collapsed",
    key="ai-input-box",
    placeholder="Enter your question here...",
    on_change=lambda: run_agot(
        _max_depth=st.session_state["agot-max-depth"],
        _num_layers=st.session_state["agot-num-layers"],
        _max_new_tasks=st.session_state["agot-max-new-tasks"],
        _graphs=_graphs,
    ),
    height=68,
)

# RENDERING
# ---------
if st.session_state.first_ever_run:
    st_link_analysis(elements={"nodes": [], "edges": []})
    st.session_state.first_ever_run = False

graph_changed = _graph_hash != st.session_state.graph_hash
if (
    (t - st.session_state.timestamp) >= RENDER_INTERVAL
    and graph_changed
):
    st.session_state.graph_hash = _graph_hash
    render_graphs(_graphs=_graphs)
    st.session_state.timestamp = t
# ---------

time.sleep(REFRESH_INTERVAL)

st.rerun()
