"""AGoT graph visualizer"""

import itertools
import os
import threading
import time
from uuid import uuid4

import streamlit as st
from agot import AGoT
from multi_agent_llm import OpenAILLM
from st_link_analysis import st_link_analysis

from agot_graph._events import add_agot_node
from agot_graph._models import AnswerSchema, MultiGraphData
from agot_graph._render import (RENDER_INTERVAL, display_final_answer,
                                display_sample_questions, render_graphs,
                                render_graphs_text)
from agot_graph._state import load_latest_state
from agot_graph._styles import BORDER_COLORS

REFRESH_INTERVAL = 0.1  # seconds

GRAPHS = []


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
    st.session_state["agot-max-depth"] = 0

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
    _graphs: MultiGraphData,
    _openai_api_key: str,
    _openai_model: str,
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

    os.environ["OPENAI_API_KEY"] = _openai_api_key

    llm = OpenAILLM(_openai_model)
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
        max_value=5,
        key="agot-max-new-tasks"
    )
    st.button(
        label="Show Answer",
        on_click=lambda: display_final_answer(
            st.session_state.final_content,
            st.session_state.final_content_long,
        ),
        use_container_width=True,
    )

t = time.monotonic()

_openai_api_key = st.session_state.get("OPENAI-API-KEY", "")
_openai_model = st.session_state.get("OPENAI-MODEL", "gpt-4o-mini")


def _run_agot_wrapper():

    if not _openai_api_key:
        st.session_state.raise_key_error = True
        return None

    run_agot(
        _max_depth=st.session_state["agot-max-depth"],
        _num_layers=st.session_state["agot-num-layers"],
        _max_new_tasks=st.session_state["agot-max-new-tasks"],
        _openai_api_key=_openai_api_key,
        _openai_model=_openai_model,
        _graphs=_graphs,
    )
    st.session_state.raise_key_error = False
    return None


if st.session_state.get("raise_key_error", False):
    st.error("Please enter your OpenAI API Key")
    st.session_state["ai-input-box"] = ""

left_1, right_1 = st.columns([6, 1])

with left_1:
    st.text_area(
        label="AI Input",
        label_visibility="collapsed",
        key="ai-input-box",
        placeholder="Enter your question here...",
        height=94,
    )
with right_1:
    st.button(
        label="Run AGoT",
        on_click=_run_agot_wrapper,
        use_container_width=True,
    )
    st.button(
        label="Show Samples",
        on_click=display_sample_questions,
        use_container_width=True,
    )

left_2, right_2 = st.columns([8, 5])

with left_2:
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

with right_2:
    render_graphs_text(_graphs=_graphs)

time.sleep(REFRESH_INTERVAL)

st.rerun()
