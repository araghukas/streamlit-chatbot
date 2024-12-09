"""Helper functions"""
import itertools
import json
import os
import time
from pathlib import Path

import streamlit as st
from agot_graph._models import MultiGraphData

CWD = Path(__file__).parent.absolute()

STATES_DIR = CWD / "states"
if not STATES_DIR.exists():
    STATES_DIR.mkdir()


def load_latest_state() -> MultiGraphData:
    """Load the latest state from the state files."""
    sid = st.session_state.get("session_id", None)

    def _load_helper(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    if sid is not None:
        state_files = sorted(
            STATES_DIR.glob(f"nodes_state_{sid}_*.json"),
            key=os.path.getmtime
        )
        if state_files:
            load_attempts = 0
            while load_attempts < 5:
                try:
                    return MultiGraphData.model_validate(
                        _load_helper(STATES_DIR / state_files[-1])
                    )
                except json.JSONDecodeError:
                    print("State file load error, retrying...")
                    load_attempts += 1
                    time.sleep(0.1)


    return MultiGraphData(graphs=[])


def clean_up_files(t_mins: float = 15.0) -> None:
    """Delete old state files."""
    state_files = list(STATES_DIR.glob("nodes_state*.json"))
    # delete files older than 15 minutes
    for f in state_files:
        if time.time() - os.path.getmtime(f) > t_mins * 60:
            os.remove(f)


def update_state_file(
    _graphs: MultiGraphData,
    _state_counter: itertools.count,
    _session_id: str,
) -> None:
    """Write the current state to a file."""
    i = next(_state_counter)
    sid = _session_id
    target = STATES_DIR / f"nodes_state_{sid}_{i}.json"
    with open(target, "w", encoding="utf-8") as f:
        f.write(_graphs.model_dump_json())
    clean_up_files()
