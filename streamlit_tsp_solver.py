import json
import pickle
from typing import List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from PIL import Image

TSP_URL = st.secrets["TSP_URL"]
TSP_TOKEN = st.secrets["TSP_TOKEN"]

APP_DESCRIPTION = "Obtain a solution to the Traveling Salesman Problem (TSP) \
    across 312 major cities in the United States and Canada. This app parses \
    general news information into travel delays and restrictions using an AI agent. \
    The agent's output then informs a TSP solver via edge weight adjustments that \
    determine the shortest path through all 312 cities."

DEFAULT_SENSITIVITY = 2

THEME = st.session_state.get("map_style", "dark")
MAP_STYLES = {
    "dark": "dark_no_labels",
    "light": "light_no_labels",
}

SCATTER_COLOR = {
    "dark": [255, 255, 255, 160],
    "light": [255, 0, 0, 160],
}

BG_COLOR = {
    "dark": "green",
    "light": "red",
}


###########################
# LOAD FILES AND OTHER DATA
###########################
if st.session_state.get("cities_data") is None:
    data = pd.read_csv("./usca312/usca312_geo.csv")
    st.session_state["cities_data"] = data
    st.session_state["cities_index"] = dict(zip(data["city"], data.index))

LOGO = Image.open("./app_assets/logo.png")

if "default_result" not in st.session_state:
    with open("./default-result.pkl", "rb") as f:
        st.session_state["default_result"] = pickle.load(f)


##################
# HELPER FUNCTIONS
##################
def get_usca312_tsp_solution() -> None:
    """Request a solution from the Covalent TSP solver service"""
    address = st.session_state.get("server_address", TSP_URL)
    token = st.session_state.get("api_key", TSP_TOKEN)

    info = st.session_state["info"]
    inputs = {"info": info} if info else None
    url = address + "/solve"
    print(f"Requesting solution from {url}")
    print(f"Token: ...{token[-5:]}")
    print(f"Inputs: {inputs}")
    with st.spinner("Generating delays and solution..."):
        try:
            response = requests.post(
                url,
                headers={"x-api-key": token},
                json=inputs,
                timeout=600
            )
            response.raise_for_status()
            st.session_state["result"] = response.json()
        except Exception as e:  # pylint: disable=broad-except
            st.error(e)


def get_layers() -> List[pdk.Layer]:
    layers = []
    if "result" in st.session_state and st.session_state.get("show_optimal_path"):
        layers.append(
            pdk.Layer(
                type="PathLayer",
                data=convert_tour_indices_to_path(),
                pickable=True,
                get_color="color",
                width_scale=20,
                width_min_pixels=1,
                get_path="path",
                get_width=1,
            )
        )
    if st.session_state.get("show_default_path"):
        layers.append(
            pdk.Layer(
                type="PathLayer",
                data=convert_tour_indices_to_path(default=True),
                pickable=True,
                get_color=(0, 0, 255),
                width_scale=20,
                width_min_pixels=1,
                get_path="path",
                get_width=1,
            )
        )

    if "result" in st.session_state and st.session_state.get("show_issues"):
        layers.append(
            pdk.Layer(
                type="PathLayer",
                data=convert_issues_to_paths(),
                pickable=True,
                get_color="color",
                width_scale=20,
                width_min_pixels=1,
                get_path="path",
                get_width=1,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=st.session_state["cities_data"],
            pickable=True,
            get_position="[lon, lat]",
            get_color=SCATTER_COLOR[THEME],
            get_radius=10_000,
        )
    )
    return layers


def convert_tour_indices_to_path(default=False) -> pd.DataFrame:
    if default:
        print("loading default result")
        tour = st.session_state["default_result"]["result"][0]["tour"]
    else:
        result_index = st.session_state.get("sensitivity", DEFAULT_SENSITIVITY) - 1
        print(f"loading result index {result_index}")
        result = st.session_state["result"]["result"][result_index]
        tour = result["tour"]

    # Dataframe columns
    names: List[str] = []
    paths: List[List[List[float]]] = []
    colors: List[Tuple[int, int, int]] = []

    # Process solution path
    path = []
    cities_data = st.session_state["cities_data"]
    for city_index in tour:
        entry = cities_data.iloc[city_index]
        path.append([entry["lon"], entry["lat"]])

    # Make circular
    path.append(path[0])

    paths.append(path)
    colors.append((0, 232, 0))
    names.append("Optimal Path")

    return pd.DataFrame({"name": names, "color": colors, "path": paths})


def convert_issues_to_paths() -> pd.DataFrame:
    result_index = st.session_state.get("sensitivity", DEFAULT_SENSITIVITY) - 1
    print(f"loading result index {result_index}")
    result = st.session_state["result"]["result"][result_index]
    cities_data = st.session_state["cities_data"]

    # Dataframe columns
    names: List[str] = []
    paths: List[List[List[float]]] = []
    colors: List[Tuple[int, int, int]] = []

    # Highlight issue paths
    index = st.session_state["cities_index"]
    for issue in result["issues"]:
        cities = issue["cities"]
        if any(c not in index for c in cities):
            continue

        indices = [index[city] for city in cities]
        issue_path: List[List[float]] = [
            [cities_data.loc[i]["lon"], cities_data.loc[i]["lat"]]
            for i in indices
        ]
        paths.append(issue_path)
        colors.append((255, 0, 0))
        names.append(" - ".join(cities))

    return pd.DataFrame({"name": names, "color": colors, "path": paths})


def render_issues() -> Optional[str]:
    result_index = st.session_state.get("sensitivity", DEFAULT_SENSITIVITY) - 1
    if not (result := st.session_state.get("result")):
        return None

    issues = result["result"][result_index]["issues"]
    index = st.session_state["cities_index"]

    _issues = []
    for issue in issues:
        cities = issue["cities"]
        if any(c not in index for c in cities):
            continue
        _issues.append(issue)
    return json.dumps(_issues)


#########################
# MAIN APP IMPLEMENTATION
#########################
st.set_page_config(layout="wide", page_title="AI-Assisted Routing")
st.title("📍 AI-Assisted Routing", help=APP_DESCRIPTION)

left, middle, right = st.columns([2, 3, 1])
with left:
    # TEXT AREA
    st.text_area(
        label="text area",
        placeholder="Enter information to infer travel delays from...",
        key="info",
        height=470,
        label_visibility="hidden",
        on_change=get_usca312_tsp_solution,
    )

with middle:
    # MAP SECTION
    default_view = pdk.ViewState(
        latitude=st.session_state["cities_data"]["lat"].mean(),
        longitude=st.session_state["cities_data"]["lon"].mean() - 10,
        pitch=0,
        zoom=3,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style=MAP_STYLES[THEME],
            initial_view_state=default_view,
            layers=get_layers(),
            tooltip={
                "text": "{city}",
                "style": {"backgroundColor": BG_COLOR[THEME]}
            },
        )
    )
with right:
    # MAP SETTINGS
    st.toggle(label="Issues", value=True, key="show_issues", help="show issue edges inferred by the AI agent")
    st.toggle(label="Optimal Path", value=True, key="show_optimal_path", help="show the shortest path with issues considered")
    st.toggle(label="Default Path", value=False, key="show_default_path", help="show the default shortest path")
    st.container(height=118, border=False)
    st.slider(
        label="Issue Sensitivity",
        value=DEFAULT_SENSITIVITY,
        min_value=1,
        max_value=4,
        step=1,
        format="%d",
        key="sensitivity",
        help="Base sensitivity to issues extracted by the AI agent.",
    )
    st.selectbox(label="Map Style", options=list(MAP_STYLES), key="map_style")


# ISSUES JSON
if issues_string := render_issues():
    with st.expander(label="AI-extracted issues", expanded=True, icon="🚧"):
        st.json(issues_string)


st.text_input(
    label="Server Address",
    value=TSP_URL,
    key="server_address"
)
st.text_input(
    label="API Key",
    value=TSP_TOKEN,
    key="api_key",
    type="password"
)

st.caption("powered by")
st.image(LOGO, output_format="PNG")
