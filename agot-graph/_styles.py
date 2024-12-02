"""Render style definitions"""

from st_link_analysis import EdgeStyle, NodeStyle

FINAL_ANSWER_TEMPLATE = (
    """
{content}

---

<details>
{long_content}
<br>
<span style="color: grey; font-style: italic">
graph backend: '{model}'
</span>
</details>
"""
)

BORDER_COLORS = [
    '#ffe119',
    '#f58231',
    '#42d4f4',
    '#fabed4',
    '#aaffc3',
    '#5885AF',
]

_NODE_COLORS = {
    "ANSWER": "#3cb44b",
    "COMPLEX-ANSWER": "#3cb44b",
    "ANSWER-SUB": "#3cb44b",
    "COMPLEX": "#ffffff",
    "THOUGHT": "#ffffff",
    "THOUGHT-SUB": "#ffffff",
}

_NODE_SHAPES = {
    "THOUGHT": "ellipse",
    "THOUGHT-SUB": "ellipse",
    "ANSWER": "ellipse",
    "ANSWER-SUB": "ellipse",
    "COMPLEX": "hexagon",
    "COMPLEX-ANSWER": "hexagon",
}

NODE_SIZE = 10
NODE_SIZE_SUB = 5
BORDER_WIDTH = 4
BORDER_WIDTH_SUB = 2

_NODE_STYLE_KWARGS = {
    "THOUGHT": {
        "caption": "name",
        "border_width": BORDER_WIDTH,
        "size": NODE_SIZE,
    },
    "ANSWER": {
        "caption": "name",
        "border_width": BORDER_WIDTH,
        "size": NODE_SIZE,
    },
    "COMPLEX": {
        "caption": "name",
        "border_width": BORDER_WIDTH,
        "size": NODE_SIZE,
    },
    "COMPLEX-ANSWER": {
        "caption": "name",
        "border_width": BORDER_WIDTH,
        "size": NODE_SIZE,
    },
    "THOUGHT-SUB": {
        "caption": "name",
        "border_width": BORDER_WIDTH_SUB,
        "size": NODE_SIZE_SUB,
    },
    "ANSWER-SUB": {
        "caption": "name",
        "border_width": BORDER_WIDTH_SUB,
        "size": NODE_SIZE_SUB,
    },
}


class CustomNodeStyle(NodeStyle):
    """Sub-class of NodeStyle to include more parameters."""

    def __init__(self, *args, **kwargs):
        self._size = kwargs.pop("size")
        self._border_width = kwargs.pop("border_width")
        self._border_color = kwargs.pop("border_color")
        self._shape = kwargs.pop("shape")
        super().__init__(*args, **kwargs)

    def dump(self) -> dict:
        d = super().dump()
        style = d["style"]

        style["height"] = style["width"] = self._size

        if self._border_width is not None:
            style["border-width"] = self._border_width

        if self._border_color is not None:
            style["border-color"] = self._border_color

        if self._shape is not None:
            style["shape"] = self._shape

        return d


def _style_node_color(_colors, _color_cycle, dag_id):
    if dag_id in _colors:
        return _colors[dag_id]

    color = next(_color_cycle)
    _colors[dag_id] = color
    return color


def style_nodes(_colors, _color_cycle, elements) -> CustomNodeStyle:
    """Get a node style based on the node type."""
    nodes = elements["nodes"]
    styles = []

    for n in nodes:

        label = n["data"]["label"]
        dag_id = n["data"]["dag_id"]
        border_color = _style_node_color(_colors, _color_cycle, dag_id)

        style_kwargs = _NODE_STYLE_KWARGS[label]
        color = _NODE_COLORS[label]
        shape = _NODE_SHAPES[label]
        style = CustomNodeStyle(
            label=label,
            color=color,
            border_color=border_color,
            shape=shape,
            **style_kwargs,
        )
        styles.append(style)

    return styles


class CustomEdgeStyle(EdgeStyle):
    """Sub-class of EdgeStyle to include more parameters."""

    def __init__(
        self,
        *args,
        width: float = 1,
        line_style: str = 'solid',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.width = width
        self.line_style = line_style

    def dump(self) -> dict:
        d = super().dump()
        style = d["style"]

        style["width"] = self.width
        style["line-style"] = self.line_style
        style["opacity"] = 0.5
        return d


ST_COLORS = {
    "st-white": "#FAFAFA",
    "st-red": "#F64948",
    "st-grey": "#25262F",
}

EDGE_STYLES = [
    CustomEdgeStyle(
        label="NORMAL",
        width=0.5,
        directed=True,
        color=ST_COLORS["st-white"],
    ),
    CustomEdgeStyle(
        label="SUBGRAPH",
        width=0.25,
        line_style="dashed",
        color=ST_COLORS["st-white"],
    ),
    CustomEdgeStyle(
        label="SUBGRAPH-HIDDEN",
        width=0.0,
        directed=False,
    ),
]
