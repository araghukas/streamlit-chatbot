"""Pydantic models for the AGoT graph data"""
from typing import List, Optional

from pydantic import BaseModel, Field


class NodeData(BaseModel):
    """Render-relevant data for a single node in the graph"""
    id: str
    dag_id: str
    label: str
    layer: int
    content: str
    long_content: str
    name: str
    final: bool = False
    complex: bool = False

    def model_dump_render(self) -> dict:
        """"Dump the node data in render-appropriate format"""
        d = self.model_dump()
        d.pop("long_content", None)
        return d

class EdgeData(BaseModel):
    """Render-relevant data for a single edge in the graph"""
    id: str
    source: str
    target: str
    label: str


class GraphData(BaseModel):
    """Data for a single graph or subgraph"""
    graph_id: str
    depth: int
    nodes: list[NodeData]
    edges: list[EdgeData]
    tmp_edges: dict[str, EdgeData] = Field(default_factory=dict)
    parent_node_id: Optional[str] = None

    def export_data(self) -> dict:
        """Export the graph data in render-appropriate format"""
        return {
            "nodes": [{"data": n.model_dump_render()} for n in self.nodes],
            "edges": [{"data": e.model_dump()} for e in self.edges],
        }


class MultiGraphData(BaseModel):
    """Container for the global AGoT graphs"""

    graphs: List[GraphData]

    # Need this to record parent complex nodes of sub-graphs
    # before the subgraph is done evaluating.
    sub_graph_parents: dict[str, str] = Field(default_factory=dict)

    def add_node(self, graph_id: int | str, node: dict) -> None:
        """Add a node to the graph with the given ID"""
        graph = None
        _id = str(graph_id)
        for g in self.graphs:
            if g.graph_id == _id:
                graph = g
                break

        if graph is None:
            graph = GraphData(graph_id=_id)
            self.graphs.append(graph)

        graph.nodes.append(node)

    def get_graph(self, graph_id: int | str) -> Optional[GraphData]:
        """Get the graph with the given ID"""
        _id = str(graph_id)
        for g in self.graphs:
            if g.graph_id == _id:
                return g
        return None

    def set_graph(self, graph_id: int | str, graph: GraphData) -> None:
        """Set the graph with the given ID"""
        replaced = False
        _id = str(graph_id)
        for i, g in enumerate(self.graphs):
            if g.graph_id == _id:
                self.graphs[i] = graph
                replaced = True
                break

        if not replaced:
            self.graphs.append(graph)

    def get_node(self, node_id: int | str) -> Optional[NodeData]:
        """Get the node with the given ID"""
        # NOTE: requires GLOBALLY unique node_id's
        _id = str(node_id)
        for g in self.graphs:
            for n in g.nodes:
                if n.id == _id:
                    return n
        return None

    def get_edge(self, edge_id: int | str) -> Optional[EdgeData]:
        """Get the edge with the given ID"""
        _id = str(edge_id)
        for g in self.graphs:
            for e in g.edges:
                if e.id == _id:
                    return e
        return None

    def reset(self) -> None:
        """Wipe all graph data, reset to empty state"""
        self.graphs = []


class AnswerSchema(BaseModel):
    """Structure for the LLM's final answer"""
    short_content: str = Field(
        ...,
        description=(
            "The gist of the answer. "
            "Generally as short as possible, "
            "like a word, phrase, or sentence or two."
        ),
    )
    long_content: str = Field(
        ...,
        description=(
            "The full answer. "
            "Can be a few sentences or a paragraph."
            "This answer clearly articulates the short_answer as well."
        ),
    )
