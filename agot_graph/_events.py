"""AGoT event handlers"""
import itertools

from agot_graph._models import EdgeData, GraphData, MultiGraphData, NodeData, AnswerSchema
from agot_graph._state import update_state_file

EVENT_ICONS = {
    "add_task": "🔵",
    "answer": "🟢",
    "is_complex": "🔴",
    "update_complex": "✅",
    "new_subgraph": "🟡",
}


def add_agot_node(
    event_name: str,
    *,
    _graphs: MultiGraphData,
    _state_counter: itertools.count,
    _session_id: str,
    **kwargs
):
    """Add a node to the graph based on an event from running AGoT."""
    node_id = str(kwargs["node_id"])
    depth = kwargs.get("depth", -1)
    layer = kwargs.get("layer", -1)

    dag = kwargs.get("dag", None)
    dag_id = str(id(dag))
    node_dag = dag.nodes.get(int(node_id), None)
    graph = _graphs.get_graph(dag_id)

    content = long_content = ""

    if node_dag is not None:
        content = node_dag.get("answer", "")
        if isinstance(content, AnswerSchema):
            content, long_content = content.short_content, content.long_content
        elif hasattr(content, "content"):
            content = long_content = content.content

    if node_id is None:
        raise RuntimeError(
            f"Node ID not found in event ['{event_name}'] : {kwargs}")

    node = _graphs.get_node(node_id)
    node_exists = node is not None

    current_label = getattr(node, "label", 'None')
    print(
        f"NODE{'[' + node_id + ']':<2}: {EVENT_ICONS[event_name]} {event_name.upper()} "
        f"-> dag_id={dag_id} with node.label={current_label}"
    )

    # A new AGoT node was added.
    # Add this node to the rendered graph.
    if event_name == "add_task":

        if node_exists:
            raise RuntimeError(f"Node {node_id} already exists in graph")

        if graph is None:
            if depth < 0:
                raise RuntimeError(
                    "Can't create a new graph without knowing its depth"
                )
            new_graph = GraphData(
                graph_id=dag_id,
                depth=depth,
                nodes=[],
                edges=[],
            )
            _graphs.set_graph(dag_id, new_graph)
            graph = new_graph

        graph.nodes.append(
            NodeData(
                id=str(node_id),
                dag_id=dag_id,
                label="THOUGHT" if depth == 0 else "THOUGHT-SUB",
                layer=layer,
                content=content,
                long_content=long_content,
                name=node_dag["title"] if node_dag else "",
            )
        )
        # update all edges
        for u, v in dag.edges:
            graph.edges.append(
                EdgeData(
                    id=f"{u}-{v}",
                    source=str(u),
                    target=str(v),
                    label="NORMAL"
                )
            )

        # for subgraphs, add temporary edges fro all nodes to parent node
        parent_node_id = _graphs.sub_graph_parents.get(dag_id)
        if parent_node_id is not None:
            graph.tmp_edges[node_id] = EdgeData(
                id=f"{node_id}-parent",
                source=str(node_id),
                target=parent_node_id,
                label="SUBGRAPH-HIDDEN"
            )

        update_state_file(_graphs, _state_counter, _session_id)
        return None

    if not node_exists:
        raise RuntimeError(f"Node {node_id} not found in graph")

    # Update an existing node in the graph. Complex nodes don't pass through here.
    if event_name == "answer":

        node.label = "ANSWER" if depth == 0 else "ANSWER-SUB"
        node.dag_id = dag_id
        node.content = content if content else node.content
        node.long_content = long_content if long_content else node.long_content
        node.final = kwargs.get("final", False)

    elif event_name == "is_complex":  # mark a node as complex for special rendering

        node.label = "COMPLEX"
        node.dag_id = dag_id
        node.complex = True

    elif event_name == "new_subgraph":  # create a new subgraph for a complex node

        _graphs.sub_graph_parents[dag_id] = str(node_id)

    elif event_name == "update_complex":  # update a complex node and its subgraph

        sub_dag = kwargs.get("sub_dag", None)
        if sub_dag is None:
            raise RuntimeError(f"Subgraph not found for node {node_id}")

        sub_dag_id = id(sub_dag)
        sub_graph = _graphs.get_graph(sub_dag_id)

        if sub_graph is None:
            raise RuntimeError(
                f"Subgraph {sub_dag_id} not found for node {node_id}"
            )
        sub_graph.parent_node_id = str(node_id)

        if node.complex:
            node.label = "COMPLEX-ANSWER"
        elif depth == 0:
            node.label = "ANSWER"
        else:
            node.label = "ANSWER-SUB"

        node.dag_id = dag_id
        node.content = content if content else node.content
        node.long_content = long_content if long_content else node.long_content

    update_state_file(_graphs, _state_counter, _session_id)
    return None
