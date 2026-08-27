from __future__ import annotations

from cgr_exp.graph.edges import GraphEdge

def build_tree_edges(nodes):
    node_ids = {n.node_id for n in nodes}
    edges = []
    for node in nodes:
        if node.parent_id is not None and node.parent_id in node_ids:
            edges.append(GraphEdge(source=node.parent_id, target=node.node_id, edge_type="parent_child", reason="node parent_id links to expanded parent"))
    return edges
