import networkx as nx
import numpy as np

def convert_graph_2_nx(graph):
    """
    Convert the Graph from the graph_pkg_core to a networkX graph

    :param graph:
    :return:
    """
    nx_graph = nx.Graph()

    # Add nodes and their corresponding lbls value
    for node in graph.get_nodes():
        lbl = np.array(node.label.vector).tolist()
        nx_graph.add_node(node.idx, lbl=lbl)

    # Add edges
    for idx_edge, edges in graph.get_edges().items():
        for edge in edges:
            if edge is not None:
                nx_graph.add_edge(edge.idx_node_start,
                                  edge.idx_node_end)

    return nx_graph
