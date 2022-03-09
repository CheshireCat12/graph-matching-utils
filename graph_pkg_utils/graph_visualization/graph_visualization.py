from graph_pkg_utils.graph_converter.graph_2_nx import convert_graph_2_nx
from pyvis.network import Network
from os.path import join


def visualize(graph, folder_result: str):
    nx_graph = convert_graph_2_nx(graph)
    nt = Network('1000px', '1000px')
    nt.from_nx(nx_graph)
    
    nt.show_buttons(filter_=['physics'])
    filename = join(folder_result,
                    f'{graph.name}.html')
    nt.save_graph(filename)

