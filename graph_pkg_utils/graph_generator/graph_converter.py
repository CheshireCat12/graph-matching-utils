import xml.dom.minidom as md
import xml.etree.ElementTree as ET
from collections import defaultdict
from os.path import join
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import torch_geometric
import torch_geometric.utils as tg_utils
from networkx.readwrite.graphml import write_graphml_lxml
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def convert_2_nx(graph: torch_geometric.data.Data) -> nx.Graph:
    """
    Convert the graph from torch_geometric.Data to nx with the correct
    formatting of the node features.

    Args:
        graph (torch_geometric.Data): the graph to convert

    Returns:
        nx.Graph: the converted graph
    """
    nx_graph = tg_utils.to_networkx(graph,
                                    node_attrs=['x'],
                                    to_undirected=True)

    # Convert the node feature vector into string.
    # The write_graphml() does not support lists/vectors.
    for node, d in nx_graph.nodes(data=True):
        for k, v in d.items():
            if isinstance(v, float):
                print('A float has to be changed is float', )
                v = [v] * graph.x.size(1)
            d[k] = str(v)

    return nx_graph


def save_graph(nx_reduced_graph: nx.Graph, idx: int, folder: str) -> None:
    """
    Write the graph in the given folder under 'gr_<idx>.graphml' filename.

    Args:
        graph (nx.Graph): graph to save
        idx (int): idx of the graph (used in the filename 'gr_<idx>.graphml')
        folder (str): folder where to solve the graph

    Returns:
        None
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    filename = join(folder, f'gr_{idx}.graphml')
    write_graphml_lxml(nx_reduced_graph,
                       filename,
                       infer_numeric_types=True)


def save_classes(graph_classes: List[Tuple[int, int]],
                 name_set: str,
                 folder: str) -> None:
    """
    Save the corresponding classes for each graph.

    Args:
        graph_classes (defaultdict):
            Dict containing the set of data as key and
             the value is the list of tuple containing
             the idx of the graph and its corresponding class.
        folder (str): folder where to save the classes

    Returns:
        None
    """
    graph_collection = ET.Element('GraphCollection')

    finger_prints = ET.SubElement(graph_collection, 'fingerprints')

    for idx_graph, class_ in graph_classes:
        print_ = ET.SubElement(finger_prints, 'print')
        print_.set('file', f'gr_{idx_graph}.graphml')
        print_.set('class', str(class_))

    b_xml = ET.tostring(graph_collection).decode()
    newxml = md.parseString(b_xml)

    Path(folder).mkdir(parents=True, exist_ok=True)
    filename = join(folder, f'{name_set}.cxl')
    with open(filename, mode='w') as f:
        f.write(newxml.toprettyxml(indent=' ', newl='\n'))
#
# from src.models.graph_u_net import GraphUNet

def convert_and_save(name_set: str,
                     dataset: torch_geometric.datasets,
                     indices: List[int],
                     folder: str,
                     format_: str='graphml') -> None:
    """Convert the graphs into the given format"""

    graph_classes = []

    for graph, idx in zip(dataset, indices):

        nx_reduced_graph = convert_2_nx(graph)

        save_graph(nx_reduced_graph, idx, folder)

        # Get the class value
        graph_classes.append((idx, int(graph.y.data[0])))

    sorted_graph_classes = sorted(graph_classes,
                                  key=(lambda x: x[0]))
    save_classes(sorted_graph_classes, name_set, folder)
