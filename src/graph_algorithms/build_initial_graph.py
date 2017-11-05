from typing import List

import networkx as nx
import numpy as np

from utility_classes import Location
from utility_classes import Obstacle

def build_initial_graph(boundary: List[Location], stat_obstacles: List[Obstacle],
                        granularity: float) -> nx.Graph:
    """Builds a space-filling graph within flight boundaries and removes nodes in obstacles.
    Args:
        boundary: the gps locations of the polygon defining the flight boundary
        stat_obstacles: the locations and sizes of the stationary obstacles
        granularity: the distance between graph nodes (TODO in what measure?)
    Returns:
        graph: a graph filling the space, whose nodes are Locations
    """
    graph = nx.Graph()

    # TODO replace with space filling graph
    graph.add_nodes_from(boundary)
    for pair in zip(boundary, boundary[1:] + boundary[0:1]):
        graph.add_edge(pair[0], pair[1])

    # TODO remove nodes in obstacles

    return graph

# test if run as main
if __name__ == "__main__":
    boundary = [Location(0.0, 0.0), Location(1.1, 0.0), Location(1.1, 1.1)]
    stat_obstacles = []
    granularity = 0.0

    initial_graph = build_initial_graph(boundary, stat_obstacles, granularity)
    print(list(initial_graph.edges()))
