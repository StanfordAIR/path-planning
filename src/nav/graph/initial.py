# -*- coding: utf-8 -*-
"""build_initial_graph.py
Builds the initial graph, within the flight boundary and outside obstacles.
Todo:
    * Add flight boundary check with polygon
"""
from .xgrid import xgrid_graph

from .classes import Location
from .classes import Obstacle

from typing import List
from typing import Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def build(boundary: List[Location], stat_obstacles: List[Obstacle],
          granularity: float) -> nx.Graph:
    """Builds a space-filling graph within flight boundaries and removes nodes in obstacles.
    Args:
        boundary: the gps locations of the polygon defining the flight boundary
        stat_obstacles: the locations and sizes of the stationary obstacles
        granularity: the distance between graph nodes for (lat, lon)
    Returns:
        graph: a graph filling the space, whose nodes are Locations
    """
    # space filling graph
    min_boundary = Location(*(min(coord) for coord in zip(*boundary)))
    max_boundary = Location(*(max(coord) for coord in zip(*boundary)))
    graph_size = Location.granularity_diff(max_boundary, min_boundary, granularity)
    
    graph = xgrid_graph(*graph_size)

    # TODO remove nodes in obstacles
    graph_origin = min_boundary
    for obs in stat_obstacles:
        to_remove = set()
        for node in graph:
            loc = Location.from_grid(node, graph_origin, granularity)
            if loc in obs:
                to_remove.add(node)
        for node in to_remove:
            graph.remove_node(node)

    # TODO return mapping from location to node, node to location?

    return graph, graph_origin

# test if run as main
if __name__ == "__main__":
    boundary = [Location(-50.0, -50.0), Location(-50.0, 150.0),
                Location(150.0, -50.0), Location(150.0, 150.0)]
    stat_obstacles = [Obstacle(Location(0.0, 0.0), 20),
                      Obstacle(Location(30.0, 30.0), 10)]
    granularity = 20.0

    initial_graph, _ = build(boundary, stat_obstacles, granularity)
    plt.figure()
    pos = nx.get_node_attributes(initial_graph, 'pos')
    nx.draw(initial_graph, pos, with_labels=True, font_weight='bold')
    plt.show()
