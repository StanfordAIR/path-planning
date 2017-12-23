# -*- coding: utf-8 -*-
"""initial.py
Builds the initial graph, within the flight boundary and outside obstacles.
TODO:
    Remove nodes outside non-rectangular flight boundary
"""
from nav.graph.xgrid import xgrid_graph

from nav.utility.classes import Location
from nav.utility.classes import Obstacle

from typing import List
from typing import Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def build_graph(boundary: List[Location], stat_obstacles: List[Obstacle],
                params) -> nx.Graph:
    """Builds a space-filling graph within flight boundaries and removes nodes in obstacles.
    Args:
        boundary: the gps locations of the polygon defining the flight boundary
        stat_obstacles: the locations and sizes of the stationary obstacles
        params: algorithm parameters dictionary
    Returns:
        graph: a graph filling the space, whose nodes are Locations
    """
    # space filling graph
    min_boundary = Location(*(min(coord) for coord in zip(*boundary)))
    max_boundary = Location(*(max(coord) for coord in zip(*boundary)))
    graph_size = Location.granularity_diff(max_boundary, min_boundary, params["granularity"])
    
    graph = xgrid_graph(*graph_size)

    # removes nodes in obstacles
    graph_origin = min_boundary
    for obs in stat_obstacles:
        to_remove = set()
        for node in graph:
            loc = Location.from_grid(node, graph_origin, params["granularity"])
            if loc in obs:
                to_remove.add(node)
        for node in to_remove:
            graph.remove_node(node)

    # TODO remove nodes outside boundary

    return graph, graph_origin
