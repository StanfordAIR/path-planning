# -*- coding: utf-8 -*-
"""initial.py
Builds the initial graph, within the flight boundary_ft and outside obstacles.
TODO:
    Remove nodes outside non-rectangular flight boundary_ft
"""
from nav.graph.xgrid import xgrid_graph

from nav.utility.classes import Location
from nav.utility.classes import Obstacle

from typing import Dict, Any

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def build_graph(boundary_ft: np.ndarray, static_obs_ft: np.ndarray,
                granularity: float) -> nx.Graph:
    """Builds a space-filling graph within flight boundaries and removes nodes in obstacles.
    Args:
        boundary_ft: the ft locations of the polygon defining the flight boundary
        static_obs_ft: the locations and sizes of the stationary obstacles
        granularity: ft separating graph nodes
    Returns:
        graph: a graph filling the space, whose nodes are Locations
    """
    # space filling graph
    lat_node_count = int(np.amin(boundary_ft[0]) // granularity)
    lon_node_count = int(np.amin(boundary_ft[1]) // granularity)
    graph = xgrid_graph(lat_node_count, lon_node_count)

    # removes nodes in obstacles #TODO vectorize
    to_remove = set()
    for obs_idx in range(static_obs_ft.shape[1]):
        for node in graph:
            distance = static_obs_ft[0:2, obs_idx] - np.array([[node[0]], [node[1]]])
            r = static_obs_ft[2]
            if np.sqrt(np.sum(distance**2)) < r:
                to_remove.add(node)
    for node in to_remove:
        graph.remove_node(node)

    # TODO remove nodes outside boundary_ft

    return graph
