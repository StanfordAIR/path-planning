# -*- coding: utf-8 -*-
"""graph.py
Builds the initial graph, within the flight boundary_ft and outside obstacles.
TODO:
    Remove nodes outside non-rectangular flight boundary_ft
"""
from nav.graph.xgrid import xgrid_graph
from nav.graph.polygon import Polygon

from typing import Dict, Any

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class FlightGraph:
    """#TODO"""
    def __init__(self, boundary_ft: np.ndarray, static_obs_ft: np.ndarray,
                 granularity: float):
        """Builds a space-filling graph within flight boundaries and removes
        nodes in obstacles.

        Args:
            boundary_ft: the ft locations of the polygon defining the flight boundary
            static_obs_ft: the locations and sizes of the stationary obstacles
            granularity: ft separating graph nodes

        Todo:
            * ...
        """
        # space filling graph
        self.granularity = granularity
        lat_size_ft = np.amax(boundary_ft[0])
        lon_size_ft = np.amax(boundary_ft[1])
        lat_node_count = int(lat_size_ft // granularity) # why not + 1 ?
        lon_node_count = int(lon_size_ft // granularity) # why not + 1 ?
        base_graph = xgrid_graph(lat_node_count, lon_node_count)
#
#        # remove nodes in obstacles and outside boundary
#        rows = np.arange(lat_node_count)
#        cols = np.arange(lon_node_count)
#        nodes_rc = np.array([np.tile(rows, len(cols)), np.repeat(cols, len(rows))])
#        nodes_ft = self.granularity * nodes_rc
#        node_idx_to_remove = set() # all nodes to remove
#
#        # find nodes in obstacles
#        for obs_idx in range(static_obs_ft.shape[1]):  #TODO vectorize out
#            dist_sqr = np.sum((nodes_ft - static_obs_ft[0:2, obs_idx, np.newaxis]) \
#                              ** 2, axis=0)
#            node_idx_inside_obs = np.flatnonzero(dist_sqr \
#                                            <= (static_obs_ft[2, obs_idx] ** 2))
#            node_idx_to_remove.update(node_idx_inside_obs.tolist())
#
#        # find nodes outside boundary #TODO vectorize
#        self.polygon = Polygon(boundary_ft)
#        for node_idx in range(nodes_ft.shape[1]):
#            lat = nodes_ft[0, node_idx]
#            lon = nodes_ft[1, node_idx]
#            if (lat, lon) not in self.polygon:
#                node_idx_to_remove.add(node_idx)
#        
#        # remove nodes
#        for node_idx in node_idx_to_remove:
#            base_graph.remove_node((nodes_rc[0, node_idx],
#                                         nodes_rc[1, node_idx]))
#

        self.base_graph = base_graph
        self.base_nodes_rc = np.array(list(base_graph)).T
        self.base_nodes_ft = self.base_nodes_rc * self.granularity
