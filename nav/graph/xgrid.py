# -*- coding: utf-8 -*-
"""xgrid.py

This module handles generating the 2d space-filling grid.

Todo:

"""
import math

from networkx import Graph
from networkx.classes import set_node_attributes
from networkx.generators.classic import empty_graph

def xgrid_graph(col_count: int, row_count: int, granularity: float = None) -> Graph:
    """Builds and returns an x grid.
    Args:
        col_count: the number of columns in the lattice.
        row_count: the number of rows in the lattice.
    Returns:
        graph: an x grid with edge weights
    """
    graph = empty_graph(0)
    if col_count == 0 or row_count == 0:
        return graph

    cols = range(col_count)
    rows = range(row_count)
    grid_weight = 1.0
    diag_weight = math.sqrt(2 * grid_weight**2)
    # Make grid
    graph.add_edges_from((((c, r), (c + 1, r)) for r in rows
                                               for c in cols[:col_count]),
                         weight = grid_weight)
    graph.add_edges_from((((c, r), (c, r + 1)) for r in rows[:row_count]
                                               for c in cols),
                         weight = grid_weight)
    # add diagonals
    graph.add_edges_from((((c, r), (c + 1, r + 1)) for r in rows[:row_count]
                                                   for c in cols[:col_count]),
                         weight = diag_weight)
    graph.add_edges_from((((c + 1, r), (c, r + 1)) for r in rows[:row_count]
                                                   for c in cols[:col_count]),
                         weight = diag_weight)

    # Add position node attributes
    if granularity != None:
        pos = {node: (granularity * row, granularity * col) for (row, col) in graph}
        set_node_attributes(graph, pos, 'pos')

    return graph
