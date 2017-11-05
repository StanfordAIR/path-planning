# -*- coding: utf-8 -*-
"""xgrid.py

This module handles generating the 2d space-filling grid.

Todo:

"""
import math

from networkx import Graph
from networkx.classes import set_node_attributes
from networkx.generators.classic import empty_graph

def xgrid_graph(row_count: int, col_count: int,
                with_positions: bool = True) -> Graph:
    """Builds and returns an x grid.
    Args:
        row_count: the number of rows in the lattice.
        col_count: the number of columns in the lattice.
    Returns:
        graph: an x grid with edge weights
    """
    graph = empty_graph(0)
    if col_count == 0 or row_count == 0:
        return graph

    rows = range(row_count + 1)
    cols = range(col_count + 1)
    grid_weight = 1.0
    diag_weight = math.sqrt(grid_weight)
    # Make grid
    graph.add_edges_from((((i, j), (i + 1, j)) for j in rows
                                               for i in cols[:col_count]),
                         weight = grid_weight)
    graph.add_edges_from((((i, j), (i, j + 1)) for j in rows[:row_count]
                                               for i in cols),
                         weight = grid_weight)
    # add diagonals
    graph.add_edges_from((((i, j), (i + 1, j + 1)) for j in rows[:row_count]
                                                   for i in cols[:col_count]),
                         weight = diag_weight)
    graph.add_edges_from((((i + 1, j), (i, j + 1)) for j in rows[:row_count]
                                                   for i in cols[:col_count]),
                         weight = diag_weight)

    # Add position node attributes
    if with_positions:
        i_positions = (i for i in cols for j in rows)
        j_positions = (j for i in cols for j in rows)
        # switched row / column for display in (row, column) format
        pos = {(i, j): (j, i) for i, j in zip(i_positions, j_positions)
               if (i, j) in graph}
        set_node_attributes(graph, pos, 'pos')

    return graph
