from typing import List

import numpy as np
import networkx as nx

from utility_classes import Obstacle

def update_graph(initial_graph: nx.Graph, moving_obstacles: List[Obstacle]) -> nx.Graph:
    """
    Args:
        initial_graph: within flight boundaries and outside stationary obstacles
        moving_obstacles: a list of moving obstacles at their current positions
    Returns:
        graph: a graph within flight boundaries and outside all obstacles
    """
    return nx.Graph
