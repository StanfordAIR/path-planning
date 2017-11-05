from typing import List

import networkx as nx

from utility_classes import Location

def graph_path(graph: nx.Graph, waypoints: List[Location]) -> np.ndarray:
    """Builds a path to the next waypoint(s)
    Args:
        graph: inside flight boundaries and outside obstacles
        waypoints: location waypoints to fly
    Returns:
        path: a list of locations corresponding to nodes in the graph path
    """
    return []
