# -*- coding: utf-8 -*-
"""planner.py

Description

Todo:
    * everything
"""
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from nav.env import Environment

class Planner:
    def __init__(self, environment: Environment,
                       params: Dict[str, Any] = None): # algorithm parameters
        self.environment = environment
        self.params = params

    def set_waypoints(self, waypoints: np.ndarray, copy: bool):
        self.waypoints_ft = self.environment.ll_to_ft(waypoints, copy)
        self.waypoints_rc = self.environment.graph.to_graph(self.waypoints_ft, True)

    def to_graph(points_ft: np.ndarray, copy: bool):
        if copy:
            points_rc = _grid_round(np.copy(points_ft))
        else:
            points_rc = _grid_round(points_ft)

        # first try closest point in grid
        np.clip(points_rc[0], 0, self.env.graph.max_row, out=points_rc[0])
        np.clip(points_rc[1], 0, self.env.graph.max_col, out=points_rc[1])

        # TODO replace points not in graph

        return points_rc

    def _grid_round(a, granularity):
        return np.round(np.array(a, dtype=float) / granularity) * granularity

