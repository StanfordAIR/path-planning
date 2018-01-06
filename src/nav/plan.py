# -*- coding: utf-8 -*-
"""planner.py

Description

Todo:
    * everything
"""
from typing import List, Tuple, Dict, Any

Location = Tuple[int, int]
Path = List[Location]

import numpy as np
import matplotlib.pyplot as plt

from nav.env import Environment

class Planner:
    """ 
    def __init__(self, boundary: Path,
                       static_obs: Path,
                       params: Dict[str, Any] = None): # algorithm parameters
        self.environment = Environment(boundary, static_obs, params)
        self.params = params

    def set_waypoints(self, waypoints: Path):
        self.waypoints = waypoints

    def find_path(self, location: Location):



