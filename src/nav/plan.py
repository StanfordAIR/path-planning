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
