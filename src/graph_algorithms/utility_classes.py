# -*- coding: utf-8 -*-
"""quantize_path.py

Description

Todo:
    * Add todo
"""
import typing
from collections import namedtuple

Location = typing.NamedTuple('Location', [('lat', float), ('lon', float)])
Obstacle = typing.NamedTuple('Obstacle', [('location', Location), ('radius', float)])
