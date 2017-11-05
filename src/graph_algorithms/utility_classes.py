import typing
from collections import namedtuple

Location = typing.NamedTuple('Location', [('lon', float), ('lat', float)])
Obstacle = typing.NamedTuple('Obstacle', [('location', Location), ('radius', float)])
