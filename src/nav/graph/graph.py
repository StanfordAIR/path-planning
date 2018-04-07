"""
Graph class for in-time generated graphs for R^3 spaces. Should honestly be
    rewritten in C, but this will do for now.
"""
import numpy as np
from typing import Iterable, Tuple


class UniformGraph():
    def __init__(self, diag_a: np.ndarray, diag_b: np.ndarray, n_points: np.ndarray) -> None:
        """
        Generates a new UniformGraph object.

        Args:
            diag_a: Top-left coordinate of the diagonal of the containing box.
            diag_b: Bottom-right coordinate of the diagonal of the containing
                box.
            n_points: Number of points to sample in each dimension.
        """

        assert diag_a.shape[0] == diag_b.shape[0] == n_points.shape[0]

        self._n_dims = n_points.shape[0]
        self._origin = np.minimum(diag_a, diag_b)
        self._all_dimensions = np.maximum(diag_a, diag_b) - np.minimum(diag_a, diag_b)
        self._relative_sizes = self._all_dimensions/self.n_points

        self.exclusions = set()

    def set_exclusions(self, pos: Iterable) -> None:
        try:
            pos = iter(pos)
        except TypeError:
            print('pos object must be an iterable.')

        nn_pos = self.find_nearest(pos)
        self.exclusions.add(nn_pos)

    def find_nearest(self, pos: Iterable) -> Iterable:
        try:
            pos = iter(pos)
        except TypeError:
            print('pos object must be an iterable.')

        # TODO: Need to check if the output is actually inside of the box!

        # Index for a given dimension is round((x[i] - x_min[i])/delta[i])
        return (tuple(np.round((p - self._origin)/self._relative_sizes, decimals=1)) for p in pos)

    def neighbors(self, node: Tuple) -> Iterable:
        pass

    def neighbors_pos(self, pos: np.ndarray) -> Iterable:
        pass

    def dist(self, u: Tuple, v: Tuple) -> float:
        pass

    def to_rn(self, nodes: Iterable) -> Iterable:
        pass


