import networkx as nx
import numpy as np
import numpy.linalg as linalg
from math import floor, ceil

def quantize_path(path: list, rn_path: np.ndarray, quantization_distance: float) -> np.ndarray:
    """Quantizes a given graph path -> a path in Rn. Guaranteed that endpoints are included
    Args:
        path: Contains a list of vertex names in order of the path.
        rn_path: A numpy array of points in r3 of each vertex in path, where the columns are each point.
        quantization_distance: the distance between every two points in the quantization path.
    """
    if rn_path.shape[1] <= 1:
        raise ValueError('A path must have at least two points')
    
    dist_mat = linalg.norm(rn_path[:,:-1] - rn_path[:,1:], axis=0)
    total_norm = sum(dist_mat)
    
    if total_norm <= quantization_distance:
        raise ValueError('A path\'s total distance must be larger than the quantization distance')

    curr_path_len = 0 # ∆(dist) between current quantization path and last node explored

    full_path = np.zeros((rn_path.shape[0], int(ceil(total_norm/quantization_distance))))
    curr_idx = 0

    for v_idx, edge_len in enumerate(dist_mat):
        curr_vertex, next_vertex = rn_path[:,v_idx], rn_path[:,v_idx+1]
        conv_comb = 0
        n_iter = int(floor((edge_len - curr_path_len)/quantization_distance))
        
        for _ in range(n_iter):
            conv_comb += quantization_distance/edge_len
            assert conv_comb <= 1
            full_path[:,curr_idx] = conv_comb*curr_vertex + (1-conv_comb)*next_vertex
        
        curr_path_len += edge_len - n_iter*quantization_distance
        
        if curr_path_len > quantization_distance:
            curr_path_len -= int(floor(curr_path_len/quantization_distance))*quantization_distance

    return full_path
