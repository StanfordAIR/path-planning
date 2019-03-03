from nav.graph.initial import build_graph
from nav.graph.path import plan_graph_path
from nav.graph.path import quantize_graph_path
from nav.path.optimize import optimize_path
from nav.utility.classes import Location, Obstacle

import networkx as nx
import matplotlib.pyplot as plt

def point_to_point(boundary, waypoints, stat_obstacles, params, display=False, verbose=True):
    # Parameters
    granularity = params["granularity"]
    quantization_distance = params["quantization_distance"]
    
    ### Script
    if display:
        fig, ax = plt.subplots(1, 3, figsize=(12,4))
    
    # Graph path
    initial_graph, origin_loc = build_graph(boundary, stat_obstacles, granularity)
    graph_path = plan_graph_path(waypoints, initial_graph, origin_loc, params)
    
    pos = nx.get_node_attributes(initial_graph, 'pos')
    if display:
        nx.draw(initial_graph, pos, nodelist=graph_path, font_weight='bold', ax=ax[0])
    
    # Quantized Path
    quantized_path = quantize_graph_path(graph_path, origin_loc, params)
    if display:
        ax[1].axis((-50.0,150.0,-50.0,150.0))
        ax[1].plot(quantized_path[0], quantized_path[1], '.')
        for obs in stat_obstacles:
            circle = plt.Circle((obs.location.lat, obs.location.lon),
                                obs.radius, color='r', fill=False)
            ax[1].add_artist(circle)
    
    # Flight Path
    flight_path = optimize_path(quantized_path, boundary, stat_obstacles, params, verbose=verbose)

    if display:
        ax[2].axis((-50.0, 150.0, -50.0, 150.0)) #TODO replace hardcoded values
        ax[2].plot(flight_path[0], flight_path[1], '.')
        for obs in stat_obstacles:
            circle = plt.Circle((obs.location.lat, obs.location.lon),
                                obs.radius, color='r', fill=False)
            ax[2].add_artist(circle)
        
        plt.show()

    return flight_path
