# path-planning
Complete path planning algorithm. Continuously generates a path for the autopilot to follow. The path is based on static and non-static information.

*Static information*
- flight boundary
- static obstacles

*Non-static information*
- waypoints
- moving obstacles
- current position

## Stages

### Built Initial Graph
Build a graph filling the flight boundary, with nodes inside static obstacles removed.

### Semi-Continuously Update Graph
Keep the graph up to date, with nodes within moving obstacles removed.

### Semi-Continuously Build Path Approximation
Build a path approximation using the current graph, from the current waypoint to the next few.

### Continuously Update Path
Smooth and optimize path based on current position and moving obstacles.
