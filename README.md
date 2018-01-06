# path-planning
Complete path planning algorithm. Continuously generates a path for the autopilot to follow. The path is based on static and non-static information.

*Static information*
- flight boundary
- static obstacles

*Non-static information*
- desired waypoints
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

## Location Abstraction Levels

### Latitude/Longitude
Locations are represented as (latitude, longitude), where north latitude is positive and west longitude is negative.

### Area grid in feet
A grid measured from a point of the minimum latitude and longitude in the flight boundary, in (feet north, feet east). An approximate conversion of latitude and longitude to feet at the flight latitude is used. This allows for simpler manipulation of distances as obstacle sizes are given in feet and since latitude and longitude are not comparable.

### Row, column in graph
The location of a point in (row, column) of the graph filling the flight space. Latitude corresponds to rows while longitude corresponds to the column.

## Navigator Classes

### Planner
Top level and *only* external interface. Operates completely in Lat/Lon representation.

### Environment
Responsible for maintaining the state of the flight area. Internally converts between Lat/Lon and Ft representations.

### FlightGraph
Maintains the graph representation of the flight area. Internally converts between Ft and Row/Col representations
