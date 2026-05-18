ENPM661 Project 03: A* Path Planning for a Mobile Robot (Phase 1 & 2)

Team Members
Anusha Sharma (asharm50)

Overview
This repository contains a collection of Python demos and solvers for A* path planning with risk-aware decision making. It includes:
- `rm_astar.py`: a backward AD* / A* style solver on a small graph example.
- `test.py`: a risk-margin A* planner for a blind-corner grid scenario with metric comparison between baseline A* and RM-A*.
- `1demo.py`, `2demo.py`, `3demo.py`, `4demo.py`: visualization/demo scripts for robot motion, risk zones, and scenario behaviors.

The code is intended to show how a robot can safely plan around unknown or occluded regions in a grid world.

Dependencies
The following libraries are required to run this project:

- Python 3.x
- NumPy (`numpy`) - used for grid setup and math operations.
- Matplotlib (`matplotlib`) - used for plotting and visualization.
- SciPy (`scipy`) - used for distance transform and risk computation.
- NetworkX (`networkx`) - used for graph visualization in `rm_astar.py`.
- `math`, `heapq` - standard Python libraries used by the planners.

You can install the external Python dependencies using pip:

```
pip install numpy matplotlib scipy networkx
```

How to run:
Here are the step-by-step instructions to run the project:

1. Open your terminal and navigate to the repository folder:
```
cd .\RM-Peek-A\"
```

2. Install dependencies if needed:
```
pip install numpy matplotlib scipy networkx
```

3. Run the main solver for the graph-based AD*/A* example:
```
python rm_astar.py
```
This script runs baseline A* and RM-PeekA* for comparison data


4. Run a demo visualization script:
```
python 1demo.py
python 2demo.py
python 3demo.py
python 4demo.py
```
This script generates a visualization image named `fig4_fork.png` showing baseline and RM-Peek A* behavior.

Part 01 sample input:
This repository uses hardcoded example scenarios rather than interactive console input, so you can run the scripts directly without entering coordinates.

- The demo scripts (`1demo.py` through `4demo.py`) are designed to show output plots and scenario behavior visually.
