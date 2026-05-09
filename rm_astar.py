import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import heapq

# --- 1. SETUP PARAMETERS ---
GRID_SIZE = 50
V_HIDDEN = 1.0     # Speed of potential hidden obstacle (cells/sec)
V_ROBOT = 1.0      # Speed of robot (cells/sec)
TAU_SAFE = 4.0     # Required reaction time (seconds)
LAMBDA_RISK = 5.0  # Weight of risk in A* cost

# Grid States
FREE = 0
WALL = 1
UNKNOWN = 2

# --- 2. BUILD THE MAP ---
def create_blind_corner_map():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Create a hallway / blind corner
    grid[0:20, 20:25] = WALL  # Top wall
    grid[30:50, 20:25] = WALL  # Bottom wall
    
    # Define the occluded/unknown region behind the bottom wall
    grid[30:50, 25:50] = UNKNOWN
    
    return grid

# --- 3. THE REACTION-MARGIN RISK ENGINE ---
def compute_risk_grid(grid):
    # Find Occlusion Frontiers (Free cells adjacent to Unknown)
    frontiers = np.zeros_like(grid)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == FREE:
                # Check neighbors for UNKNOWN
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if grid[nr, nc] == UNKNOWN:
                            frontiers[r, c] = 1
                            break

    # O(N) Distance Transform: Calculates distance from every cell to nearest frontier
    # We invert the frontiers mask because EDT measures distance to 0
    dist_to_frontier = distance_transform_edt(1 - frontiers)
    
    # Calculate Risk
    risk_grid = np.zeros_like(grid)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == FREE:
                dist = dist_to_frontier[r, c]
                
                # THE CORRECTED KINEMATIC FORMULA
                reaction_margin = dist / (V_HIDDEN + V_ROBOT)
                
                if reaction_margin < TAU_SAFE:
                    # Square it to heavily penalize getting too close
                    risk_grid[r, c] = (TAU_SAFE - reaction_margin) ** 2
                    
    return risk_grid, frontiers

# --- 4. THE A* PLANNER ---
def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(grid, risk_grid, start, goal, use_risk=False):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
            
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
            neighbor = (current[0]+dr, current[1]+dc)
            
            # Check bounds and walls/unknowns
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if grid[neighbor[0], neighbor[1]] != FREE:
                    continue
                
                # Diagonal movement cost is 1.414, straight is 1.0
                move_cost = np.sqrt(dr**2 + dc**2)
                
                # ADD RISK COST IF ENABLED
                if use_risk:
                    move_cost += LAMBDA_RISK * risk_grid[neighbor[0], neighbor[1]]
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return None # No path found

# --- 5. RUN EXPERIMENT AND GENERATE METRICS ---
if __name__ == "__main__":
    grid = create_blind_corner_map()
    risk_grid, frontiers = compute_risk_grid(grid)
    
    start_node = (40, 10) 
    goal_node = (10, 40)  
    
    print("Running Simulations...\n")
    path_baseline = astar(grid, risk_grid, start_node, goal_node, use_risk=False)
    path_rm = astar(grid, risk_grid, start_node, goal_node, use_risk=True)

    # --- METRICS CALCULATOR ---
    def calculate_metrics(path, risk_grid):
        length = 0
        total_risk = 0
        min_reaction_margin = 999.0
        
        for i in range(len(path)):
            r, c = path[i]
            total_risk += risk_grid[r, c]
            
            # Recalculate reaction margin for the metric
            dist_to_frontier = distance_transform_edt(1 - frontiers)[r, c]
            rm = dist_to_frontier / (V_HIDDEN + V_ROBOT)
            if rm < min_reaction_margin:
                min_reaction_margin = rm
                
            if i > 0:
                pr, pc = path[i-1]
                length += np.sqrt((r-pr)**2 + (c-pc)**2)
                
        return length, total_risk, min_reaction_margin

    len_base, risk_base, rm_base = calculate_metrics(path_baseline, risk_grid)
    len_rm, risk_rm, rm_rm = calculate_metrics(path_rm, risk_grid)

    print("-" * 65)
    print(f"{'Method':<20} | {'Path Length':<12} | {'Total Risk':<12} | {'Min Reaction Margin'}")
    print("-" * 65)
    print(f"{'Baseline A*':<20} | {len_base:<12.2f} | {risk_base:<12.2f} | {rm_base:.2f} sec")
    print(f"{'RM-A* (Ours)':<20} | {len_rm:<12.2f} | {risk_rm:<12.2f} | {rm_rm:.2f} sec")
    print("-" * 65)