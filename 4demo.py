import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import distance_transform_edt
import heapq

# Constants
GRID_SIZE = 50
FREE = 0
WALL = 1
FRAMES = 150

def create_skeld_map():
    grid = np.ones((GRID_SIZE, GRID_SIZE))
    # Cafeteria (Top Right)
    grid[35:48, 25:45] = FREE
    # Admin Hallway (Going Down)
    grid[20:35, 32:38] = FREE
    # Storage (Bottom Center)
    grid[5:20, 20:40] = FREE
    # Electrical (Bottom Left)
    grid[10:25, 5:18] = FREE
    # Doorway: Storage -> Electrical
    grid[12:17, 18:20] = FREE

    # THE ALTERNATE ROUTE (The Long Way)
    # MedBay Hallway (Top across)
    grid[38:42, 10:25] = FREE
    # Upper/Lower Engine Hallway (Left vertical)
    grid[5:40, 5:10] = FREE

    return grid

def astar(grid, dynamic_costs, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # 4-way movement
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            neighbor = (current[0]+dr, current[1]+dc)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if grid[neighbor[0], neighbor[1]] == WALL:
                    continue

                # Add the base cost + any dynamic risk penalties (like seeing an Imposter)
                move_cost = 1.0 + dynamic_costs[neighbor[0], neighbor[1]]

                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.sqrt((neighbor[0]-goal[0])**2 + (neighbor[1]-goal[1])**2)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

def initialize_paths(grid):
    start_node = (42, 35)  # Start in Cafeteria
    goal_node = (15, 12)   # Task in Electrical

    # Calculate Initial Baseline Path (Assumes map is empty)
    base_risk = np.zeros_like(grid)
    initial_path = astar(grid, base_risk, start_node, goal_node)
    return start_node, goal_node, initial_path

def run_simulation(grid, start_node, goal_node, initial_path):
    baseline_history = []
    rm_history = []
    imposter_history = []
    path_history = []

    # State Trackers
    base_idx = 0
    base_crashed = False
    rm_pos = start_node
    rm_path = initial_path.copy()
    rm_idx = 0
    imposter_detected = False

    print("Running Among Us Event-Triggered Replanning...")
    for t in range(FRAMES):
        # IMPOSTER MOVEMENT: Patrols inside Storage, blocking the door to Electrical
        if t < 50:
            iy = 15
            ix = 35 - (t * 0.3)
        elif t < 100:
            iy = 15
            ix = 20 + ((t-50) * 0.3)
        else:
            iy = 15
            ix = 35 - ((t-100) * 0.3)
        imposter_pos = (int(iy), int(ix))
        imposter_history.append(imposter_pos)

        # 1. BASELINE CREWMATE: Blindly follows the GPS line
        if not base_crashed and base_idx < len(initial_path) - 1:
            base_idx += 1
        br, bc = initial_path[base_idx]
        baseline_history.append((br, bc))

        # Check if Imposter kills Baseline
        if not base_crashed and np.sqrt((bc - imposter_pos[1])**2 + (br - imposter_pos[0])**2) < 2.0:
            base_crashed = True
            crash_time = t

        # 2. RM-PEEK CREWMATE: Active Scanning & Replanning
        if not imposter_detected:
            # Check Line of Sight / Distance
            # If Crewmate is at the bottom of Admin hallway (peeking into Storage)
            if rm_pos[0] <= 21 and rm_pos[1] >= 32:
                dist_to_imposter = np.sqrt((rm_pos[1] - imposter_pos[1])**2 + (rm_pos[0] - imposter_pos[0])**2)
                if dist_to_imposter < 15.0: # "Vision Radius"
                    imposter_detected = True
                    # EVENT TRIGGER: Fill Storage with "Lava" cost so A* avoids it
                    dynamic_risk = np.zeros_like(grid)
                    dynamic_risk[5:20, 20:40] = 5000
                    # Replan from current position!
                    rm_path = astar(grid, dynamic_risk, rm_pos, goal_node)
                    rm_idx = 0

        if rm_idx < len(rm_path) - 1:
            rm_idx += 1
        rm_pos = rm_path[rm_idx]

        rm_history.append((rm_pos, imposter_detected))
        path_history.append(rm_path.copy())

    return baseline_history, rm_history, imposter_history, path_history

def create_animation(grid, initial_path, baseline_history, rm_history, imposter_history, path_history, start_node, goal_node):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("The Skeld: Active Perception & Dynamic Rerouting", fontsize=18, fontweight='bold', color='white')
    fig.patch.set_facecolor('#111111') # Space black

    for i, ax in enumerate(axes):
        ax.imshow(grid, cmap='bone', origin='lower')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.axis('off')

        # Map Labels
        ax.text(35, 42, 'CAFETERIA', color='white', alpha=0.5, ha='center', fontweight='bold')
        ax.text(30, 10, 'STORAGE', color='white', alpha=0.5, ha='center', fontweight='bold')
        ax.text(11, 20, 'ELEC', color='yellow', alpha=0.5, ha='center', fontweight='bold')
        ax.text(7, 35, 'ENGINE', color='white', alpha=0.5, ha='center', fontweight='bold')

        if i == 0:
            ax.set_title("Standard A* (Crewmate)", color='white', fontsize=14)
            # Draw static intended path
            py, px = zip(*initial_path)
            ax.plot(px, py, 'w--', alpha=0.4, linewidth=2)
        else:
            ax.set_title("RM-Peek A* (Crewmate)", color='white', fontsize=14)

    # Visual Assets
    rob_base, = axes[0].plot([], [], 'co', markersize=12, markeredgecolor='white', markeredgewidth=2) # Cyan Crewmate
    obs_base, = axes[0].plot([], [], 'ro', markersize=14, markeredgecolor='black', markeredgewidth=2) # Red Imposter
    crash_text = axes[0].text(25, 25, '', color='red', fontsize=20, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))

    rob_rm, = axes[1].plot([], [], 'co', markersize=12, markeredgecolor='white', markeredgewidth=2)
    obs_rm, = axes[1].plot([], [], 'ro', markersize=14, markeredgecolor='black', markeredgewidth=2)
    status_text = axes[1].text(25, 25, '', color='cyan', fontsize=16, fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.7))
    dynamic_path_line, = axes[1].plot([], [], 'w--', alpha=0.4, linewidth=2)
    vision_beam, = axes[1].plot([], [], 'y-', alpha=0.0, linewidth=2)

    def init():
        return rob_base, obs_base, crash_text, rob_rm, obs_rm, status_text, dynamic_path_line, vision_beam

    def animate(t):
        iy, ix = imposter_history[t]
        obs_base.set_data([ix], [iy])
        obs_rm.set_data([ix], [iy])

        # Baseline Animation
        br, bc = baseline_history[t]
        if t > crash_time if 'crash_time' in globals() else False:
            rob_base.set_marker('X')
            rob_base.set_color('red')
            crash_text.set_text('DEAD BODY REPORTED')
        else:
            rob_base.set_data([bc], [br])

        # RM-Peek Animation
        (rr, rc), detected = rm_history[t]
        rob_rm.set_data([rc], [rr])

        # Update the dynamic path line
        current_path = path_history[t]
        py, px = zip(*current_path)
        dynamic_path_line.set_data(px, py)

        if detected:
            status_text.set_text('IMPOSTER DETECTED: RE-ROUTING')
            status_text.set_color('lime')
            if t < 60: # Flash vision beam briefly
                vision_beam.set_data([rc, ix], [rr, iy])
                vision_beam.set_alpha(0.8)
            else:
                vision_beam.set_alpha(0.0)
        else:
            status_text.set_text('DOING TASKS')
            status_text.set_color('cyan')

        return rob_base, obs_base, crash_text, rob_rm, obs_rm, status_text, dynamic_path_line, vision_beam

    print("Rendering Among Us Simulation... (~15 seconds)")
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=FRAMES, interval=50, blit=True)
    ani.save('among_us_replanning.gif', writer='pillow', fps=20)
    print("Saved among_us_replanning.gif successfully!")

if __name__ == '__main__':
    grid = create_skeld_map()
    start_node, goal_node, initial_path = initialize_paths(grid)
    baseline_history, rm_history, imposter_history, path_history = run_simulation(grid, start_node, goal_node, initial_path)
    create_animation(grid, initial_path, baseline_history, rm_history, imposter_history, path_history, start_node, goal_node)