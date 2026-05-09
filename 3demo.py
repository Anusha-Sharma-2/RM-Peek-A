import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import distance_transform_edt
import heapq

# Constants
GRID_SIZE = 50
V_HIDDEN = 1.0
V_ROBOT = 1.0
TAU_SAFE = 4.0
LAMBDA_RISK = 8.0
FREE = 0
WALL = 1
UNKNOWN = 2
FRAMES = 340

def create_spiral_map():
    grid = np.ones((GRID_SIZE, GRID_SIZE))
    corridors = [
        (40, 45, 5, 45), (5, 45, 45, 50), (5, 10, 5, 45),
        (10, 35, 5, 10), (30, 35, 10, 35), (15, 30, 30, 35),
        (15, 20, 15, 30), (20, 25, 15, 20), (20, 25, 20, 25)
    ]
    for r1, r2, c1, c2 in corridors:
        grid[r1:r2, c1:c2] = FREE

    grid[25:30, 20:30] = UNKNOWN
    grid[35:40, 10:45] = UNKNOWN
    grid[10:15, 10:30] = UNKNOWN
    return grid

def compute_risk_grid(grid):
    frontiers = np.zeros_like(grid)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == FREE:
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if grid[nr, nc] == UNKNOWN or grid[nr, nc] == WALL:
                            frontiers[r, c] = 1
                            break
    dist_to_frontier = distance_transform_edt(1 - frontiers)
    risk_grid = np.zeros_like(grid)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == FREE:
                dist = dist_to_frontier[r, c]
                rm = dist / (V_HIDDEN + V_ROBOT)
                if rm < TAU_SAFE:
                    risk_grid[r, c] = (TAU_SAFE - rm) ** 2
    return risk_grid

def astar(grid, risk_grid, start, goal, use_risk=False):
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

        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]: # 4-way movement for clean grid traversal
            neighbor = (current[0]+dr, current[1]+dc)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if grid[neighbor[0], neighbor[1]] != FREE:
                    continue
                move_cost = 1.0
                if use_risk:
                    move_cost += LAMBDA_RISK * risk_grid[neighbor[0], neighbor[1]]

                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.sqrt((neighbor[0]-goal[0])**2 + (neighbor[1]-goal[1])**2)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

def compute_paths(grid):
    start_node = (42, 7)
    goal_node = (22, 22)
    risk_grid = compute_risk_grid(grid)
    path_baseline = astar(grid, risk_grid, start_node, goal_node, use_risk=False)
    path_rm = astar(grid, risk_grid, start_node, goal_node, use_risk=True)
    return start_node, goal_node, path_baseline, path_rm, risk_grid

def run_simulation(grid, path_baseline, path_rm, start_node, goal_node):
    base_idx = 0
    rm_idx = 0
    base_crashed = False

    # Windmill parameters
    cx, cy = 25, 25
    num_arms = 4
    balls_per_arm = 6
    arm_spacing = 4

    baseline_history = []
    rm_history = []
    obstacles_history = []

    print("Running Dynamic Event-Triggered Simulation...")
    for t in range(FRAMES):
        # Update Windmill
        angle = np.radians(t * 3.5)
        current_obs = []
        for arm in range(num_arms):
            arm_angle = angle + arm * (np.pi / 2)
            for b in range(1, balls_per_arm + 1):
                ox = cx + b * arm_spacing * np.cos(arm_angle)
                oy = cy + b * arm_spacing * np.sin(arm_angle)
                current_obs.append((ox, oy))
        obstacles_history.append(current_obs)

        # Update Baseline (Blindly steps forward every frame)
        if not base_crashed and base_idx < len(path_baseline) - 1:
            base_idx += 1

        br, bc = path_baseline[base_idx]
        baseline_history.append((br, bc))

        # Check Baseline Crash
        if not base_crashed:
            for ox, oy in current_obs:
                if np.sqrt((bc - ox)**2 + (br - oy)**2) < 1.5:
                    base_crashed = True
                    crash_time = t

        # Update RM-Peek (DYNAMIC SENSOR CHECK)
        rm_waiting = False
        if rm_idx < len(path_rm) - 1:
            rr, rc = path_rm[rm_idx]
            next_r, next_c = path_rm[rm_idx + 1]

            # ACTIVE PERCEPTION: Check if any ball is too close to our current or next position
            danger = False
            for ox, oy in current_obs:
                dist_to_current = np.sqrt((rc - ox)**2 + (rr - oy)**2)
                dist_to_next = np.sqrt((next_c - ox)**2 + (next_r - oy)**2)

                # If a ball is within 4.5 cells of our next step, TRIGGER PEEK (Wait)
                if dist_to_next < 4.5 or dist_to_current < 3.0:
                    danger = True
                    break

            if danger:
                rm_waiting = True # Halts execution
            else:
                rm_idx += 1       # Safe to proceed

        current_rr, current_rc = path_rm[rm_idx]
        rm_history.append(((current_rr, current_rc), rm_waiting))

    return baseline_history, rm_history, obstacles_history, num_arms, balls_per_arm

def create_animation(grid, path_baseline, path_rm, baseline_history, rm_history, obstacles_history, num_arms, balls_per_arm, start_node, goal_node):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("True Autonomous Execution: Dynamic RM-Peek Algorithm", fontsize=18, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1e1e1e')

    for i, ax in enumerate(axes):
        ax.imshow(grid, cmap='bone', origin='lower')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.axis('off')
        ax.add_patch(plt.Rectangle((4, 40), 5, 5, color='lime', alpha=0.5))
        ax.add_patch(plt.Rectangle((20, 20), 5, 5, color='lime', alpha=0.5))

        if i == 0:
            ax.set_title("Standard A* (Blindly Advances)", color='white', fontsize=14)
            py, px = zip(*path_baseline)
            ax.plot(px, py, 'w-', alpha=0.3, linewidth=2) # Draw intended path
        else:
            ax.set_title("RM-Peek A* (Dynamic TTC Trigger)", color='white', fontsize=14)
            py, px = zip(*path_rm)
            ax.plot(px, py, 'w-', alpha=0.3, linewidth=2) # Draw intended path

    rob_base, = axes[0].plot([], [], 'rs', markersize=12, markeredgecolor='black', markeredgewidth=2)
    crash_text = axes[0].text(25, 25, '', color='red', fontsize=30, fontweight='bold', ha='center', va='center')
    obs_plots_base = [axes[0].plot([], [], 'bo', markersize=8, markeredgecolor='black')[0] for _ in range(num_arms * balls_per_arm)]

    rob_rm, = axes[1].plot([], [], 'rs', markersize=12, markeredgecolor='black', markeredgewidth=2)
    status_text = axes[1].text(25, 47, '', color='cyan', fontsize=14, fontweight='bold', ha='center')
    obs_plots_rm = [axes[1].plot([], [], 'bo', markersize=8, markeredgecolor='black')[0] for _ in range(num_arms * balls_per_arm)]

    def init():
        rob_base.set_data([], [])
        rob_rm.set_data([], [])
        for o1, o2 in zip(obs_plots_base, obs_plots_rm):
            o1.set_data([], [])
            o2.set_data([], [])
        return [rob_base, rob_rm, crash_text, status_text] + obs_plots_base + obs_plots_rm

    def animate(t):
        current_obs = obstacles_history[t]
        for i, (ox, oy) in enumerate(current_obs):
            obs_plots_base[i].set_data([ox], [oy])
            obs_plots_rm[i].set_data([ox], [oy])

        # Baseline
        br, bc = baseline_history[t]
        if t > crash_time if 'crash_time' in globals() else False:
            rob_base.set_color('black')
            rob_base.set_marker('x')
            crash_text.set_text('GAME OVER')
        else:
            rob_base.set_data([bc], [br])

        # RM-Peek
        (rr, rc), waiting = rm_history[t]
        rob_rm.set_data([rc], [rr])

        if waiting:
            status_text.set_text('THREAT DETECTED: WAITING')
            status_text.set_color('orange')
            rob_rm.set_color('yellow')
        else:
            status_text.set_text('PROCEEDING')
            status_text.set_color('lime')
            rob_rm.set_color('red')

        if rc == goal_node[1] and rr == goal_node[0]:
            status_text.set_text('GOAL REACHED!')
            status_text.set_color('gold')

        return [rob_base, rob_rm, crash_text, status_text] + obs_plots_base + obs_plots_rm

    print("Rendering True Algorithm Animation... (~40 seconds)")
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=FRAMES, interval=40, blit=True)
    ani.save('true_autonomous_whg.gif', writer='pillow', fps=25)
    print("Saved true_autonomous_whg.gif successfully!")

if __name__ == '__main__':
    grid = create_spiral_map()
    start_node, goal_node, path_baseline, path_rm, risk_grid = compute_paths(grid)
    baseline_history, rm_history, obstacles_history, num_arms, balls_per_arm = run_simulation(grid, path_baseline, path_rm, start_node, goal_node)
    create_animation(grid, path_baseline, path_rm, baseline_history, rm_history, obstacles_history, num_arms, balls_per_arm, start_node, goal_node)