import numpy as np
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 50
FRAMES = 90

def setup_fork_map():
    grid = np.ones((GRID_SIZE, GRID_SIZE))
    grid[18:23, 0:15] = 0   # Entrance
    grid[18:23, 35:50] = 0  # Exit
    grid[10:42, 10:15] = 0  # Left split
    grid[10:42, 35:40] = 0  # Right merge
    grid[10:15, 15:35] = 0  # LOWER ROUTE
    grid[37:42, 15:35] = 0  # UPPER ROUTE
    return grid

def get_path_coordinates():
    lower_path_x = [5, 12, 12, 37, 37, 45]
    lower_path_y = [20, 20, 12, 12, 20, 20]
    upper_path_x = [5, 12, 12, 37, 37, 45]
    upper_path_y = [20, 20, 40, 40, 20, 20]
    return lower_path_x, lower_path_y, upper_path_x, upper_path_y

def generate_trajectories(grid):
    baseline_robot_pos = []
    rm_robot_pos = []
    obstacle_pos = []

    crashed = False
    crash_time = 999
    crash_pos = (0, 0)

    for t in range(FRAMES):
        # DYNAMIC OBSTACLE (Fixed: Bound it to x=38 so it doesn't phase through the wall)
        ox = 39 - t if t < 24 else min(15 + (t - 24), 38)
        oy = 12
        obstacle_pos.append((ox, oy))

        # BASELINE A* (Blindly follows lower route)
        if not crashed:
            if t <= 7: bx, by = 5 + t, 20
            elif 7 < t <= 15: bx, by = 12, 20 - (t - 7)
            else: bx, by = 12 + (t - 15), 12

            dist = np.sqrt((bx - ox)**2 + (by - oy)**2)
            if dist <= 1.5:
                crashed = True
                crash_time = t
                crash_pos = (bx, by)
            baseline_robot_pos.append((bx, by))
        else:
            baseline_robot_pos.append(crash_pos)

        # RM-PEEK A* (The Replanning Logic)
        if t <= 7: rx, ry = 5 + t, 20
        elif 7 < t <= 15: rx, ry = 12, 20
        elif 15 < t <= 35: rx, ry = 12, 20 + (t - 15)
        elif 35 < t <= 60: rx, ry = 12 + (t - 35), 40
        elif 60 < t <= 80: rx, ry = 37, 40 - (t - 60)
        else: rx, ry = min(37 + (t - 80), 45), 20
        rm_robot_pos.append((rx, ry))

    return baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time

def create_plot(grid, baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time, lower_path_x, lower_path_y, upper_path_x, upper_path_y):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#2b2b2b')
    risk_zone = np.zeros((GRID_SIZE, GRID_SIZE))
    risk_zone[12:22, 12:22] = 1

    time_steps = [12, 21, 55]
    column_titles = ["t = 1.2s: Detection at Peek Point", "t = 2.1s: Collision vs. Re-Routing", "t = 5.5s: Result"]

    for col in range(3):
        t = time_steps[col]
        ox, oy = obstacle_pos[t]

        for row in range(2):
            ax = axes[row, col]
            ax.imshow(grid, cmap='binary_r', origin='lower')
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
            ax.axis('off')

            if row == 0:
                ax.set_title(column_titles[col], color='white', fontsize=16, fontweight='bold', pad=15)

            if col == 0:
                label = "Baseline A*" if row == 0 else "RM-Peek A* (Ours)"
                ax.text(-10, 25, label, color='white', fontsize=18, fontweight='bold', rotation=90, va='center')

            # Draw S and G
            ax.text(2, 20, 'S', color='lime', fontsize=14, fontweight='bold')
            ax.text(46, 20, 'G', color='blue', fontsize=14, fontweight='bold')

            # Plot Obstacle
            ax.plot([ox], [oy], 'rs', markersize=14, markeredgecolor='white', markeredgewidth=2)

            if row == 0: # BASELINE ROW
                ax.plot(lower_path_x, lower_path_y, 'b--', linewidth=2, alpha=0.4)
                rx, ry = baseline_robot_pos[t]

                if t >= crash_time:
                    ax.plot([rx], [ry], marker='X', color='red', markersize=16, markeredgecolor='white')
                    ax.text(rx, ry+6, 'COLLISION', color='red', fontsize=12, fontweight='bold', ha='center')
                else:
                    ax.plot([rx], [ry], 'bo', markersize=16, markeredgecolor='white')

                if t >= crash_time:
                    path_x, path_y = zip(*baseline_robot_pos[:crash_time+1])
                else:
                    path_x, path_y = zip(*baseline_robot_pos[:t+1])

                ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7)

            else: # RM-PEEK ROW
                ax.imshow(np.ma.masked_where(risk_zone == 0, risk_zone), cmap='Reds', alpha=0.3, origin='lower')
                rx, ry = rm_robot_pos[t]

                if t < 15:
                    ax.plot(lower_path_x, lower_path_y, 'g--', linewidth=2, alpha=0.4)
                    if t == 12:
                        ax.plot([rx, ox], [ry, oy], 'y-', alpha=0.8, linewidth=2)
                        ax.text(12, 28, 'PEEK POINT:\nTHREAT DETECTED', color='orange', fontsize=11, fontweight='bold', ha='center')
                else:
                    ax.plot(upper_path_x, upper_path_y, 'g--', linewidth=2, alpha=0.4)
                    if t == 21:
                        ax.text(12, 32, 'INFINITE COST APPLIED:\nRE-ROUTING', color='lime', fontsize=11, fontweight='bold', ha='center')
                    elif t == 55:
                        ax.text(rx, ry+6, 'SAFE PASSAGE', color='lime', fontsize=11, fontweight='bold', ha='center')

                ax.plot([rx], [ry], 'go', markersize=16, markeredgecolor='white', markeredgewidth=2)
                path_x, path_y = zip(*rm_robot_pos[:t+1])
                ax.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.7)

    plt.tight_layout()
    plt.savefig('fig_fork_replan.png', dpi=300, facecolor='#2b2b2b', bbox_inches='tight')
    print("Successfully generated high-res fig_fork_replan.png!")

if __name__ == '__main__':
    grid = setup_fork_map()
    lower_path_x, lower_path_y, upper_path_x, upper_path_y = get_path_coordinates()
    baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time = generate_trajectories(grid)
    create_plot(grid, baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time, lower_path_x, lower_path_y, upper_path_x, upper_path_y)