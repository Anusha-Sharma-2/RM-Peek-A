import numpy as np
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 50
FRAMES = 60
START_X = 5
START_Y = 25

def setup_grid():
    grid = np.ones((GRID_SIZE, GRID_SIZE))
    grid[20:30, :] = 0  # Horizontal corridor
    grid[:, 20:30] = 0  # Vertical corridor
    return grid

def generate_trajectories(grid):
    baseline_robot_pos = []
    rm_robot_pos = []
    obstacle_pos = []

    crashed = False
    crash_time = 999
    crash_pos = (0, 0)

    for t in range(FRAMES):
        # DYNAMIC OBSTACLE
        oy = 5 + (t * 1.0) if t < 45 else 50 - ((t - 45) * 1.0)
        obstacle_pos.append((25, oy))

        # BASELINE A* (Moves forward until collision)
        if not crashed:
            bx = START_X + t
            dist = np.sqrt((bx - 25)**2 + (START_Y - oy)**2)
            if dist <= 2.0:
                crashed = True
                crash_time = t
                crash_pos = (bx, START_Y)
            baseline_robot_pos.append((min(bx, 45), START_Y))
        else:
            # Robot is dead, stays at crash site
            baseline_robot_pos.append(crash_pos)

        # RM-PEEK A* (Stops to wait)
        rx = START_X + t if t <= 12 else (17 if t <= 27 else 17 + (t - 27))
        rm_robot_pos.append((min(rx, 45), START_Y))

    return baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time

def create_plot(grid, baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#2b2b2b')
    risk_zone = np.zeros((GRID_SIZE, GRID_SIZE))
    risk_zone[15:35, 15:35] = 1

    time_steps = [10, 21, 35]
    column_titles = ["t = 1.0s: Approach", "t = 2.1s: Interaction", "t = 3.5s: Result"]

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

            # Plot Obstacle
            ax.plot([ox], [oy], 'rs', markersize=14, markeredgecolor='white', markeredgewidth=2)

            if row == 0: # BASELINE ROW
                rx, ry = baseline_robot_pos[t]

                # Check if we are at or past the crash time
                if t >= crash_time:
                    ax.plot([rx], [ry], marker='X', color='red', markersize=16, markeredgecolor='white')
                    ax.text(rx, ry+5, 'COLLISION', color='red', fontsize=12, fontweight='bold', ha='center')
                    # Draw a broken path line up to crash point
                    path_x, path_y = zip(*baseline_robot_pos[:crash_time+1])
                    ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.5)
                else:
                    ax.plot([rx], [ry], 'bo', markersize=16, markeredgecolor='white')
                    # Draw path line up to t
                    path_x, path_y = zip(*baseline_robot_pos[:t+1])
                    ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.5)

            else: # RM-PEEK ROW
                ax.imshow(np.ma.masked_where(risk_zone == 0, risk_zone), cmap='Reds', alpha=0.3, origin='lower')
                rx, ry = rm_robot_pos[t]
                ax.plot([rx], [ry], 'go', markersize=16, markeredgecolor='white', markeredgewidth=2)

                # Trail
                path_x, path_y = zip(*rm_robot_pos[:t+1])
                ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.7)

                if 12 < t <= 27:
                    ax.text(17, 32, 'PEEK POINT\nWAITING', color='orange', fontsize=11, fontweight='bold', ha='center')
                elif t > 27:
                    ax.text(rx, ry+5, 'SAFE PASSAGE', color='lime', fontsize=11, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig('fig4_fork.png', dpi=300, facecolor='#2b2b2b', bbox_inches='tight')
    print("Successfully generated high-res fig4_fork.png!")

if __name__ == '__main__':
    grid = setup_grid()
    baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time = generate_trajectories(grid)
    create_plot(grid, baseline_robot_pos, rm_robot_pos, obstacle_pos, crash_time)