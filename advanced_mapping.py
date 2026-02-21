import numpy as np
import math
import time
import heapq
from picarx import Picarx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MAP_SIZE = 100
MAX_DISTANCE = 100
CM_PER_CELL = 1
ANGLE_START = -60
ANGLE_END = 60
ANGLE_STEP = 5
OBSTACLE_THRESHOLD = 50
CM_PER_SEC = 10
DRIVE_SPEED = 30          
TURN_DEG_PER_SEC = 45
_robot_artists = []
ROBOT_MARGIN = 15

px = Picarx()
grid_map = np.zeros((MAP_SIZE, MAP_SIZE))

# Car position (cm, radians)
car_x = MAP_SIZE // 2
car_y = 0
car_theta = math.pi / 2  # car starts facing the positive y direction (90 degrees)

def main():
    """
    Drive the car forward while repeatedly scanning the environment.
    Runs for a fixed number of iterations or until stopped by the user.
    """
    try:
        for step in range(5):
            print(f'Main loop iteration: {step + 1}/5')

            decision = scan_environment()
            print(f'\nDecision: {decision}\n')

            if decision == 'center':
                px.stop()
                reverse(0.5)
                turn_left(0.5)
            elif decision == 'left':
                px.stop()
                reverse(0.5)  
                turn_right(0.4)
            elif decision == 'right':
                px.stop()
                reverse(0.5)
                turn_left(0.4)
            else:
                drive_forward(0.6)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print('Stopped by user')

    finally:
        px.stop()

def mark_cell(x, y):
    """
    Mark the cell at (x, y) as an obstacle in the grid map.
    Also mark neighboring cells to create a more visible obstacle area.

    param x: X coordinate in cm
    type x: int
    param y: Y coordinate in cm
    type y: int
    """
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                grid_map[ny, nx] = 1

def interpolate(point1, point2):
    """
    Mark cells along the line between two detected obstacle points to create a more continuous map.

    param point1: First detected obstacle point (x, y)
    type point1: tuple of (int, int)
    param point2: Second detected obstacle point (x, y)
    type point2: tuple of (int, int)
    """
    x1, y1 = point1
    x2, y2 = point2

    steps = max(abs(x2 - x1), abs(y2 - y1))

    if steps == 0: 
        return

    for i in range(steps + 1):
        # Linear interpolation formula
        x = int(x1 + i * (x2 - x1) / steps)
        y = int(y1 + i * (y2 - y1) / steps)
        mark_cell(x, y)

def read_distance_median(samples = 7, delay = 0.02):
    """
    Read multiple ultrasonic sensor values and return the median to reduce noise.
    
    param samples: Number of readings to take for median calculation
    type samples: int
    param delay: Delay between readings in seconds
    type delay: float
    return: Median distance reading in cm, or None if no valid readings
    rtype: float or None
    """
    vals = []
    for _ in range(samples):
        d = px.ultrasonic.read()
        if d is not None and 0 < d <= MAX_DISTANCE: # Reject outliers and invalid readings
            vals.append(d)
        time.sleep(delay)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals)//2]

def scan_environment():
    """
    Rotate the pan servo to scan the environment in front of the car.
    Mark detected obstacles on the grid map and count hits in left, right, and center zones
    to make a decision on where to drive next.

    return: Decision on where to drive next ('left', 'right', 'center', or None)
    rtype: str or None
    """
    previous_point = None
    previous_distance = None

    left_hits = 0
    right_hits = 0
    center_hits = 0

    FRONT_ANGLE_WINDOW = 10
    FRONT_STOP_CM = 25

    for servo_angle in range(ANGLE_START, ANGLE_END + 1, ANGLE_STEP):
        px.set_cam_pan_angle(servo_angle)
        time.sleep(0.15)

        _ = px.ultrasonic.read()
        time.sleep(0.03)

        distance = read_distance_median()
        print(f'Angle: {servo_angle}, Distance: {distance}')

        if distance is not None and previous_distance is not None:
            if abs(distance - previous_distance) > 8:
                previous_point = None

        previous_distance = distance

        if distance is None:
            previous_point = None
            continue

        # Detect obstacles in front and count hits in left, right, and center zones
        if distance <= FRONT_STOP_CM:
            if servo_angle < -FRONT_ANGLE_WINDOW:
                left_hits += 1
            elif servo_angle > FRONT_ANGLE_WINDOW:
                right_hits += 1
            else:
                center_hits += 1

        if distance <= OBSTACLE_THRESHOLD:
            distance_cells = distance / CM_PER_CELL

            # Convert polar to Cartesian coordinates
            angle_radians = math.radians(-servo_angle) + car_theta
            x = int(round(car_x + distance_cells * math.cos(angle_radians)))
            y = int(round(car_y + distance_cells * math.sin(angle_radians)))

            # Mark cells and interpolate if within map bounds
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                mark_cell(x, y)

                if previous_point is not None:
                        interpolate(previous_point, (x, y))
                previous_point = (x, y)

            else:
                previous_point = None
        else:
            previous_point = None

    # Update the displayed map after scanning
    show_map()

    # Reset pan servo to center after scan
    px.set_cam_pan_angle(0)

    # Decide on action based on hit counts
    if center_hits > 0:
        return 'center'
    elif left_hits > right_hits:
        return 'left'
    elif right_hits > left_hits:
        return 'right'
    else:
        return None

def clamp_pose():
    """
    Ensure (car_x, car_y) stays within the bounds of the map.
    """
    global car_x, car_y

    car_x = max(0, min(MAP_SIZE - 1, car_x))
    car_y = max(0, min(MAP_SIZE - 1, car_y))

def update_position(distance_cm):
    """
    Update (car_x, car_y) assuming we moved forward distance_cm.
    Localization is achieved using velocity and through dead reckoning.

    param distance_cm: Distance moved forward in centimeters
    type distance_cm: float
    """
    global car_x, car_y

    distance_cells = distance_cm / CM_PER_CELL
    car_x = int(round(car_x + distance_cells * math.cos(car_theta)))
    car_y = int(round(car_y + distance_cells * math.sin(car_theta)))

    clamp_pose()

def drive_forward(seconds, speed = DRIVE_SPEED):
    """
    Drive the car forward for a specified duration and update position estimate.

    param seconds: Duration to drive forward in seconds
    type seconds: float
    param speed: Speed to drive at (0-100)
    type speed: int
    """
    px.forward(speed)
    time.sleep(seconds)
    px.stop()

    update_position(CM_PER_SEC * seconds)

def reverse(seconds, speed = DRIVE_SPEED):
    """
    Reverse the car for a short duration and update position estimate.

    param seconds: Duration to reverse in seconds
    type seconds: float
    param speed: Speed to reverse at
    type speed: int
    """
    px.backward(speed)
    time.sleep(seconds)
    px.stop()

    update_position(-CM_PER_SEC * seconds)

def turn_left(seconds):
    """
    Turn the car left for a specified duration and update orientation estimate.

    param seconds: Duration to turn in seconds
    type seconds: float
    """
    global car_theta

    px.set_dir_servo_angle(-30)
    px.forward(DRIVE_SPEED)
    time.sleep(seconds)
    px.stop()
    px.set_dir_servo_angle(0)

    car_theta = (car_theta + math.radians(TURN_DEG_PER_SEC * seconds)) % (2 * math.pi)

def turn_right(seconds):
    """
    Turn the car right for a specified duration and update orientation estimate.

    param seconds: Duration to turn in seconds
    type seconds: float
    """
    global car_theta
  
    px.set_dir_servo_angle(30)
    px.forward(DRIVE_SPEED)
    time.sleep(seconds)
    px.stop()
    px.set_dir_servo_angle(0)

    car_theta = (car_theta - math.radians(TURN_DEG_PER_SEC * seconds)) % (2 * math.pi)

def draw_robot(ax, x, y, theta):
    """
    Draw the robot as a rectangle with a heading line.
    Robot is drawn OUTSIDE the map (below y=0), with its front at (x, y).
    """
    ROBOT_WIDTH = 8
    ROBOT_LENGTH = 12
    HEADING_LENGTH = 10

    rect = patches.Rectangle(
        (x - ROBOT_WIDTH / 2, y - ROBOT_LENGTH),
        ROBOT_WIDTH,
        ROBOT_LENGTH,
        linewidth=2,
        edgecolor='yellow',
        facecolor='none'
    )
    ax.add_patch(rect)

    hx = x + HEADING_LENGTH * math.cos(theta)
    hy = y + HEADING_LENGTH * math.sin(theta)
    line, = ax.plot([x, hx], [y, hy], color='cyan', linewidth=2)

    return [rect, line]
    
plt.ion()  # Enable interactive mode for live updates

_fig, _ax = plt.subplots()
_img = _ax.imshow(grid_map, origin='lower', vmin=0, vmax=1)
_ax.set_title('Obstacle Map (1=obstacle)')

plt.show()

def show_map():
    """
    Update the displayed map with the current grid_map data.
    """
    global _robot_artists

    _img.set_data(grid_map)

    for a in _robot_artists:
        a.remove()

    _robot_artists = draw_robot(_ax, car_x, car_y, car_theta)

    _ax.set_xlim(0, MAP_SIZE)
    _ax.set_ylim(-ROBOT_MARGIN, MAP_SIZE)

    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)

#  A* PATHFINDING - NOT TESTED

SAFETY_BUFFER = 3          # cells of clearance around obstacles (mentioned in step 8, obstacle avoidance)
REPLAN_STEPS  = 5          # follow this many steps before re-scanning / new pathfidn
GOAL_TOLERANCE = 3         # num cells close enough to declare arrival / success


def _inflate_obstacles(grid, buff=SAFETY_BUFFER):
    """
    Return a copy of grid where every obstacle cell has been expanded by
    buff cells in every direction.  This keeps paths away from small gaps.
    """
    inflated = grid.copy()
    obstacles = np.argwhere(grid == 1)          # (row=y, col=x)
    for oy, ox in obstacles:
        y_lo = max(0, oy - buff)
        y_hi = min(grid.shape[0], oy + buff + 1)
        x_lo = max(0, ox - buff)
        x_hi = min(grid.shape[1], ox + buff + 1)
        inflated[y_lo:y_hi, x_lo:x_hi] = 1
    return inflated


def heuristic(a, b):
    """
    Manhattan distance between two (x, y) tuples.
    Used for astar algorithm to speed up calculation.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    """
    A* search on a 2-D grid (0 = free, 1 = obstacle).
    Essentially Djikstra's algo / Breadth First Search (since edge weights are 1)
    but with a heuristic to speed up computation.

    :param grid:  numpy array (MAP_SIZE x MAP_SIZE), 0/1
    :param start: (x, y) start cell
    :param goal:  (x, y) goal cell
    :return: list of (x, y) waypoints from start to goal, or None if
             no path exists
    """
    rows, cols = grid.shape      # rows -> y, cols -> x

    # Inflate obstacles so the car body has clearance
    cost_map = _inflate_obstacles(grid)

    # Ensure start/goal are not inside inflated obstacles
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]),  int(goal[1])
    if not (0 <= sx < cols and 0 <= sy < rows):
        return None
    if not (0 <= gx < cols and 0 <= gy < rows):
        return None
    # 4-connected neighbours (dx, dy)
    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    if cost_map[gy, gx] == 1:
        # Goal is blocked -> BFS outward to find the nearest free cell
        print('[astar] Goal is inside an inflated obstacle, searching for nearest free cell...')
        from collections import deque
        queue = deque([(gx, gy)])
        visited = {(gx, gy)}
        found = None
        while queue:
            fx, fy = queue.popleft()
            if cost_map[fy, fx] == 0:
                found = (fx, fy)
                break
            for ddx, ddy in DIRS:
                nnx, nny = fx + ddx, fy + ddy
                if 0 <= nnx < cols and 0 <= nny < rows and (nnx, nny) not in visited:
                    visited.add((nnx, nny))
                    queue.append((nnx, nny))
        if found is None:
            print('[astar] No reachable free cell near goal.')
            return None
        gx, gy = found
        print(f'[astar] Rerouting to nearest free cell: ({gx}, {gy})')

    open_set = []   # min-heap of (f, x, y)
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, sx, sy))
    came_from = {} # dict for path building
    g_score = {(sx, sy): 0}

    while open_set:
        f, g, cx, cy = heapq.heappop(open_set)

        if (cx, cy) == (gx, gy):
            # Reconstruct path
            path = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path

        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows and cost_map[ny, nx] == 0:
                tentative_g = g + 1
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = tentative_g
                    f_new = tentative_g + heuristic((nx, ny), (gx, gy))
                    heapq.heappush(open_set, (f_new, tentative_g, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)

    return None   # no path found


def visualize_path(path):
    """
    Overlay the planned path on the matplotlib map as a green line.
    """
    if path is None:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    _ax.plot(xs, ys, color='lime', linewidth=1.5, marker='.', markersize=2)
    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)


#  Movement helpers -> translate grid direction into physical car movement

def _angle_between(target_rad, current_rad):
    """
    Signed shortest angular difference (radians).
    """
    diff = (target_rad - current_rad + math.pi) % (2 * math.pi) - math.pi
    return diff


def move_to_adjacent_cell(dx, dy):
    """
    Turn the car (if needed) to face the (dx, dy) direction and drive
    forward one cell.

    dx, dy \in {-1, 0, 1} -> only one is non-zero at a time (4-connected).
    """
    target_theta = math.atan2(dy, dx)   # desired world heading
    angle_diff   = _angle_between(target_theta, car_theta)

    # turn car toward the target direction
    ANGLE_TOL = math.radians(10)
    if abs(angle_diff) > ANGLE_TOL:
        turn_time = abs(angle_diff) / math.radians(TURN_DEG_PER_SEC)
        turn_time = max(0.15, min(turn_time, 2.0))   # clamp
        if angle_diff > 0:
            turn_left(turn_time)
        else:
            turn_right(turn_time)

    # drive forward one cell
    cell_distance = CM_PER_CELL
    drive_time = cell_distance / CM_PER_SEC
    drive_time = max(0.1, drive_time)
    drive_forward(drive_time)


#  Main navigation loop  (A* + periodic re-mapping)

def navigate_to_goal(goal_x, goal_y, replan_steps=REPLAN_STEPS):
    """
    Drive the car from its current position to (goal_x, goal_y) using
    repeated A* planning interleaved with environment re-scans.

    :param goal_x: target X cell
    :param goal_y: target Y cell
    :param replan_steps: how many path steps to follow before re-scanning
    """
    goal = (int(goal_x), int(goal_y))
    iteration = 0

    while True:
        iteration += 1
        print(f'\n===== Navigation iteration {iteration} =====')
        print(f'  Car  : ({car_x}, {car_y})  θ={math.degrees(car_theta):.0f}°')
        print(f'  Goal : {goal}')

        # check if we are close enough to the goal 
        dist = heuristic((car_x, car_y), goal)
        if dist <= GOAL_TOLERANCE:
            print('\n Goal reached!')
            px.stop()
            return True

        # scan and rebuild map 
        scan_environment()

        # plan with A* 
        path = astar(grid_map, (car_x, car_y), goal)

        if path is None or len(path) < 2:
            print('[nav] No path found, trying a blind turn and re-scan')
            turn_left(0.5)
            continue

        visualize_path(path)
        print(f'  Path length: {len(path)} cells')

        # follow the path for up to *replan_steps* moves 
        for step_idx in range(1, min(replan_steps + 1, len(path))):
            prev = path[step_idx - 1]
            curr = path[step_idx]
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]

            print(f'  Step {step_idx}: ({prev[0]},{prev[1]}) -> '
                  f'({curr[0]},{curr[1]})  Δ=({dx},{dy})')
            move_to_adjacent_cell(dx, dy)

            # Quick distance check, if something is very close, bail early
            quick_dist = px.ultrasonic.read()
            if quick_dist is not None and 0 < quick_dist < 15:
                print('[nav] Close obstacle, re-planning early')
                px.stop()
                break

        # After following partial path, loop back to re-scan & replan

    return False   # should not reach here


def navigation_main():
    """
    Entry point: ask for a goal and navigate using A* + ultrasonic mapping.
    The original main() reactive-avoidance loop is still available if needed.
    """
    print('\n=== A* Navigation for PiCar-X ===')
    print(f'Map size : {MAP_SIZE}x{MAP_SIZE} cells  ({CM_PER_CELL} cm/cell)')
    print(f'Car start: ({car_x}, {car_y})\n')

    gx = int(input('Goal X (0-{0}): '.format(MAP_SIZE - 1)))
    gy = int(input('Goal Y (0-{0}): '.format(MAP_SIZE - 1)))

    try:
        navigate_to_goal(gx, gy)
    except KeyboardInterrupt:
        print('\nStopped by user')
    finally:
        px.stop()
        px.set_dir_servo_angle(0)
        px.set_cam_pan_angle(0)


if __name__ == '__main__':
    # To use the original reactive obstacle-avoidance loop instead, call:
    #   main()
    navigation_main()