import numpy as np
import math
import time
from picarx import Picarx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import threading
import random

from safety_state import is_person_detected, is_stop_sign_detected
# import object_detection
import object_detection_5MP_cam as object_detection

# Constants
MAP_SIZE = 40
MAX_DISTANCE = 100
CM_PER_CELL = 5
ANGLE_START = -60
ANGLE_END = 60
ANGLE_STEP = 5
OBSTACLE_THRESHOLD = 60
CM_PER_SEC = 20
DRIVE_SPEED = 5
TURN_DEG_PER_SEC = 35
_robot_artists = []
ROBOT_MARGIN = 15
CLEARANCE_RADIUS = 1
STEPS_PER_PLAN = 3

# Turn tuning
TURN_SPEED = DRIVE_SPEED
LEFT_TURN_DEG_PER_SEC  = 45
RIGHT_TURN_DEG_PER_SEC = 34
DT_TURN = 0.02

# Forward drift during turning (cm per second)
RIGHT_TURN_FWD_CM_PER_SEC = 16
LEFT_TURN_FWD_CM_PER_SEC  = 13
LEFT_TURN_RADIUS_CM  = 16
RIGHT_TURN_RADIUS_CM = 27

# Global variable to track if we've already handled a detected stop sign
_stop_sign_handled = False

# Initialize the car and the map
px = Picarx()
grid_map = np.zeros((MAP_SIZE, MAP_SIZE))

# Car position (cm, radians)
car_x = MAP_SIZE // 2
car_y = 0
car_theta = math.pi / 2  # car starts facing the positive y direction (90 degrees)

def main():
    """
    Main function to start the self-driving behavior. 
    Initializes the object detection thread and repeatedly plans 
    and executes paths to the goal while updating the map with sensor data.
    """
    global grid_map
    
    print('Starting object detection camera thread...')
    stop_event = threading.Event()
    od_thread = threading.Thread(
        target=object_detection.run,
        args=('efficientdet_lite0.tflite', 0, 640, 480, 4, False, stop_event),
        daemon=True
    )
    od_thread.start()

    # --- Quick detection test: wait for camera warmup, then check ---
    print('[TEST] Warming up camera for 5 seconds...')
    time.sleep(5)
    print(f'[TEST] Stop sign detected? {is_stop_sign_detected()}')
    print(f'[TEST] Person detected?    {is_person_detected()}')
    print('[TEST] Detection test done. Starting navigation in 3 seconds...')
    time.sleep(3)

    try:
        grid_map[:] = 0
        goal = (12, 18) # Target cell in the map (x, y)
        navigate_to_goal(goal=goal, steps_per_plan=STEPS_PER_PLAN)
    finally:
        stop_event.set()
        px.stop()
        od_thread.join()

def mark_cell(x, y):
    """
    Mark a cell as an obstacle and clear surrounding cells within the clearance radius.

    param x: X coordinate in cm
    type x: int
    param y: Y coordinate in cm
    type y: int
    """
    for dx in range(-CLEARANCE_RADIUS, CLEARANCE_RADIUS + 1):
        for dy in range(-CLEARANCE_RADIUS, CLEARANCE_RADIUS + 1):
            if dx*dx + dy*dy <= CLEARANCE_RADIUS*CLEARANCE_RADIUS:
                nx = x + dx
                ny = y + dy
                if nx == car_x and ny == car_y:
                    continue
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

            # Mark the detected obstacle cell and interpolate with the previous point for continuity
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                mark_cell(x, y)
                if previous_point is not None:
                        interpolate(previous_point, (x, y))
                previous_point = (x, y)
            else:
                previous_point = None
        else:
            previous_point = None

    # Reset pan servo to center after scan
    px.set_cam_pan_angle(0)

    # Current pose and obstacle count
    print('Pose:', car_x, car_y, 'theta(deg):', round(math.degrees(car_theta), 1))
    print('Obstacle cells:', int(grid_map.sum()))

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

def drive_forward(seconds, speed=DRIVE_SPEED):
    """
    Drive straight forward for `seconds`, then do a tiny reverse “brake tap”
    to reduce rolling. Update position estimate based on time driven.
    
    param seconds: Duration to drive forward in seconds
    type seconds: float
    param speed: Speed to drive at (0-100)
    type speed: int
    """
    px.forward(speed)
    time.sleep(seconds)
    px.stop()

    # Small reverse tap to reduce drift from coasting
    px.backward(speed)
    time.sleep(0.05)
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

def reverse_no_pose(seconds, speed=DRIVE_SPEED):
    """
    Reverse the car for a short duration without updating position estimate (used for small corrections during turning).

    param seconds: Duration to reverse in seconds
    type seconds: float
    param speed: Speed to reverse at
    type speed: int
    """
    px.backward(speed)
    time.sleep(seconds)
    px.stop()

def turn_left(seconds):
    """
    Turn the car left for a specified duration while updating the pose estimate based on an arc motion model.

    param seconds: Duration to turn in seconds
    type seconds: float
    """
    px.set_dir_servo_angle(-30)
    px.forward(TURN_SPEED)

    v = LEFT_TURN_FWD_CM_PER_SEC    # Forward speed during left turn (cm/s)
    omega = v / LEFT_TURN_RADIUS_CM # Turning rate (rad/s) based on turn radius

    # Update our estimated pose in small time steps while the car is turning
    t = 0.0
    while t < seconds:
        dt = min(DT_TURN, seconds - t)
        time.sleep(dt)
        update_pose_arc(v, +omega, dt)
        t += dt

    px.stop()
    px.set_dir_servo_angle(0)

def turn_right(seconds):
    """
    Turn the car right for a specified duration while updating the pose estimate based on an arc motion model.

    param seconds: Duration to turn in seconds
    type seconds: float
    """
    px.set_dir_servo_angle(30)
    px.forward(TURN_SPEED)

    v = RIGHT_TURN_FWD_CM_PER_SEC
    omega = v / RIGHT_TURN_RADIUS_CM

    t = 0.0
    while t < seconds:
        dt = min(DT_TURN, seconds - t)
        time.sleep(dt)
        update_pose_arc(v, -omega, dt)
        t += dt

    px.stop()
    px.set_dir_servo_angle(0)

def rotate_to_heading(desired_theta):
    """
    Rotate the car in place to face the desired heading (desired_theta in radians).
    Uses a simple proportional control approach with a maximum turn step to ensure smooth rotation.
    After each turn step, it does a small reverse to help reduce drift and improve accuracy.
    
    param desired_theta: Desired heading angle in radians (0 to 2*pi)
    type desired_theta: float
    """
    tol = math.radians(3)
    max_pulse_deg = 15.0
    settle = 0.02
    backup_sec = 0.30

    for _ in range(8):
        diff = wrap_angle(desired_theta - car_theta)
        if abs(diff) <= tol:
            return

        step_deg = min(max_pulse_deg, abs(math.degrees(diff)))
        seconds = step_deg / (LEFT_TURN_DEG_PER_SEC if diff > 0 else RIGHT_TURN_DEG_PER_SEC)

        # Pulse turn
        if diff > 0:
            turn_left(seconds)
        else:
            turn_right(seconds)

        reverse_no_pose(backup_sec)

        # Small settle to keep it smooth
        px.stop()
        time.sleep(settle)

# TURN HELPERS
def wrap_angle(a):
    """
    Wrap angle to [-pi, pi].

    param a: Angle in radians
    type a: float
    return: Wrapped angle in radians
    rtype: float
    """
    return (a + math.pi) % (2 * math.pi) - math.pi

def update_pose_arc(speed_cm_s, turn_rate_rad_s, dt_s):
    """
    Update the car's pose when driving along an arc (turning).

    speed_cm_s: forward speed in cm/s
    turn_rate_rad_s: turning rate in rad/s (+ left, - right)
    dt_s: time step in seconds
    """
    global car_x, car_y, car_theta

    # Convert from grid cells to centimeters for smoother math
    x_pos_cm = car_x * CM_PER_CELL
    y_pos_cm = car_y * CM_PER_CELL
    heading_rad = car_theta

    if abs(turn_rate_rad_s) < 1e-6:
        # Straight motion
        x_pos_cm += speed_cm_s * dt_s * math.cos(heading_rad)
        y_pos_cm += speed_cm_s * dt_s * math.sin(heading_rad)
    else:
        # Arc motion (turning)
        turn_radius_cm = speed_cm_s / turn_rate_rad_s
        heading_change_rad = turn_rate_rad_s * dt_s

        x_pos_cm += turn_radius_cm * (math.sin(heading_rad + heading_change_rad) - math.sin(heading_rad))
        y_pos_cm -= turn_radius_cm * (math.cos(heading_rad + heading_change_rad) - math.cos(heading_rad))
        heading_rad += heading_change_rad

    # Save updated heading
    car_theta = heading_rad % (2 * math.pi)

    # Convert back to grid cells for planning/drawing
    car_x = int(round(x_pos_cm / CM_PER_CELL))
    car_y = int(round(y_pos_cm / CM_PER_CELL))
    clamp_pose()

# DRAWING
def draw_robot(ax, x, y, theta):
    """
    Draw the robot as a rectangle with a heading line.
    Robot is drawn OUTSIDE the map (below y=0), with its front at (x, y).
    """
    ROBOT_WIDTH = 8
    ROBOT_LENGTH = 12
    HEADING_LENGTH = 10

    # Center the rectangle on (x, y) with the front edge at y and the body extending downward
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
    
plt.ion()  # Interactive mode

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
    _ax.set_ylim(-ROBOT_MARGIN, MAP_SIZE)  # Show robot outside map for better visualization

    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)

# A* PATHFINDING
def heuristic(a, b):
    """
    Manhattan distance heuristic for grid-based pathfinding.

    param a: First node (x, y)
    type a: tuple
    param b: Second node (x, y)
    type b: tuple
    return: Manhattan distance between a and b
    rtype: int
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node):
    """
    Returns valid 4-direction neighbors (up, down, left, right).

    param node: Current node (x, y)
    type node: tuple
    return: List of neighboring nodes that are not obstacles
    rtype: list of tuples
    """
    x, y = node
    neighbors = []

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
            if grid_map[ny, nx] == 0:  # Not an obstacle
                neighbors.append((nx, ny))

    return neighbors

def astar(start, goal):
    """
    A* algorithm implementation.
    Returns path as list of (x, y) coordinates.

    param start: Starting node (x, y)
    type start: tuple
    param goal: Goal node (x, y)
    type goal: tuple
    return: List of nodes from start to goal, or None if no path found
    rtype: list of tuples or None
    """
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                heapq.heappush(open_set, (f, neighbor))

    return None  # No path found

def reactive_avoidance():
    """
    Executes the backup-and-turn maneuver from environment_scan.py
    """
    px.stop()
    time.sleep(0.3)
    
    # Back up
    reverse(1.0)
    time.sleep(0.3)
    
    # Choose another direction
    angle = random.choice([-35, 35])
    print(f'[AVOIDANCE] Turning to {angle} degrees')
    
    if angle < 0:
        turn_left(0.4)
    else:
        turn_right(0.4)
        
    # Move forward in new direction
    drive_forward(0.8)

def follow_path(path, goal):
    """
    Follow the given path of grid coordinates, driving the car accordingly.
    After each step, check if we've arrived at the goal (within a tolerance).

    param path: List of grid coordinates to follow
    type path: list of tuples
    param goal: Goal node (x, y)
    type goal: tuple
    """
    global car_theta, _stop_sign_handled

    seconds_per_cell = CM_PER_CELL / CM_PER_SEC

    for i in range(1, len(path)):
        current = path[i - 1]
        target = path[i]

        dx = target[0] - current[0]
        dy = target[1] - current[1]

        if dx == 1:
            desired_theta = 0
        elif dx == -1:
            desired_theta = math.pi
        elif dy == 1:
            desired_theta = math.pi / 2
        elif dy == -1:
            desired_theta = 3 * math.pi / 2
        else:
            continue

        rotate_to_heading(desired_theta)

        # Traffic Signs and Pedestrian Detection
        if is_stop_sign_detected() and not _stop_sign_handled:
            px.stop()
            print('[TRAFFIC] STOP SIGN detected → stopping 3 sec')
            time.sleep(3)
            _stop_sign_handled = True
        elif not is_stop_sign_detected():
            _stop_sign_handled = False

        while is_person_detected():
            px.stop()
            print('[TRAFFIC] Person detected → waiting')
            time.sleep(0.1)
            
        # Quick front ultrasonic scan to trigger avoidance before driving the cell
        distance = px.ultrasonic.read()
        if distance is not None and 0 < distance < 25: 
            print('[OBSTACLE] Immediate obstacle! Executing avoidance')
            reactive_avoidance()
            return  # Skip forward this cycle so A* replans from new position

        # Move exactly 1 cell
        drive_forward(seconds_per_cell)

        if at_goal(goal=goal, tol_cells=1):
            print('[GOAL] Arrived (during execution)')
            px.stop()
            return

        show_map()

def at_goal(goal, tol_cells=1):
    """
    Check if the car is within tol_cells of the goal.
    
    param goal: Goal node (x, y)
    type goal: tuple
    param tol_cells: Tolerance in grid cells to consider "at goal"
    type tol_cells: int
    """
    gx, gy = goal
    return abs(car_x - gx) <= tol_cells and abs(car_y - gy) <= tol_cells

def navigate_to_goal(goal, max_iters=50, steps_per_plan=STEPS_PER_PLAN):
    """
    Navigate to the specified goal using repeated A* planning and execution.

    param goal: Goal node (x, y)
    type goal: tuple
    param max_iters: Maximum number of planning iterations
    type max_iters: int
    param steps_per_plan: Number of steps to execute per plan
    type steps_per_plan: int
    return: True if goal reached, False if failed
    rtype: bool
    """
    global grid_map

    if at_goal(goal, tol_cells=1):
        print('[GOAL] Already at goal')
        px.stop()
        return True

    for _ in range(max_iters):
        # Update the environment
        scan_environment()

        grid_map *= 0.8                 # Slowly "forget" old obstacles so the map doesn't get stuck with noise forever
        grid_map[grid_map < 0.2] = 0    # If it's faint, delete it entirely

        # Find path to goal using A*
        start = (car_x, car_y)
        path = astar(start, goal)

        if not path:
            print('[PATH] No path found')
            return False

        # Planned path
        print('[PATH] Planned path:', path[:10], '... len =', len(path))

        if len(path) <= 1:
            print('[GOAL] Arrived')
            return True

        # Execute a segment of the planned path
        execute_steps = min(steps_per_plan, len(path)-1)
        follow_path(path[:execute_steps+1], goal) 

    print('[PATH] Gave up after max attempts')
    return False


if __name__ == "__main__":
    main()