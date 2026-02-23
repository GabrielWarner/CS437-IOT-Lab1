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
import object_detection

# Constants
MAP_SIZE = 20            # 20x20 grid
MAX_DISTANCE = 100        # max valid ultrasonic sensor reading (cm)
CM_PER_CELL = 5          
ANGLE_START = -60
ANGLE_END = 60
ANGLE_STEP = 5
OBSTACLE_THRESHOLD = 100   # distance threshold to mark cells as obstacles (cm)
CM_PER_SEC = 20.0        # measured with car moving forward at  using tape measure
DRIVE_SPEED = 5          
TURN_DEG_PER_SEC = 35.0   # rough estimate of turning speed in degrees per second (feel free to adjust based on testing)
_robot_artists = []  # holds the rectangle + line so we can remove them
ROBOT_MARGIN = 15  # extra visible space below the map
CLEARANCE_RADIUS = 0 # increase or decrease based on map size
STEPS_PER_PLAN = 3   # how many steps execute before replanning

# Turn tuning
TURN_SPEED = DRIVE_SPEED
LEFT_TURN_DEG_PER_SEC  = 45.0
RIGHT_TURN_DEG_PER_SEC = 34.0
DT_TURN = 0.02 # delta time for turn pose updates (seconds)
# Forward drift during turning (cm per second)
RIGHT_TURN_FWD_CM_PER_SEC = 16.0
LEFT_TURN_FWD_CM_PER_SEC  = 13.0
LEFT_TURN_RADIUS_CM  = 16.6
RIGHT_TURN_RADIUS_CM = 27.0

_stop_sign_handled = False

# Initialize the car and the map
px = Picarx()
grid_map = np.zeros((MAP_SIZE, MAP_SIZE))

# Car position (cm, radians)
car_x = MAP_SIZE // 2
car_y = 0
car_theta = math.pi / 2  # car starts facing the positive y direction (90 degrees)

def main():
    global grid_map
    

    # --- ADDED: Start your object detection in a background thread ---
    print("Starting object detection camera thread...")
    stop_event = threading.Event()
    od_thread = threading.Thread(
        target=object_detection.run,
        args=('efficientdet_lite0.tflite', 0, 640, 480, 4, False, stop_event),
        daemon=True
    )
    od_thread.start()
    # -----------------------------------------------------------------

    try:
        grid_map[:] = 0
        # goal grid coordinates (x, y) in cells

        goal = (12, 18)
        navigate_to_goal(goal=goal, steps_per_plan=STEPS_PER_PLAN)

        # """
        # Drive the car forward while repeatedly scanning the environment.
        # Runs for a fixed number of iterations or until stopped by the user.
        # """
        # try:
        #     for step in range(5):
        #         print(f'Main loop iteration: {step + 1}/5')

        #         decision = scan_environment()
        #         print(f'\nDecision: {decision}\n')

        #         if decision == 'center':
        #             px.stop()
        #             reverse(0.5)  # back up a bit
        #             turn_left(0.5)   # obstacle directly ahead
        #         elif decision == 'left':
        #             px.stop()
        #             reverse(0.5)  
        #             turn_right(0.4) # obstacle more on the left
        #         elif decision == 'right':
        #             px.stop()
        #             reverse(0.5)
        #             turn_left(0.4)  # obstacle more on the right
        #         else:
        #             drive_forward(0.60)  # path is clear

        #         time.sleep(0.2)

        # except KeyboardInterrupt:
        #     print('Stopped by user')

        # finally:
        #         px.stop()
    finally:
        # --- ADDED: Cleanly stop the detection thread when done ---
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
    global grid_map
    r = CLEARANCE_RADIUS
    y_min, y_max = max(0, y-r), min(MAP_SIZE, y+r+1)
    x_min, x_max = max(0, x-r), min(MAP_SIZE, x+r+1)
    grid_map[y_min:y_max, x_min:x_max] = 1

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
        # Reject outliers and invalid readings
        if d is not None and 0 < d <= MAX_DISTANCE:
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

    FRONT_ANGLE_WINDOW = 10     # consider obstacles within +/- 10 degrees as "ahead"
    FRONT_STOP_CM = 25          # if an obstacle is detected within this distance in front, consider it blocked

    for servo_angle in range(ANGLE_START, ANGLE_END + 1, ANGLE_STEP):
        px.set_cam_pan_angle(servo_angle)
        time.sleep(0.15)

        _ = px.ultrasonic.read()   # discard first reading after moving servo to allow it to stabilize
        time.sleep(0.03)

        distance = read_distance_median()  # get a more reliable distance reading using the median
        print(f'Angle: {servo_angle}, Distance: {distance}')

        if distance is not None and previous_distance is not None:
            if abs(distance - previous_distance) > 8:
                previous_point = None  # break interpolation if jump is large

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

            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                # Mark obstacle
                mark_cell(x, y)

                # Interpolate with previous detection if available
                if previous_point is not None:
                        interpolate(previous_point, (x, y))
                previous_point = (x, y)

            else:
                previous_point = None
        else:
            previous_point = None

    # CHANGED: Moved to follow_path so the map updates after the car moves.
    # show_map()

    # Reset pan servo to center after scan
    px.set_cam_pan_angle(0)

    # Debug print of current pose and obstacle count
    print("Pose:", car_x, car_y, "theta(deg):", round(math.degrees(car_theta), 1))
    print("Obstacle cells:", int(grid_map.sum()))

    # Decide on action based on hit counts
    if center_hits > 0:
        return 'center'
    elif left_hits > right_hits:
        return 'left'
    elif right_hits > left_hits:
        return 'right'
    else:
        return None # path is clear

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

    # brake tap to reduce coasting
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

    # Update position estimate (reverse is negative forward motion)
    update_position(-CM_PER_SEC * seconds)

# reverse that doesnt update position estimate, used for turns
def reverse_no_pose(seconds, speed=DRIVE_SPEED):
    px.backward(speed)
    time.sleep(seconds)
    px.stop()

def turn_left(seconds):
    px.set_dir_servo_angle(-30)
    px.forward(TURN_SPEED)
    # Use our measured forward speed during a left turn
    v = LEFT_TURN_FWD_CM_PER_SEC
    # Convert turn radius into turn rate (radians per second)
    omega = v / LEFT_TURN_RADIUS_CM
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

# Turns in pulses and reverse a bit in between to reduce drift, repeat until we're facing the desired angle within a tolerance.
def rotate_to_heading(desired_theta):
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

        # pulse turn
        if diff > 0:
            turn_left(seconds)
        else:
            turn_right(seconds)

        reverse_no_pose(backup_sec)

        # small settle to keep it smooth
        px.stop()
        time.sleep(settle)

# TURN HELPERS

def wrap_angle(a):
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

# Robot drawing + map display
def draw_robot(ax, x, y, theta):
    """
    Draw the robot as a rectangle with a heading line.
    Robot is drawn OUTSIDE the map (below y=0), with its front at (x, y).
    """
    ROBOT_WIDTH = 8
    ROBOT_LENGTH = 12
    HEADING_LENGTH = 10

    # CHANGED: draw the body BELOW the front point so it's outside the map
    rect = patches.Rectangle(
        (x - ROBOT_WIDTH / 2, y - ROBOT_LENGTH),  # body extends downward
        ROBOT_WIDTH,
        ROBOT_LENGTH,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none"
    )
    ax.add_patch(rect)

    hx = x + HEADING_LENGTH * math.cos(theta)
    hy = y + HEADING_LENGTH * math.sin(theta)
    line, = ax.plot([x, hx], [y, hy], color="cyan", linewidth=2)

    return [rect, line]
    
plt.ion()  # interactive mode

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
    _ax.set_ylim(-ROBOT_MARGIN, MAP_SIZE)  # show robot outside map for better visualization

    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)

# A* Pathfinding implementation
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def get_neighbors(node):
    """
    Returns valid 4-direction neighbors (up, down, left, right).
    """
    x, y = node
    neighbors = []

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
            if grid_map[ny, nx] == 0:  # not an obstacle
                neighbors.append((nx, ny))

    return neighbors

def astar(start, goal):
    """
    A* algorithm implementation.
    Returns path as list of (x, y) coordinates.
    """

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # reconstruct path
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

    return None  # no path found

# Avoidance logic
def reactive_avoidance():
    """Executes the backup-and-turn maneuver from environment_scan.py"""
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

# Execute the first part of the path, then replan
def follow_path(path, goal):
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
        stop_detected = is_stop_sign_detected()

        if is_stop_sign_detected() and not _stop_sign_handled:
            px.stop()
            print("[TRAFFIC] STOP SIGN detected → stopping 3 sec")
            time.sleep(3)
            _stop_sign_handled = True
        elif not is_stop_sign_detected():
            _stop_sign_handled = False

        while is_person_detected():
            px.stop()
            print("[TRAFFIC] Person detected → waiting")
            time.sleep(0.1)
            
        # Quick front ultrasonic scan to trigger avoidance before driving the cell
        distance = px.ultrasonic.read()
        if distance is not None and 0 < distance < 25: 
            print("[OBSTACLE] Immediate obstacle! Executing avoidance")
            reactive_avoidance()
            return  # skip forward this cycle so A* replans from new position

        # move exactly 1 cell
        drive_forward(seconds_per_cell)

        if at_goal(goal=goal, tol_cells=1):
            print("Arrived (during execution)")
            px.stop()
            return

        show_map()

def at_goal(goal, tol_cells=1):
    gx, gy = goal
    return abs(car_x - gx) <= tol_cells and abs(car_y - gy) <= tol_cells

# Main navigation loop: repeatedly scan, plan, and execute steps towards the goal
def navigate_to_goal(goal, max_iters=50, steps_per_plan=STEPS_PER_PLAN):
    global grid_map

    if at_goal(goal, tol_cells=1):
        print("Already at goal")
        px.stop()
        return True

    for _ in range(max_iters):
        # rescan and rebuild map around current pose
        grid_map[:] = 0
        scan_environment()   # fills grid_map with obstacles (+ clearance)

        # find path to goal using A*
        start = (car_x, car_y)
        path = astar(start, goal)

        if not path:
            print("No path found")
            return False

        # Debug print of planned path
        print("Planned path:", path[:10], "... len =", len(path))

        if len(path) <= 1:
            print("Arrived")
            return True

        # executes # of STEPS_PER_PLAN steps along the path, then replans
        execute_steps = min(steps_per_plan, len(path)-1)
        follow_path(path[:execute_steps+1], goal)  # just a prefix

    print("Gave up after max attempts")
    return False

if __name__ == "__main__":
    main()