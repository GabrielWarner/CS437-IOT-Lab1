import numpy as np
import math
import time
from picarx import Picarx
import matplotlib.pyplot as plt

# Constants
MAP_SIZE = 100            # 100 x 100 grid
MAX_DISTANCE = 100        # max valid ultrasonic sensor reading (cm)
CM_PER_CELL = 1.0          
ANGLE_START = -60
ANGLE_END = 60
ANGLE_STEP = 5
OBSTACLE_THRESHOLD = 50   # distance threshold to mark cells as obstacles (cm)
CM_PER_SEC = 10.0         # assumed speed of the car in cm/s when driving forward
DRIVE_SPEED = 30          
TURN_DEG_PER_SEC = 45.0   # assumed turn rate in degrees per second when turning

# Initialize the car and the map
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
                reverse(0.5)  # back up a bit
                turn_left(0.5)   # obstacle directly ahead
            elif decision == 'left':
                px.stop()
                reverse(0.5)  
                turn_right(0.4) # obstacle more on the left
            elif decision == 'right':
                px.stop()
                reverse(0.5)
                turn_left(0.4)  # obstacle more on the right
            else:
                drive_forward(0.60)  # path is clear

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

    # Update position estimate (reverse is negative forward motion)
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
    
plt.ion()  # interactive mode

_fig, _ax = plt.subplots()
_img = _ax.imshow(grid_map, origin='lower', vmin=0, vmax=1)
_ax.set_title('Obstacle Map (1=obstacle)')

plt.show()

def show_map():
    """
    Update the displayed map with the current grid_map data.
    """
    _img.set_data(grid_map)
    _ax.set_xlabel('x (cells)')
    _ax.set_ylabel('y (cells)')
    _fig.canvas.draw()
    _fig.canvas.flush_events()


if __name__ == "__main__":
    main()