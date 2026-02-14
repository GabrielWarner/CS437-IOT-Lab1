import numpy as np
import math
import time
from picarx import Picarx

MAP_SIZE = 100            # 100 x 100 grid
MAX_DISTANCE = 100        # Max valid ultrasonic sensor reading (cm)
CM_PER_CELL = 1.0          
OBSTACLE_THRESHOLD = 50   # Distance threshold to mark cells as obstacles (cm)
ANGLE_START = -60
ANGLE_END = 60
ANGLE_STEP = 5

px = Picarx()
grid_map = np.zeros((MAP_SIZE, MAP_SIZE))

# Car position (cm, radians)
# car_x = 0.0
# car_y = 0.0
car_x = MAP_SIZE // 2
car_y = 0
car_theta = math.pi / 2   # Car starts facing the positive y direction (90 degrees)

def main():
    """
    Drive the car forward while repeatedly scanning the environment
    and updating the internal map and estimated position.
    """
    try:
        # px.forward(30)

        # 5 iterations of scanning and updating the map
        for i in range(5):
            scan_environment()
            print(f'Map after scan {i+1}:\n{grid_map}')
            time.sleep(1) 

    except KeyboardInterrupt:
        print('Stopped by user')

    finally:
        px.stop()

def mark_cell(x, y):
    """
    Marks a single cell in the map as occupied (1) 
    only when the ultrasonic sensor detects an object 
    within the obstacle threshold.

    param x: X coordinate in cm
    type x: int
    param y: Y coordinate in cm
    type y: int
    """
    if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
        grid_map[y, x] = 1

def interpolate(point1, point2):
    """
    This performs linear interpolation between two Cartesian points
    by filling in intermediate map cells between two detected obstacle points
    to approximate a continuous obstacle surface detected by
    consecutive ultrasonic measurements.

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


# gets average of 5 readings to reduce noise
def read_distance_med(samples=5, delay=0.02):
    vals = []
    for _ in range(samples):
        d = px.ultrasonic.read()
        #reject outliers and invalid readings
        if d is not None and 0 < d <= MAX_DISTANCE:
            vals.append(d)
        time.sleep(delay)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals)//2]


def scan_environment():
    """
    Perform a sweep of the ultrasonic sensor using the pan servo
    across a range of angles to detect nearby obstacles.
    """
    previous_point = None

    for servo_angle in range(ANGLE_START, ANGLE_END + 1, ANGLE_STEP):
        px.set_cam_pan_angle(servo_angle)
        time.sleep(0.15)
        _ = px.ultrasonic.read()   # discard one
        time.sleep(0.03)
        distance = read_distance_med()  # Use median reading to reduce noise

        print(f'Angle: {servo_angle}, Distance: {distance}')

        if distance is None:
            previous_point = None
            continue

        if distance <= OBSTACLE_THRESHOLD:
            distance_cells = distance / CM_PER_CELL

            # Convert polar to Cartesian coordinates
            angle_radians = math.radians(servo_angle) + car_theta
            x = int(round(car_x + distance_cells * math.cos(angle_radians)))
            y = int(round(car_y + distance_cells * math.sin(angle_radians)))

            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                # Mark obstacle
                mark_cell(x, y)

                # Interpolate with previous detection if available
                if previous_point is not None:
                    px_prev, py_prev = previous_point
                    if abs(px_prev - x) <= 10 and abs(py_prev - y) <= 10:  # Only interpolate if points are close
                        interpolate(previous_point, (x, y))
                    else:
                        pass

                previous_point = (x, y)
            else:
                previous_point = None
        else:
            previous_point = None
    # Reset pan servo to center after scan
    px.set_cam_pan_angle(0)


if __name__ == "__main__":
    main()
