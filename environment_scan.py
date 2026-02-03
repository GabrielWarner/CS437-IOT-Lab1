from picarx import Picarx
import time
import random

def main():
    """
    Main loop for environment scanning and obstacle avoidance.
    Distance readings are in cm.
    """
    px = Picarx()
    print('PiCar-X Environment Scanning Started')

    try:
        while True:
            distance = px.ultrasonic.read()

            if distance <= 0:   # Ignore invalid readings
                time.sleep(0.05)
                continue 

            print(f'Distance: {distance:.1f} cm')

            if distance > 50:    
                px.set_dir_servo_angle(0)
                px.forward(30)
            else:
                print('Obstacle detected.')

                # Stop
                px.forward(0)
                time.sleep(0.3)

                # Choose another direction
                angle = random.choice([-30, 30])
                print(f'Turning to {angle} degrees')

                # Back up
                px.set_dir_servo_angle(0)
                px.backward(40)
                time.sleep(1.0)
                px.forward(0)
                time.sleep(0.3)

                # Turn
                px.set_dir_servo_angle(angle)
                time.sleep(0.4)

                # Move foward in new direction
                px.forward(30)
                time.sleep(0.8)

                # Straighten and continue
                px.set_dir_servo_angle(0)

            time.sleep(0.05)


    except KeyboardInterrupt:
        print('\nProgram stopped by user')

    finally:
        px.forward(0)
        px.set_dir_servo_angle(0)


if __name__ == "__main__":
    main()