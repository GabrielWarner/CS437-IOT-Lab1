from picarx import Picarx
import time

POWER = 50
SafeDistance = 40   # > 40 safe
DangerDistance = 20 # > 20 && < 40 turn around, 
                    # < 20 backward

def main():
    try:
        px = Picarx()
       
        while True:
            distance = round(px.ultrasonic.read(), 2)
            print("distance: ", distance)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

