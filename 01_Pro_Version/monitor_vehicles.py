import carla
import time
import argparse

def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    
    vehicles = world.get_actors().filter('vehicle.*')
    print(f"Found {len(vehicles)} vehicles in the world.")
    
    for i in range(5):
        if len(vehicles) == 0:
            break
        speeds = [v.get_velocity().length() for v in vehicles]
        print(f"Tick {i}: Speeds (m/s): {speeds}")
        time.sleep(1)

if __name__ == '__main__':
    main()
