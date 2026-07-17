import carla
import random

client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# 1. Spawn the "Street Animal" (Using a dog/small prop as proxy)
animal_bp = bp_lib.find('walker.pedestrian.0001') # Placeholder for street element
spawn_point = carla.Transform(carla.Location(x=-5, y=135, z=2))
world.try_spawn_actor(animal_bp, spawn_point)

# 2. Spawn a "Broken Down" Vehicle (The Accident)
accident_bp = bp_lib.find('vehicle.tesla.model3')
accident_loc = carla.Transform(carla.Location(x=0, y=130, z=2))
car = world.spawn_actor(accident_bp, accident_loc)
car.set_simulate_physics(True)
# This car won't move, creating a permanent bottleneck for the AI to solve