from simulation.city_sim import CitySimulation
import time

sim = CitySimulation(grid_size=3)

state = sim.reset()

for _ in range(20):
    state = sim.step()
    print(state)
    print("-" * 20)
    time.sleep(0.5)