import random
import numpy as np

class Vehicle:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    def next_position(self):
        r, c = self.position
        if self.direction == "N": return (r - 1, c)
        elif self.direction == "S": return (r + 1, c)
        elif self.direction == "E": return (r, c + 1)
        elif self.direction == "W": return (r, c - 1)

class TrafficSignal:
    def __init__(self):
        self.state = random.choice(["NS", "EW"])
        self.timer = 0

    def set(self, state, duration):
        self.state = state
        self.timer = duration

    def step(self):
        if self.timer > 0: self.timer -= 1

class CitySimulation:
    def __init__(self, grid_size=3, max_vehicles=150): # Increased max vehicles for stress
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.vehicles = []
        self.time = 0
        self.signals = [[TrafficSignal() for _ in range(grid_size)] for _ in range(grid_size)]
        self.scenario = "random" # Default

    def set_scenario(self, mode):
        self.scenario = mode # "morning_rush", "evening_rush", "heavy_gridlock"

    def spawn_vehicles(self):
        if len(self.vehicles) >= self.max_vehicles: return

        # Scenario Logic
        spawn_rate = 0.8 # High intensity
        if random.random() < spawn_rate:
            if self.scenario == "morning_rush":
                # Heavy traffic coming from Top (North) moving South
                edge = "top" if random.random() < 0.8 else random.choice(["left", "right"])
            elif self.scenario == "evening_rush":
                # Heavy traffic coming from Left (West) moving East
                edge = "left" if random.random() < 0.8 else random.choice(["top", "bottom"])
            else:
                edge = random.choice(["top", "bottom", "left", "right"])

            if edge == "top":
                pos, direct = (0, random.randint(0, self.grid_size - 1)), "S"
            elif edge == "bottom":
                pos, direct = (self.grid_size - 1, random.randint(0, self.grid_size - 1)), "N"
            elif edge == "left":
                pos, direct = (random.randint(0, self.grid_size - 1), 0), "E"
            else:
                pos, direct = (random.randint(0, self.grid_size - 1), self.grid_size - 1), "W"

            self.vehicles.append(Vehicle(pos, direct))

    def can_move(self, vehicle):
        r, c = vehicle.position
        signal = self.signals[r][c]
        # Logic fix: If timer is 0, it means the light just expired or is neutral.
        # To keep it extraordinary, we assume timer > 0 is a COMMANDED state.
        if signal.timer > 0:
            if vehicle.direction in ["N", "S"] and signal.state == "NS": return True
            if vehicle.direction in ["E", "W"] and signal.state == "EW": return True
        return False

    def step(self):
        self.time += 1
        for row in self.signals:
            for s in row: s.step()
        
        self.spawn_vehicles()
        new_vehicles, moved = [], 0

        for v in self.vehicles:
            if not self.can_move(v):
                new_vehicles.append(v)
                continue
            
            next_pos = v.next_position()
            r, c = next_pos
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                v.position = next_pos
                new_vehicles.append(v)
                moved += 1
        
        self.vehicles = new_vehicles
        return self.get_state(), moved

    def get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for v in self.vehicles:
            r, c = v.position
            grid[r][c] += 1
        return grid

    def reset(self):
        self.vehicles = []
        self.time = 0
        return self.get_state()