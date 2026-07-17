import numpy as np
from utils.config import *

class TrafficSimulation:
    def __init__(self):
        self.time = 0

        # Two intersections
        self.lanes_A = np.zeros(NUM_LANES)
        self.lanes_B = np.zeros(NUM_LANES)

        self.wait_A = np.zeros(NUM_LANES)
        self.wait_B = np.zeros(NUM_LANES)

    def reset(self):
        self.time = 0

        self.lanes_A = np.zeros(NUM_LANES)
        self.lanes_B = np.zeros(NUM_LANES)

        self.wait_A = np.zeros(NUM_LANES)
        self.wait_B = np.zeros(NUM_LANES)

        return self.get_state()

    def step(self, action_A, action_B):
        self.time += 1

        # 🔥 Traffic arrival (with pattern)
        base_prob = CAR_ARRIVAL_PROB

        if 50 < self.time % 200 < 100:
            base_prob *= 2  # rush hour

        arrivals_A = np.random.binomial(3, base_prob, NUM_LANES)
        arrivals_B = np.random.binomial(3, base_prob, NUM_LANES)

        self.lanes_A += arrivals_A
        self.lanes_B += arrivals_B

        # Cap max cars
        self.lanes_A = np.clip(self.lanes_A, 0, MAX_CARS)
        self.lanes_B = np.clip(self.lanes_B, 0, MAX_CARS)

        # 🔥 Signal logic
        capacity = 5

        # Intersection A
        passed_A = min(self.lanes_A[action_A], capacity)
        self.lanes_A[action_A] -= passed_A

        # Some cars move to B
        flow_to_B = int(passed_A * 0.6)
        self.lanes_B[action_B] += flow_to_B

        # Intersection B
        passed_B = min(self.lanes_B[action_B], capacity)
        self.lanes_B[action_B] -= passed_B

        # Update waiting times
        self.wait_A += self.lanes_A
        self.wait_B += self.lanes_B

        return self.get_state()

    def get_state(self):
        return np.concatenate([
            self.lanes_A,
            self.lanes_B,
            self.wait_A,
            self.wait_B
        ])