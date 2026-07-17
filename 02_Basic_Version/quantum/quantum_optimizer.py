import numpy as np


def optimize_signal(lane_counts):
    """
    Quantum-inspired optimization using QUBO formulation.
    Fast, stable, and suitable for RL loop.
    """

    num_lanes = len(lane_counts)

    # Build QUBO matrix
    Q = np.zeros((num_lanes, num_lanes))

    # Objective: maximize flow (negative because we minimize energy)
    for i in range(num_lanes):
        Q[i, i] = -lane_counts[i]

    # Constraint penalty (only one lane active)
    penalty = 10
    for i in range(num_lanes):
        for j in range(num_lanes):
            if i != j:
                Q[i, j] += penalty

    # Evaluate all binary states (small problem → feasible)
    best_energy = float("inf")
    best_idx = 0

    for i in range(num_lanes):
        x = np.zeros(num_lanes)
        x[i] = 1

        energy = x @ Q @ x

        if energy < best_energy:
            best_energy = energy
            best_idx = i

    return best_idx