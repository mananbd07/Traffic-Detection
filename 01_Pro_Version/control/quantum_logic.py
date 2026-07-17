import numpy as np

def calculate_quantum_pressure(phases_queues):
    """
    Inspired by Quantum Superposition: 
    We evaluate all possible light phases simultaneously to find the 
    'Lowest Energy State' (Minimum Congestion).
    """
    # phases_queues is a list of vehicle counts for each direction
    # e.g., [North-South, East-West]
    
    # We use a Softmax-style 'Energy' calculation
    exp_queues = np.exp(phases_queues)
    probabilities = exp_queues / np.sum(exp_queues)
    
    # The 'Winning' phase is the one with the highest pressure
    # because it needs the Green light the most.
    recommended_phase = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    return recommended_phase, confidence