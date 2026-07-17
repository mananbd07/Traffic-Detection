import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import torch.nn as nn

class QuantumTrafficHead(nn.Module):
    def __init__(self, num_qubits=4):
        super().__init__()
        
        # 1. Define the Quantum Circuit
        # ZZFeatureMap encodes the "State" (Traffic density)
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
        
        # RealAmplitudes is the "Ansatz" (The trainable weights)
        ansatz = RealAmplitudes(num_qubits, reps=1)
        
        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        # 2. Define the QNN (Quantum Neural Network)
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )
        
        # 3. Connect Qiskit to PyTorch (Compatible with Stable Baselines 3)
        self.qnn_layer = TorchConnector(qnn)
        
    def forward(self, x):
        # x is the traffic data from the classical layers
        return self.qnn_layer(x)