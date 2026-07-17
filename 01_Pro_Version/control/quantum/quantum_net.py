import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class QuantumLayer9Q(nn.Module):
    def __init__(self, n_qubits=9):
        super().__init__()
        # 1. Map 9 traffic density inputs to 9 qubits
        fm = ZZFeatureMap(n_qubits, reps=1, entanglement='linear')
        
        # 2. Advanced Ansatz: Circular entanglement for 3x3 grid logic
        ansatz = RealAmplitudes(n_qubits, reps=2, entanglement='circular')
        
        qc = QuantumCircuit(n_qubits)
        qc.compose(fm, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=fm.parameters,
            weight_params=ansatz.parameters
        )
        
        self.qnn_bridge = TorchConnector(qnn)

    def forward(self, x):
        return self.qnn_bridge(x)