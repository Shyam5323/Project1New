
# Example environment check script
import qiskit
import medmnist
import numpy as np
import matplotlib.pyplot as plt

print(f"Qiskit version: {qiskit.__version__}")
print(f"MedMNIST available datasets: {medmnist.INFO}")

# Test basic quantum circuit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create a simple test circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Test simulation
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit, shots=1000).result()
counts = result.get_counts()
print(f"Test circuit results: {counts}")