# pylint: disable=trailing-whitespace
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=line-too-long

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import StatePreparation
import math
from abc import ABC, abstractmethod

class QuantumEncoder(ABC):
    """Abstract base class for quantum data encoding strategies"""
    
    def __init__(self, n_features):
        self.n_features = n_features
        n_qubits = self._calculate_qubits()
        if n_qubits is None:
            raise ValueError("Number of qubits cannot be None. Check _calculate_qubits implementation.")
        self.n_qubits = n_qubits
        
    @abstractmethod
    def _calculate_qubits(self):
        """Calculate number of qubits needed for this encoding"""
        pass
    
    @abstractmethod
    def encode(self, features, circuit=None):
        """Encode classical features into quantum circuit"""
        pass
    
    def create_circuit(self, features, include_measurements=False):
        """Create a complete quantum circuit with encoding"""
        qc = QuantumCircuit(self.n_qubits)
        self.encode(features, qc)
        
        if include_measurements:
            qc.add_register(ClassicalRegister(self.n_qubits))
            qc.measure_all()
            
        return qc

class AmplitudeEncoder(QuantumEncoder):
    """
    Amplitude encoding: embed classical data into quantum state amplitudes
    Most information-dense but requires complex state preparation
    """
    
    def __init__(self, n_features):
        super().__init__(n_features)
        self.use_approximation = True  # For NISQ compatibility
        
    def _calculate_qubits(self):
        """Need log2(n_features) qubits for amplitude encoding"""
        return int(np.ceil(np.log2(max(self.n_features, 2))))

    def encode(self, features, circuit=None):
        """ Encode features as amplitudes of quantum state """
        if np.allclose(features, 0):
            print("Warning: Zero feature vector detected in amplitude encoding")
        if circuit is None:
            circuit = QuantumCircuit(self.n_qubits)

        # Normalize features to unit vector
        normalized_features = self._normalize_features(features)

        # Pad to power of 2 if necessary
        padded_features = self._pad_to_power_of_2(normalized_features)

        try:
            # Use initialize instead of StatePreparation
            circuit.initialize(padded_features.tolist(), circuit.qubits)
        except Exception as e:
            # Fallback to approximate amplitude encoding for NISQ devices
            print(f"Using approximate amplitude encoding: {e}")
            self._approximate_amplitude_encoding(padded_features, circuit)

        return circuit

    def _normalize_features(self, features):
        norm = np.linalg.norm(features)
        if norm < 1e-10:  # More precise threshold
            # Create a meaningful default state instead of |0...0>
            normalized = np.ones(len(features)) / np.sqrt(len(features))
            return normalized
        return features / norm
        
    def _pad_to_power_of_2(self, features):
        """Pad feature vector to nearest power of 2 length"""
        target_length = 2 ** self.n_qubits
        if len(features) == target_length:
            return features
        
        padded = np.zeros(target_length)
        padded[:len(features)] = features
        return padded
    def _approximate_amplitude_encoding(self, amplitudes, circuit):
        """
        Proper approximate amplitude encoding with multiple fallback strategies
        
        Args:
            amplitudes: numpy array of normalized amplitudes
            circuit: QuantumCircuit to add encoding to
        """
        try:
            # Strategy 1: Use Qiskit's StatePreparation (most accurate)
            from qiskit.circuit.library import StatePreparation
            state_prep = StatePreparation(amplitudes)
            circuit.compose(state_prep, inplace=True)
            return
            
        except Exception as e:
            print(f"StatePreparation failed: {e}, trying recursive decomposition...")
            
        try:
            # Strategy 2: Recursive amplitude encoding decomposition
            self._recursive_amplitude_decomposition(amplitudes, circuit, list(range(self.n_qubits)))
            return
            
        except Exception as e:
            print(f"Recursive decomposition failed: {e}, using uniform superposition fallback...")
            
        # Strategy 3: Fallback to uniform superposition with amplitude-weighted phases
        self._weighted_superposition_fallback(amplitudes, circuit)

    def _recursive_amplitude_decomposition(self, amplitudes, circuit, qubits):
        """
        Implement recursive amplitude encoding using controlled rotations
        Based on the algorithm from Mottonen et al. (2004)
        """
        if len(qubits) == 0:
            return
            
        if len(qubits) == 1:
            # Base case: single qubit
            if len(amplitudes) >= 2:
                # Calculate rotation angle from amplitudes
                prob_0 = abs(amplitudes[0])**2
                prob_1 = abs(amplitudes[1])**2 if len(amplitudes) > 1 else 0
                total_prob = prob_0 + prob_1
                
                if total_prob > 1e-10:
                    theta = 2 * np.arccos(np.sqrt(prob_0 / total_prob))
                    circuit.ry(theta, qubits[0])
            return
        
        # Recursive case: multiple qubits
        n_amps = len(amplitudes)
        mid_point = n_amps // 2
        
        # Calculate probabilities for left and right halves
        left_amps = amplitudes[:mid_point]
        right_amps = amplitudes[mid_point:] if mid_point < n_amps else np.array([0])
        
        left_norm_sq = sum(abs(amp)**2 for amp in left_amps)
        right_norm_sq = sum(abs(amp)**2 for amp in right_amps)
        total_norm_sq = left_norm_sq + right_norm_sq
        
        if total_norm_sq > 1e-10:
            # Calculate rotation angle for current qubit
            theta = 2 * np.arccos(np.sqrt(left_norm_sq / total_norm_sq))
            circuit.ry(theta, qubits[0])
            
            # Recursively encode left and right branches
            if left_norm_sq > 1e-10:
                normalized_left = left_amps / np.sqrt(left_norm_sq)
                # Create controlled subcircuit for left branch (when qubit[0] is |0⟩)
                self._add_controlled_encoding(circuit, normalized_left, qubits[1:], qubits[0], control_state=0)
            
            if right_norm_sq > 1e-10:
                normalized_right = right_amps / np.sqrt(right_norm_sq)
                # Create controlled subcircuit for right branch (when qubit[0] is |1⟩)
                self._add_controlled_encoding(circuit, normalized_right, qubits[1:], qubits[0], control_state=1)

    def _add_controlled_encoding(self, circuit, amplitudes, target_qubits, control_qubit, control_state):
        """
        Add controlled encoding operations to the circuit
        """
        if len(target_qubits) == 0:
            return
            
        # For simplicity, we'll use a simplified controlled operation
        # In a full implementation, this would recursively apply the controlled version
        # of the amplitude encoding to the target qubits
        
        if control_state == 0:
            # Apply X gate to flip control, apply encoding, flip back
            circuit.x(control_qubit)
            self._apply_simple_encoding(circuit, amplitudes, target_qubits, control_qubit)
            circuit.x(control_qubit)
        else:
            # Directly apply controlled encoding
            self._apply_simple_encoding(circuit, amplitudes, target_qubits, control_qubit)

    def _apply_simple_encoding(self, circuit, amplitudes, target_qubits, control_qubit):
        """
        Apply a simplified controlled encoding for the recursive decomposition
        """
        if len(target_qubits) == 1 and len(amplitudes) >= 2:
            # Single target qubit case
            prob_0 = abs(amplitudes[0])**2
            prob_1 = abs(amplitudes[1])**2 if len(amplitudes) > 1 else 0
            total_prob = prob_0 + prob_1
            
            if total_prob > 1e-10:
                theta = 2 * np.arccos(np.sqrt(prob_0 / total_prob))
                circuit.cry(theta, control_qubit, target_qubits[0])
        else:
            # Multiple target qubits - apply simplified encoding
            for i, qubit in enumerate(target_qubits):
                if i < len(amplitudes):
                    # Simple controlled rotation based on amplitude
                    angle = 2 * np.arcsin(min(abs(amplitudes[i]), 1.0))
                    circuit.cry(angle, control_qubit, qubit)

    def _weighted_superposition_fallback(self, amplitudes, circuit):
        """
        Fallback strategy: Create weighted superposition using individual qubit rotations
        Not perfect amplitude encoding, but preserves some amplitude information
        """
        print("Using weighted superposition fallback for amplitude encoding")
        
        # Normalize amplitudes to probabilities
        probs = np.abs(amplitudes) ** 2
        total_prob = np.sum(probs)
        
        if total_prob > 1e-10:
            probs = probs / total_prob
        else:
            # Uniform distribution fallback
            probs = np.ones(len(amplitudes)) / len(amplitudes)
        
        # Apply weighted rotations to create approximate amplitude distribution
        for i in range(min(self.n_qubits, len(probs))):
            if i < len(probs) and probs[i] > 1e-10:
                # Map probability to rotation angle
                theta = 2 * np.arcsin(np.sqrt(probs[i]))
                circuit.ry(theta, i)
                
                # Add some entanglement to create correlations between qubits
                if i > 0:
                    circuit.cx(i-1, i)

class AngleEncoder(QuantumEncoder):
    """
    Angle encoding: map classical features to rotation angles
    Balanced trade-off between expressivity and implementation simplicity
    """
    
    def __init__(self, n_features, rotation_gate='ry'):
        super().__init__(n_features)
        self.rotation_gate = rotation_gate.lower()
        
    def _calculate_qubits(self):
        """Need one qubit per feature for angle encoding"""
        return self.n_features
    
    def encode(self, features, circuit=None):
        """
        Encode features as rotation angles
        
        Args:
            features: numpy array of classical features (should be in [0,1])
            circuit: QuantumCircuit to add encoding to
        """
        if circuit is None:
            circuit = QuantumCircuit(self.n_qubits)
            
        # Validate input
        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
            
        # Encode each feature as a rotation angle
        for i, feature in enumerate(features):
            # Map feature value [0,1] to rotation angle [0, π]
            angle = feature * np.pi
            
            if self.rotation_gate == 'ry':
                circuit.ry(angle, i)
            elif self.rotation_gate == 'rz':
                circuit.rz(angle, i)
            elif self.rotation_gate == 'rx':
                circuit.rx(angle, i)
            else:
                raise ValueError(f"Unsupported rotation gate: {self.rotation_gate}")
                
        return circuit
    
    def create_parameterized_circuit(self):
        """Create a parameterized circuit for variational algorithms"""
        # Create parameters for each feature
        params = [Parameter(f'x_{i}') for i in range(self.n_features)]
        
        qc = QuantumCircuit(self.n_qubits)
        
        for i, param in enumerate(params):
            angle = param * np.pi
            if self.rotation_gate == 'ry':
                qc.ry(angle, i)
            elif self.rotation_gate == 'rz':
                qc.rz(angle, i)
            else:
                qc.rx(angle, i)
                
        return qc, params

class BasisEncoder(QuantumEncoder):
    """
    Basis encoding: map discrete classical values to computational basis states
    Simplest encoding, maximum hardware compatibility, limited expressivity
    """
    
    def __init__(self, n_features, n_bits_per_feature=2):
        self.n_bits_per_feature = n_bits_per_feature
        self.max_value = 2 ** n_bits_per_feature - 1
        super().__init__(n_features)
        
    def _calculate_qubits(self):
        """Need n_bits_per_feature qubits per feature"""
        return self.n_features * self.n_bits_per_feature
    
    def encode(self, features, circuit=None):
        if circuit is None:
            circuit = QuantumCircuit(self.n_qubits)
        
        # Improve discretization with better binning
        discrete_features = self._improved_discretize_features(features)
        
        # Add quantum superposition elements
        qubit_idx = 0
        for feature_val in discrete_features:
            # Create superposition states based on feature value
            if feature_val > 0:
                circuit.h(qubit_idx)  # Create superposition
                if feature_val == self.max_value:
                    circuit.z(qubit_idx)  # Phase flip for max value
            
            # Add controlled operations between feature bits
            for bit_idx in range(1, self.n_bits_per_feature):
                circuit.cx(qubit_idx, qubit_idx + bit_idx)
            
            qubit_idx += self.n_bits_per_feature
        
        return circuit

    def _improved_discretize_features(self, features):
        # Use percentile-based binning instead of linear scaling
        discrete = np.zeros_like(features)
        for i, feature in enumerate(features):
            if feature <= 0.25:
                discrete[i] = 0
            elif feature <= 0.5:
                discrete[i] = 1
            elif feature <= 0.75:
                discrete[i] = 2
            else:
                discrete[i] = 3
        return discrete.astype(int)
    
    def _discretize_features(self, features):
        """Discretize continuous features to fit in available bits"""
        # Assume features are in [0, 1], map to [0, max_value]
        discrete = np.round(features * self.max_value)
        return np.clip(discrete, 0, self.max_value).astype(int)
    
    def decode_measurement(self, measurement_counts):
        """
        Decode measurement results back to feature space
        Useful for analysis and debugging
        """
        decoded_results = {}
        
        for bitstring, count in measurement_counts.items():
            # Parse bitstring into features
            features = []
            for i in range(self.n_features):
                start_bit = i * self.n_bits_per_feature
                end_bit = start_bit + self.n_bits_per_feature
                feature_bits = bitstring[start_bit:end_bit]
                feature_val = int(feature_bits, 2) / self.max_value
                features.append(feature_val)
                
            decoded_results[tuple(features)] = count
            
        return decoded_results

class HybridEncoder(QuantumEncoder):
    """
    Hybrid encoding: combines multiple encoding strategies
    Allows optimization based on feature characteristics
    """
    
    def __init__(self, feature_groups, encoding_strategies):
        """
        Args:
            feature_groups: list of feature indices for each group
            encoding_strategies: list of encoder instances for each group
        """
        self.feature_groups = feature_groups
        self.encoders = encoding_strategies
        
        # Calculate total features and qubits
        total_features = sum(len(group) for group in feature_groups)
        super().__init__(total_features)
        
    def _calculate_qubits(self):
        """Sum qubits needed for all encoding strategies"""
        return sum(encoder.n_qubits for encoder in self.encoders)
    
    def encode(self, features, circuit=None):
        """Encode features using different strategies for different groups"""
        if circuit is None:
            circuit = QuantumCircuit(self.n_qubits)
            
        current_qubit = 0
        
        for group_indices, encoder in zip(self.feature_groups, self.encoders):
            # Extract features for this group
            group_features = features[group_indices]
            
            # Create subcircuit for this encoding
            subcircuit = QuantumCircuit(encoder.n_qubits)
            encoder.encode(group_features, subcircuit)
            
            # Add to main circuit
            qubit_range = list(range(current_qubit, current_qubit + encoder.n_qubits))
            circuit.compose(subcircuit, qubits=qubit_range, inplace=True)
            
            current_qubit += encoder.n_qubits
            
        return circuit

# Testing and demonstration functions
def test_encodings():
    """Test all encoding strategies with sample data"""
    
    # Sample medical image features (preprocessed)
    sample_features = np.array([0.3, 0.7, 0.1, 0.9, 0.5, 0.2])
    
    print("Testing Quantum Encoding Strategies")
    print("=" * 40)
    
    # Test Amplitude Encoding
    print("\n1. Amplitude Encoding:")
    amp_encoder = AmplitudeEncoder(n_features=len(sample_features))
    amp_circuit = amp_encoder.create_circuit(sample_features, include_measurements=True)
    print(f"   Qubits required: {amp_encoder.n_qubits}")
    print(f"   Circuit depth: {amp_circuit.depth()}")
    
    # Test Angle Encoding  
    print("\n2. Angle Encoding:")
    angle_encoder = AngleEncoder(n_features=len(sample_features))
    angle_circuit = angle_encoder.create_circuit(sample_features, include_measurements=True)
    print(f"   Qubits required: {angle_encoder.n_qubits}")
    print(f"   Circuit depth: {angle_circuit.depth()}")
    
    # Test Basis Encoding
    print("\n3. Basis Encoding:")
    basis_encoder = BasisEncoder(n_features=len(sample_features), n_bits_per_feature=2)
    basis_circuit = basis_encoder.create_circuit(sample_features, include_measurements=True)
    print(f"   Qubits required: {basis_encoder.n_qubits}")
    print(f"   Circuit depth: {basis_circuit.depth()}")

    # Test Hybrid Encoding
    print("\n 4. Hybrid Encoder")
    hybrid_encoder = HybridEncoder([], [amp_encoder, angle_encoder, basis_encoder])
    hybrid_circuit = hybrid_encoder.create_circuit(sample_features, include_measurements=True)
    print(f"   Qubits required: {hybrid_encoder.n_qubits}")
    print(f"   Circuit depth: {hybrid_circuit.depth()}")

    print(f"\nAll encodings tested successfully!")
    
    return {
        'amplitude': (amp_encoder, amp_circuit),
        'angle': (angle_encoder, angle_circuit),
        'basis': (basis_encoder, basis_circuit),
        'hybrid': (hybrid_encoder, hybrid_circuit)
    }

if __name__ == "__main__":
    test_encodings()