"""
Phase 2: Systematic Noise Studies (Refactored)
Brings noise analysis quality closer to Phase 1 by:
 - Training a variational quantum classifier per encoding (Amplitude/Angle/Basis)
 - Reusing trained parameters to assess robustness under parametric, device, coherence, and error-type noise
 - Recording consistent metrics (accuracy, precision, recall, f1, execution_time, avg_circuit_depth, shots)
 - Avoiding wildcard imports and improving reproducibility
 - Providing baseline (noiseless) reference automatically if not present in noise levels
"""

# pylint: disable=trailing-whitespace
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=line-too-long


import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    pauli_error,
    ReadoutError,
)
from qiskit_ibm_runtime.fake_provider import (
    FakePerth,
    FakeLagosV2,
    FakeNairobiV2,
    FakeOslo,
)
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from quantum_encodings.quantum_encoders import AmplitudeEncoder, AngleEncoder, BasisEncoder
from data.medical_data import MedicalImageProcessor
from config.experiment_config import Phase2Config
try:
    # Import the Phase 1 classifier for consistent evaluation
    from experiments.phase1_baseline import QuantumMultiClassifier
except Exception:  # pragma: no cover
    QuantumMultiClassifier = None  # Fallback; code will guard usage

class Phase2NoiseExperiment:
    """
    Manages systematic noise studies for quantum encoding strategies
    """
    
    def __init__(self, results_dir='results/phase2', config: Optional[Phase2Config] = None):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Use provided config or create default
        self.config = config if config else Phase2Config()
        # Initialize simulators and noise models
        self.base_simulator = AerSimulator()
        self.noise_models: Dict[str, NoiseModel] = {}
        self.trained_models: Dict[str, Any] = {}  # encoding -> QuantumMultiClassifier

        # Results storage
        self.results = {
            'parametric_noise': [],
            'device_noise': [],
            'coherence_studies': [],
            'error_comparison': []
        }

        print(f"Phase 2 initialized with {len(self.config.noise_models)} noise models")
        
    def create_noise_models(self) -> Dict[str, NoiseModel]:
        """Create comprehensive set of noise models for testing"""
        print("🔧 Creating noise models...")
        
        noise_models = {}
        
        # 1. Parametric noise models
        for noise_level in self.config.noise_levels:
            for noise_type in self.config.noise_models:
                model_name = f"{noise_type}_{noise_level:.4f}"
                noise_models[model_name] = self._create_parametric_noise_model(
                    noise_type, noise_level
                )
        
        # 2. IBM backend noise models
        ibm_backends = {
            'ibm_perth': FakePerth(),
            'ibm_lagos': FakeLagosV2(),
            'ibm_nairobi': FakeNairobiV2(),
            'ibm_oslo': FakeOslo()
        }
        
        for backend_name in self.config.ibm_backends:
            if backend_name in ibm_backends:
                try:
                    backend = ibm_backends[backend_name]
                    noise_model = NoiseModel.from_backend(backend)
                    noise_models[backend_name] = noise_model
                    print(f"  ✅ Created {backend_name} noise model")
                except Exception as e:
                    print(f"  ❌ Failed to create {backend_name}: {e}")
        
        # 3. Coherence-limited models
        for t1 in self.config.t1_times[:3]:  # Limit for computational feasibility
            for t2 in self.config.t2_times[:3]:
                if t2 <= t1/2:  # Physical constraint: T2 ≤ T1/2
                    model_name = f"coherence_T1{t1}_T2{t2}"
                    noise_models[model_name] = self._create_coherence_model(t1, t2)
        
        self.noise_models = noise_models
        print(f"✅ Created {len(noise_models)} noise models")
        
        return noise_models
    
    def train_baseline_models(self, quantum_data, n_epochs: int = 2, batch_size: int = 25):
        """Train a small variational classifier for each encoding (once, noiseless)."""
        if QuantumMultiClassifier is None:
            print("Phase1 classifier unavailable; falling back to raw encoding evaluation.")
            return
        if self.trained_models:
            return
        print("\n🚧 Training baseline quantum models for noise robustness analysis...")
        for encoding in ['amplitude', 'angle', 'basis']:
            encoder = self._create_encoder(encoding, quantum_data['feature_dim'])
            model = QuantumMultiClassifier(
                encoder,
                n_classes=len(np.unique(quantum_data['train_labels'])),
                n_layers=2,
                learning_rate=0.02,
                maxiter=40,
                entanglement_pattern='linear',
                shots_warmup=256,
                shots_final=1024,
            )
            model.train(quantum_data['train_features'], quantum_data['train_labels'], batch_size=batch_size, n_epochs=n_epochs)
            self.trained_models[encoding] = model
        print("✅ Baseline models trained.")

    def _evaluate_model_under_noise(self, model: Any, quantum_data, noise_model: Optional[NoiseModel], shots: int):
        """Evaluate a trained Phase1 model under a given noise model."""
        start = datetime.now()
        original_sim = model.simulator
        model.simulator = AerSimulator(noise_model=noise_model) if noise_model else self.base_simulator
        try:
            X = quantum_data['test_features']
            y_true = quantum_data['test_labels']
            preds, _probs = model.predict(X, shots=shots)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'accuracy': float(accuracy_score(y_true, preds)),
                'precision': float(precision_score(y_true, preds, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, preds, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, preds, average='weighted', zero_division=0)),
                'execution_time': (datetime.now() - start).total_seconds()
            }
        finally:
            model.simulator = original_sim
        return metrics

    def run_parametric_noise_sweep(self, quantum_data):
        """Systematic sweep of noise parameters using trained models when available."""
        print("\n" + "=" * 60)
        print("PHASE 2A: PARAMETRIC NOISE SWEEP")
        print("=" * 60)
        
        results = []

        # Ensure baseline models
        self.train_baseline_models(quantum_data)
        # Add noiseless baseline if not in noise levels
        include_baseline = 0.0 not in self.config.noise_levels
        baseline_levels = ([0.0] if include_baseline else []) + list(self.config.noise_levels)
        
        for noise_type in self.config.noise_models:
            print(f"\nTesting {noise_type} noise...")
            
            for noise_level in baseline_levels:
                print(f"  Noise level: {noise_level:.4f}")
                
                # Create noise model
                model_name = f"{noise_type}_{noise_level:.4f}"
                if model_name not in self.noise_models:
                    noise_model = None if noise_level == 0 else self._create_parametric_noise_model(noise_type, noise_level)
                else:
                    noise_model = self.noise_models[model_name]
                
                # Test each encoding strategy
                for encoding in ['amplitude', 'angle', 'basis']:
                    print(f"    {encoding} encoding...", end='')
                    
                    # Run multiple trials for statistical robustness
                    encoding_results = []
                    for trial in range(self.config.trials_per_condition):
                        if encoding in self.trained_models:
                            metrics = self._evaluate_model_under_noise(
                                self.trained_models[encoding], quantum_data, noise_model, shots=self.config.shots_per_experiment
                            )
                            exec_time = metrics.pop('execution_time', 0)
                        else:
                            # Fallback raw encoding evaluation
                            encoder_obj = self._create_encoder(encoding, quantum_data['feature_dim'])
                            raw_metrics = self._run_noisy_experiment(
                                encoder_obj, quantum_data, noise_model or NoiseModel(),
                                experiment_id=f"{encoding}_{model_name}_trial{trial}"
                            )
                            metrics = {k: raw_metrics[k] for k in ['accuracy','precision','recall','f1_score']}
                            exec_time = raw_metrics['execution_time']
                        result_record = {
                            'encoding': encoding,
                            'noise_type': noise_type,
                            'noise_level': noise_level,
                            'trial': trial,
                            **metrics,
                            'execution_time': exec_time,
                            'evaluation_type': 'trained_classifier' if encoding in self.trained_models else 'raw_encoding'
                        }
                        encoding_results.append(result_record)
                    
                    # Calculate summary statistics for this condition
                    avg_accuracy = np.mean([r['accuracy'] for r in encoding_results])
                    print(f" Avg accuracy: {avg_accuracy:.3f}")
                    
                    results.extend(encoding_results)
        
        self.results['parametric_noise'] = results
        self._save_intermediate_results('parametric_noise', results)
        
        return results
    
    def run_device_specific_studies(self, quantum_data):
        """
        Test performance on realistic device noise models
        """
        print("\n" + "=" * 60)
        print("PHASE 2B: DEVICE-SPECIFIC NOISE STUDIES")
        print("=" * 60)
        
        results = []
        
        # Test each IBM backend noise model
        for backend_name in self.config.ibm_backends:
            if backend_name not in self.noise_models:
                print(f"  ⚠️  Skipping {backend_name} - noise model not available")
                continue
                
            print(f"\nTesting {backend_name} noise model...")
            noise_model = self.noise_models[backend_name]
            
            for encoding in ['amplitude', 'angle', 'basis']:
                print(f"  {encoding} encoding on {backend_name}...")
                
                for trial in range(self.config.trials_per_condition):
                    if encoding in self.trained_models:
                        metrics = self._evaluate_model_under_noise(self.trained_models[encoding], quantum_data, noise_model, shots=self.config.shots_per_experiment)
                        circuit_depth = None
                        exec_time = 0
                    else:
                        encoder_obj = self._create_encoder(encoding, quantum_data['feature_dim'])
                        raw_metrics = self._run_noisy_experiment(
                            encoder_obj, quantum_data, noise_model,
                            experiment_id=f"{encoding}_{backend_name}_trial{trial}"
                        )
                        metrics = {k: raw_metrics[k] for k in ['accuracy','precision','recall','f1_score']}
                        circuit_depth = raw_metrics['circuit_depth']
                        exec_time = raw_metrics['execution_time']
                    result_record = {
                        'encoding': encoding,
                        'backend': backend_name,
                        'trial': trial,
                        **metrics,
                        'circuit_depth': circuit_depth,
                        'execution_time': exec_time,
                        'evaluation_type': 'trained_classifier' if encoding in self.trained_models else 'raw_encoding'
                    }
                    results.append(result_record)
        
        self.results['device_noise'] = results
        self._save_intermediate_results('device_noise', results)
        
        return results
    
    def run_coherence_time_studies(self, quantum_data):
        """
        Analyze impact of coherence times (T1, T2) on performance
        """
        print("\n" + "=" * 60)
        print("PHASE 2C: COHERENCE TIME STUDIES")
        print("=" * 60)
        
        results = []
        
        for t1 in self.config.t1_times[:3]:  # Limit for feasibility
            for t2 in self.config.t2_times[:3]:
                if t2 > t1/2:  # Skip unphysical combinations
                    continue
                    
                print(f"\nTesting T1={t1}μs, T2={t2}μs...")
                
                # Create coherence-limited noise model
                model_name = f"coherence_T1{t1}_T2{t2}"
                if model_name not in self.noise_models:
                    noise_model = self._create_coherence_model(t1, t2)
                else:
                    noise_model = self.noise_models[model_name]
                
                for encoding in ['amplitude', 'angle', 'basis']:
                    print(f"  {encoding} encoding...")
                    
                    for trial in range(self.config.trials_per_condition):
                        if encoding in self.trained_models:
                            metrics = self._evaluate_model_under_noise(self.trained_models[encoding], quantum_data, noise_model, shots=self.config.shots_per_experiment)
                            exec_time = 0
                        else:
                            encoder_obj = self._create_encoder(encoding, quantum_data['feature_dim'])
                            raw_metrics = self._run_noisy_experiment(
                                encoder_obj, quantum_data, noise_model,
                                experiment_id=f"{encoding}_T1{t1}_T2{t2}_trial{trial}"
                            )
                            metrics = {k: raw_metrics[k] for k in ['accuracy','precision','recall','f1_score']}
                            exec_time = raw_metrics['execution_time']
                        result_record = {
                            'encoding': encoding,
                            't1_time': t1,
                            't2_time': t2,
                            'trial': trial,
                            **metrics,
                            'execution_time': exec_time,
                            'decoherence_impact': None,
                            'evaluation_type': 'trained_classifier' if encoding in self.trained_models else 'raw_encoding'
                        }
                        results.append(result_record)
        
        self.results['coherence_studies'] = results
        self._save_intermediate_results('coherence_studies', results)
        
        return results
    
    def run_error_type_comparison(self, quantum_data):
        """
        Compare impact of different error types at equal error rates
        """
        print("\n" + "=" * 60)
        print("PHASE 2D: ERROR TYPE COMPARISON")
        print("=" * 60)
        
        results = []
        
        # Fix error rate for fair comparison
        fixed_error_rate = 0.01
        
        print(f"Comparing error types at fixed rate: {fixed_error_rate}")
        
        for noise_type in self.config.noise_models:
            print(f"\nTesting {noise_type} errors...")
            
            noise_model = self._create_parametric_noise_model(noise_type, fixed_error_rate)
            
            for encoding in ['amplitude', 'angle', 'basis']:
                print(f"  {encoding} encoding with {noise_type}...")
                
                for trial in range(self.config.trials_per_condition):
                    if encoding in self.trained_models:
                        metrics = self._evaluate_model_under_noise(self.trained_models[encoding], quantum_data, noise_model, shots=self.config.shots_per_experiment)
                        exec_time = 0
                    else:
                        encoder_obj = self._create_encoder(encoding, quantum_data['feature_dim'])
                        raw_metrics = self._run_noisy_experiment(
                            encoder_obj, quantum_data, noise_model,
                            experiment_id=f"{encoding}_{noise_type}_comparison_trial{trial}"
                        )
                        metrics = {k: raw_metrics[k] for k in ['accuracy','precision','recall','f1_score']}
                        exec_time = raw_metrics['execution_time']
                    result_record = {
                        'encoding': encoding,
                        'error_type': noise_type,
                        'error_rate': fixed_error_rate,
                        'trial': trial,
                        **metrics,
                        'execution_time': exec_time,
                        'evaluation_type': 'trained_classifier' if encoding in self.trained_models else 'raw_encoding'
                    }
                    results.append(result_record)
        
        self.results['error_comparison'] = results
        self._save_intermediate_results('error_comparison', results)
        
        return results
    
    def _create_parametric_noise_model(self, noise_type: str, noise_level: float) -> NoiseModel:
        """Create a parametric noise model of specified type and level"""
        noise_model = NoiseModel()
        
        if noise_type == 'depolarizing':
            # Depolarizing error on single and two-qubit gates
            error_1q = depolarizing_error(noise_level, 1)
            error_2q = depolarizing_error(noise_level * 2, 2)  # Higher for 2Q gates
            
            noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'x', 'y', 'z'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
            
        elif noise_type == 'amplitude_damping':
            # Amplitude damping (energy relaxation)
            error_1q = amplitude_damping_error(noise_level)
            noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'x', 'y', 'z'])
            
        elif noise_type == 'phase_damping':
            # Phase damping (dephasing)
            error_1q = phase_damping_error(noise_level)
            noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'x', 'y', 'z'])
            
        elif noise_type == 'pauli':
            # Random Pauli errors
            error_1q = pauli_error([('X', noise_level/3), ('Y', noise_level/3), 
                                  ('Z', noise_level/3), ('I', 1-noise_level)])
            noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'x', 'y', 'z'])
            
        elif noise_type == 'thermal_relaxation':
            # Thermal relaxation with realistic T1, T2
            t1 = 50  # microseconds
            t2 = 20  # microseconds  
            gate_time = 0.1  # microseconds
            
            error_1q = thermal_relaxation_error(t1, t2, gate_time, excited_state_population=noise_level)
            noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'x', 'y', 'z'])
        
        # Add readout errors
        readout_error = noise_level * 0.1  # Typically smaller than gate errors
        ro_error = ReadoutError([[1-readout_error, readout_error], 
                                [readout_error, 1-readout_error]])
        noise_model.add_all_qubit_readout_error(ro_error)
        
        return noise_model
    
    def _create_coherence_model(self, t1: float, t2: float) -> NoiseModel:
        """Create noise model based on coherence times"""
        noise_model = NoiseModel()
        
        # Typical gate times (microseconds)
        single_gate_time = 0.05
        two_gate_time = 0.3
        
        # Thermal relaxation for single-qubit gates
        error_1q = thermal_relaxation_error(t1, t2, single_gate_time)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'x', 'y', 'z'])
        
        # Thermal relaxation for two-qubit gates  
        error_2q = thermal_relaxation_error(t1, t2, two_gate_time).tensor(
                   thermal_relaxation_error(t1, t2, two_gate_time))
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        
        return noise_model
    
    def _create_encoder(self, encoding_type: str, n_features: int):
        """Create encoder instance"""
        if encoding_type == 'amplitude':
            return AmplitudeEncoder(n_features)
        elif encoding_type == 'angle':
            return AngleEncoder(n_features)
        elif encoding_type == 'basis':
            return BasisEncoder(n_features, n_bits_per_feature=2)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _run_noisy_experiment(self, encoder, quantum_data, noise_model, experiment_id: str):
        """Run experiment with specified noise model"""
        start_time = datetime.now()
        
        # Create noisy simulator
        noisy_simulator = AerSimulator(noise_model=noise_model)
        
        # Sample data for experiment
        n_test_samples = min(30, len(quantum_data['test_features']))  # Smaller for noisy sims
        test_indices = np.random.choice(len(quantum_data['test_features']), 
                                       n_test_samples, replace=False)
        
        predictions = []
        
        for idx in test_indices:
            features = quantum_data['test_features'][idx]
            
            # Create quantum circuit
            qc = encoder.create_circuit(features, include_measurements=True)
            
            # Run on noisy simulator
            job = noisy_simulator.run(qc, shots=self.config.shots_per_experiment)
            result = job.result()
            counts = result.get_counts()
            
            # Classify based on measurements
            prediction = self._classify_from_measurements(counts)
            predictions.append(prediction)
        
        # Calculate metrics
        true_labels = quantum_data['test_labels'][test_indices]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels, predictions, average='weighted', zero_division=0),
            'circuit_depth': qc.depth(),
            'execution_time': (datetime.now() - start_time).total_seconds()
        }
        
        return metrics
    
    def _classify_from_measurements(self, counts):
        """Fixed classification based on expectation value"""
        total_shots = sum(counts.values())
        expectation = 0
        
        for bitstring, count in counts.items():
            # Use first qubit as classification bit: 0 -> -1, 1 -> +1
            first_qubit = int(bitstring[0])
            expectation += (2 * first_qubit - 1) * count / total_shots
        
        # Classify based on expectation value sign
        return 1 if expectation > 0 else 0
    
    def _save_intermediate_results(self, experiment_name: str, results: List[Dict]):
        """Save intermediate results"""
        filename = os.path.join(self.results_dir, f"{experiment_name}_results.json")
        
        # Clean results for JSON serialization
        clean_results = []
        for result in results:
            clean_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    clean_result[key] = float(value)
                else:
                    clean_result[key] = value
            clean_results.append(clean_result)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"  Results saved to {filename}")
    
    # Fix the generate_noise_summary method to handle MultiIndex properly

    def generate_noise_summary(self):
        """Generate comprehensive summary of noise studies"""
        print("\n" + "=" * 60)
        print("PHASE 2 NOISE STUDY SUMMARY")
        print("=" * 60)
        
        summary = {}
        
        # Parametric noise summary
        if self.results['parametric_noise']:
            df = pd.DataFrame(self.results['parametric_noise'])
            
            noise_summary = df.groupby(['encoding', 'noise_type']).agg({
                'accuracy': ['mean', 'std', 'min', 'max'],
                'noise_level': ['min', 'max']
            }).round(4)
            
            # Flatten MultiIndex columns
            noise_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                for col in noise_summary.columns.values]
            
            # Convert MultiIndex to string keys for JSON serialization
            noise_dict = {}
            noise_summary_reset = noise_summary.reset_index()
            for _, row in noise_summary_reset.iterrows():
                key = f"{row['encoding']}_{row['noise_type']}"
                row_dict = row.drop(['encoding', 'noise_type']).to_dict()
                noise_dict[key] = row_dict
            
            summary['parametric_noise'] = noise_dict
            print("\nParametric Noise Impact:")
            print(noise_summary.head())
            
            # Find most resilient encoding for each noise type
            resilience_analysis = {}
            for noise_type in df['noise_type'].unique():
                noise_data = df[df['noise_type'] == noise_type]
                encoding_performance = noise_data.groupby('encoding')['accuracy'].mean()
                best_encoding = encoding_performance.idxmax()
                resilience_analysis[noise_type] = {
                    'best_encoding': best_encoding,
                    'performance': float(encoding_performance[best_encoding])  # Convert to Python float
                }
            
            summary['resilience_analysis'] = resilience_analysis
            print(f"\nNoise Resilience Analysis:")
            for noise_type, analysis in resilience_analysis.items():
                print(f"  {noise_type}: {analysis['best_encoding']} "
                    f"(avg accuracy: {analysis['performance']:.3f})")
        
        # Device noise summary
        if self.results['device_noise']:
            df = pd.DataFrame(self.results['device_noise'])
            
            device_summary = df.groupby(['encoding', 'backend']).agg({
                'accuracy': ['mean', 'std'],
                'f1_score': 'mean'
            }).round(4)
            
            # Flatten MultiIndex columns
            device_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                    for col in device_summary.columns.values]
            
            # Convert MultiIndex to string keys
            device_dict = {}
            device_summary_reset = device_summary.reset_index()
            for _, row in device_summary_reset.iterrows():
                key = f"{row['encoding']}_{row['backend']}"
                row_dict = row.drop(['encoding', 'backend']).to_dict()
                device_dict[key] = row_dict
            
            summary['device_noise'] = device_dict
            print(f"\nDevice-Specific Performance:")
            print(device_summary.head())
        
        # Coherence studies summary
        if self.results['coherence_studies']:
            df = pd.DataFrame(self.results['coherence_studies'])
            
            coherence_summary = df.groupby(['encoding', 't1_time', 't2_time']).agg({
                'accuracy': ['mean', 'std'],
                'f1_score': 'mean'
            }).round(4)
            
            # Flatten MultiIndex columns
            coherence_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                    for col in coherence_summary.columns.values]
            
            # Convert MultiIndex to string keys
            coherence_dict = {}
            coherence_summary_reset = coherence_summary.reset_index()
            for _, row in coherence_summary_reset.iterrows():
                key = f"{row['encoding']}_T1{row['t1_time']}_T2{row['t2_time']}"
                row_dict = row.drop(['encoding', 't1_time', 't2_time']).to_dict()
                coherence_dict[key] = row_dict
            
            summary['coherence_studies'] = coherence_dict
            print(f"\nCoherence Time Impact:")
            print(coherence_summary.head())
        
        # Error comparison summary
        if self.results['error_comparison']:
            df = pd.DataFrame(self.results['error_comparison'])
            
            error_summary = df.groupby(['encoding', 'error_type']).agg({
                'accuracy': ['mean', 'std'],
                'f1_score': 'mean'
            }).round(4)
            
            # Flatten MultiIndex columns
            error_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                for col in error_summary.columns.values]
            
            # Convert MultiIndex to string keys
            error_dict = {}
            error_summary_reset = error_summary.reset_index()
            for _, row in error_summary_reset.iterrows():
                key = f"{row['encoding']}_{row['error_type']}"
                row_dict = row.drop(['encoding', 'error_type']).to_dict()
                error_dict[key] = row_dict
            
            summary['error_comparison'] = error_dict
            print(f"\nError Type Comparison:")
            print(error_summary.head())
        
        # Save summary
        def make_json_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        clean_summary = make_json_serializable(summary)
        summary_file = os.path.join(self.results_dir, 'phase2_noise_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(clean_summary, f, indent=2)
        
        print(f"\nNoise study summary saved to {summary_file}")
        return clean_summary

    
    def run_complete_phase2(self, quantum_data):
        """Run all Phase 2 noise studies"""
        print("🔊 STARTING PHASE 2: SYSTEMATIC NOISE STUDIES")
        print(f"📊 Planned noise conditions: {len(self.config.noise_levels) * len(self.config.noise_models)}")
        print(f"🖥️  IBM backend models: {len(self.config.ibm_backends)}")
        print(f"⏱️  Estimated duration: 4-5 days")
        print(f"💰 Cost: FREE (local simulation)")
        
        start_time = datetime.now()
        
        try:
            # Initialize noise models
            self.create_noise_models()
            
            # Phase 2A: Parametric noise sweep
            self.run_parametric_noise_sweep(quantum_data)
            
            # Phase 2B: Device-specific studies
            self.run_device_specific_studies(quantum_data)
            
            # Phase 2C: Coherence time studies  
            self.run_coherence_time_studies(quantum_data)
            
            # Phase 2D: Error type comparison
            self.run_error_type_comparison(quantum_data)
            
            # Generate summary
            summary = self.generate_noise_summary()
            
            total_time = datetime.now() - start_time
            
            print("\n" + "🎉" * 20)
            print("PHASE 2 COMPLETED SUCCESSFULLY!")
            print(f"Total execution time: {total_time}")
            print(f"Results saved in: {self.results_dir}")
            print("Ready for Phase 3: Advanced Analysis")
            print("🎉" * 20)
            
            return summary
            
        except Exception as e:
            print(f"\n❌ Phase 2 failed with error: {e}")
            print("Saving partial results...")
            self.generate_noise_summary()
            raise

# Example usage
if __name__ == "__main__":
    # Initialize Phase 2 experiment
    phase2 = Phase2NoiseExperiment()
    
    print("Phase 2 Noise Studies Framework Ready!")
    print("\nTo run experiments:")
    print("1. Ensure quantum_encoders.py and medical_data.py are importable")
    print("2. Load quantum_data from Phase 1")  
    print("3. Call phase2.run_complete_phase2(quantum_data)")
    
    print(f"\nNoise models to be tested: {len(phase2.config.noise_models)}")
    print(f"IBM backends: {len(phase2.config.ibm_backends)}")
    print(f"Coherence time combinations: {len(phase2.config.t1_times) * len(phase2.config.t2_times)}")
    
    # Show what noise models will be created
    print("\nNoise model categories:")
    print(f"  • Parametric models: {len(phase2.config.noise_models) * len(phase2.config.noise_levels)}")
    print(f"  • IBM backend models: {len(phase2.config.ibm_backends)}")
    print(f"  • Coherence-limited models: ~{len(phase2.config.t1_times) * 2}")  # Approximate