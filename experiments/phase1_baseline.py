"""
Phase 1: Quantum ML Baseline - Comprehensive Training & Evaluation
Optimized for laptop execution (6-10 hours) with research breadth
"""


# pylint: disable=trailing-whitespace
# pylint: disable=wrong-import-position
# pylint: disable=line-too-long
# pylint: disable=invalid-name


import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA, COBYLA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quantum_encodings.quantum_encoders import AmplitudeEncoder, AngleEncoder, BasisEncoder
from data.medical_data import MedicalImageProcessor

np.random.seed(42)
class QuantumMultiClassifier:
    """Variational Quantum Classifier with proper training"""
    
    def __init__(self, encoder, n_classes=2, n_layers=2, optimizer='SPSA',
                 learning_rate=0.05, maxiter=100, shots_final=1024):
        self.encoder = encoder
        self.n_qubits = encoder.n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.shots_final = shots_final
        
        if n_classes == 2:
            self.n_readout_qubits = 1
        else:
            self.n_readout_qubits = max(1, int(np.ceil(np.log2(n_classes))))
        
        if optimizer == 'SPSA':
            # SPSA requires both learning_rate and perturbation if one is provided.
            pert = learning_rate * 0.5 if learning_rate is not None else None
            if learning_rate is not None:
                self.optimizer = SPSA(maxiter=maxiter, learning_rate=learning_rate, perturbation=pert)
            else:
                self.optimizer = SPSA(maxiter=maxiter)
        else:
            self.optimizer = COBYLA(maxiter=maxiter)
        
        self.n_params = self.n_qubits * self.n_layers * 2
        # Initialize parameters as 1D numpy array
        self.params = np.random.normal(0, 0.1, self.n_params).astype(np.float64)
        self.simulator = AerSimulator()
        self.is_trained = False
        self.classical_head = None

    def _ensure_param_array(self):
        """Guarantee self.params is a 1D float ndarray of length n_params."""
        if not isinstance(self.params, np.ndarray):
            self.params = np.asarray(self.params, dtype=np.float64).flatten()
        
        if self.params.ndim != 1:
            self.params = self.params.flatten()
            
        if len(self.params) != self.n_params:
            if len(self.params) < self.n_params:
                # Pad with zeros
                pad_size = self.n_params - len(self.params)
                self.params = np.concatenate([self.params, np.zeros(pad_size, dtype=np.float64)])
            else:
                # Truncate
                self.params = self.params[:self.n_params]
        
        self.params = self.params.astype(np.float64)

    def _sanitize_params(self, candidate):
        """Convert optimizer-supplied param candidate to correct ndarray shape."""
        if candidate is None:
            return np.random.normal(0, 0.1, self.n_params).astype(np.float64)
            
        # Convert to numpy array and flatten
        arr = np.asarray(candidate, dtype=np.float64).flatten()
        
        # Handle size mismatch
        if len(arr) < self.n_params:
            # Pad with random values
            pad_size = self.n_params - len(arr)
            pad_values = np.random.normal(0, 0.01, pad_size)
            arr = np.concatenate([arr, pad_values])
        elif len(arr) > self.n_params:
            arr = arr[:self.n_params]
            
        return arr.astype(np.float64)
        
    def create_variational_circuit(self, features, parameters):
        # Ensure parameters are properly formatted
        parameters = self._sanitize_params(parameters)
        
        # Ensure features is a numpy array
        features = np.asarray(features, dtype=np.float64)
        
        qc = self.encoder.create_circuit(features, include_measurements=False)

        param_idx = 0
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                if param_idx < len(parameters):
                    qc.ry(float(parameters[param_idx]), qubit)
                    param_idx += 1
                if param_idx < len(parameters):
                    qc.rz(float(parameters[param_idx]), qubit)
                    param_idx += 1

            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)

        return qc
    
    def measure_expectation_values(self, features, parameters, shots=1024):
        try:
            qc = self.create_variational_circuit(features, parameters)
            
            classical_reg = ClassicalRegister(self.n_readout_qubits)
            qc.add_register(classical_reg)
            
            for i in range(self.n_readout_qubits):
                qc.measure(i, i)
            
            job = self.simulator.run(qc, shots=shots, seed_simulator=42)
            result = job.result()
            counts = result.get_counts()
            
            expectations = []
            for qubit_idx in range(self.n_readout_qubits):
                p0 = sum(count for bitstring, count in counts.items() 
                        if len(bitstring) > qubit_idx and bitstring[-(qubit_idx+1)] == '0')
                p1 = sum(counts.values()) - p0
                total_shots = sum(counts.values())
                if total_shots > 0:
                    expectation = (p0 - p1) / total_shots
                else:
                    expectation = 0.0
                expectations.append(expectation)
            
            return np.array(expectations, dtype=np.float64)
            
        except Exception as e:
            # Return random expectations if circuit fails
            print(f"Warning: Circuit execution failed: {e}")
            return np.random.uniform(-1, 1, self.n_readout_qubits).astype(np.float64)
    
    def cost_function(self, parameters, X, y, shots):
        # Ensure inputs are numpy arrays
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        parameters = self._sanitize_params(parameters)
        
        total_loss = 0
        n_samples = len(X)
        
        for i in range(n_samples):
            try:
                features = X[i]
                label = int(y[i])
                
                expectations = self.measure_expectation_values(features, parameters, shots=shots)
                
                # Binary classification
                if self.n_classes == 2:
                    prob_1 = (1 - expectations[0]) / 2
                    prob_1 = np.clip(prob_1, 1e-8, 1 - 1e-8)
                    class_probs = np.array([1 - prob_1, prob_1])
                else:
                    # Multi-class via independent qubits
                    qubit_probs = (1 - expectations) / 2
                    qubit_probs = np.clip(qubit_probs, 1e-8, 1 - 1e-8)
                    
                    class_probs = np.zeros(self.n_classes)
                    for class_idx in range(min(self.n_classes, 2 ** self.n_readout_qubits)):
                        binary_repr = format(class_idx, f'0{self.n_readout_qubits}b')
                        prob = 1.0
                        for qubit_idx, bit in enumerate(binary_repr):
                            if qubit_idx < len(qubit_probs):
                                prob *= qubit_probs[qubit_idx] if bit == '1' else (1 - qubit_probs[qubit_idx])
                        class_probs[class_idx] = prob
                    
                    # Normalize
                    prob_sum = np.sum(class_probs)
                    if prob_sum > 0:
                        class_probs /= prob_sum
                    else:
                        class_probs = np.ones(self.n_classes) / self.n_classes
                
                # Cross-entropy loss
                epsilon = 1e-8
                class_probs = np.clip(class_probs, epsilon, 1 - epsilon)
                
                if 0 <= label < self.n_classes:
                    loss = -np.log(class_probs[label])
                else:
                    # Invalid label, assign uniform loss
                    loss = -np.log(1.0 / self.n_classes)
                    
                total_loss += loss
                
            except Exception as e:
                print(f"Warning: Error in cost function for sample {i}: {e}")
                # Add penalty for failed samples
                total_loss += 10.0
        
        return total_loss / max(1, n_samples)
    
    def train(self, X_train, y_train, batch_size=20, n_epochs=4, show_progress=True):
        # Convert inputs to numpy arrays
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train, dtype=int)
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        n_batches = max(1, len(X_train_split) // batch_size)
        self._ensure_param_array()
        best_params = self.params.copy()
        best_val_loss = float('inf')

        if show_progress:
            print(
                f"Training quantum classifier...\n"
                f"Dataset: {len(X_train)} samples, {self.n_classes} classes\n"
                f"Training epochs: {n_epochs}, batch size: {batch_size}\n"
                f"Training samples: {len(X_train_split)}, Validation: {len(X_val)}\n"
                f"Batches per epoch: {n_batches}"
            )

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train_split))
            X_shuffled = X_train_split[indices]
            y_shuffled = y_train_split[indices]

            epoch_losses = []
            for batch_iter in range(n_batches):
                start_idx = batch_iter * batch_size
                end_idx = min(start_idx + batch_size, len(X_train_split))
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                shots = 256 + int(256 * epoch / max(1, n_epochs - 1))

                def objective(params):
                    try:
                        clean_params = self._sanitize_params(params)
                        return self.cost_function(clean_params, X_batch, y_batch, shots)
                    except Exception as obj_err:
                        print(f"  Warning: Objective function failed: {obj_err}")
                        return 1e6

                try:
                    self._ensure_param_array()
                    result = self.optimizer.minimize(fun=objective, x0=self.params)
                    
                    if hasattr(result, 'x') and result.x is not None:
                        new_params = self._sanitize_params(result.x)
                        self.params = new_params
                    
                except Exception as opt_err:
                    print(f"  Warning: Optimizer failed, skipping update: {opt_err}")

                # Compute batch loss for progress
                if show_progress:
                    try:
                        batch_loss = self.cost_function(self.params, X_batch, y_batch, shots)
                        epoch_losses.append(batch_loss)
                        
                        # Progress bar
                        prog = (batch_iter + 1) / n_batches
                        bar_len = 20
                        filled = int(bar_len * prog)
                        bar = '█' * filled + '-' * (bar_len - filled)
                        if (batch_iter % 2 == 0) or batch_iter == n_batches - 1:
                            print(f"Epoch {epoch+1}/{n_epochs}  [{bar}]  Batch {batch_iter+1}/{n_batches} loss {batch_loss:.4f} shots={shots}")
                    except Exception:
                        pass

            # Validation
            try:
                val_loss = self.cost_function(self.params, X_val, y_val, shots=256)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._ensure_param_array()
                    best_params = self.params.copy()
                    if show_progress:
                        print(f"  ✓ New best val loss {best_val_loss:.4f}")
            except Exception as e:
                print(f"  Warning: Validation failed: {e}")
                val_loss = float('inf')

            if show_progress:
                mean_train = np.mean(epoch_losses) if epoch_losses else float('nan')
                print(f"  Epoch summary: train {mean_train:.4f} val {val_loss:.4f}")

        self.params = best_params
        self.is_trained = True

        # Train classical head
        if self.n_classes > 1:
            try:
                exp_rows = []
                for features in X_train_split:
                    exp = self.measure_expectation_values(features, self.params, shots=self.shots_final)
                    exp_rows.append(exp)
                exp_rows = np.array(exp_rows)
                
                self.classical_head = LogisticRegression(max_iter=500, random_state=42)
                self.classical_head.fit(exp_rows, y_train_split)
            except Exception as e:
                print(f"Warning: Classical head training failed: {e}")
                self.classical_head = None

        if show_progress:
            print(f"Training completed. Best val loss {best_val_loss:.4f}")

        return best_val_loss
    
    def predict(self, X, shots=1024):
        X = np.asarray(X)
        expectation_rows = []
        
        for features in X:
            exp = self.measure_expectation_values(features, self.params, shots)
            expectation_rows.append(exp)
        expectation_rows = np.array(expectation_rows)
        
        if self.classical_head is not None:
            try:
                probs = self.classical_head.predict_proba(expectation_rows)
                preds = np.argmax(probs, axis=1)
            except Exception:
                # Fallback to simple prediction
                preds = np.zeros(len(X), dtype=int)
                probs = np.ones((len(X), self.n_classes)) / self.n_classes
        else:
            # Binary classification fallback
            preds = []
            probs = []
            for exp in expectation_rows:
                prob_1 = (1 - exp[0]) / 2
                prob_1 = np.clip(prob_1, 0, 1)
                class_probs = np.array([1 - prob_1, prob_1])
                preds.append(1 if prob_1 > 0.5 else 0)
                probs.append(class_probs)
            preds = np.array(preds)
            probs = np.array(probs)
        
        return preds, probs
    
class Phase1BaselineExperiment:
    """Phase 1 quantum ML experiment with comprehensive analysis"""
    
    def __init__(self, results_dir='results/phase1_baseline'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Experiment parameters optimized for laptop execution
        self.encoding_strategies = ['amplitude', 'angle', 'basis']
        self.feature_dimensions = [2, 4, 6, 8, 10]
        self.circuit_depths = [2, 3, 4, 8, 10]
        self.trials_per_condition = 4
        
        self.results = {
            'encoding_comparison': [],
            'dimension_scaling': [],
            'depth_analysis': [],
            'cross_validation': [],
            'classical_baselines': {}
        }
        
    def prepare_dataset(self, datasets, n_classes=2, samples_per_class=300):
        processor = MedicalImageProcessor(target_dim=6)
        full_data = processor.prepare_for_quantum(datasets)
        
        X_full = full_data['train_features']
        y_full = full_data['train_labels']
        
        # Ensure X_full and y_full are numpy arrays
        X_full = np.asarray(X_full)
        y_full = np.asarray(y_full)
        
        if n_classes == 2:
            # Get indices and ensure they're numpy arrays of integers
            class_0_mask = (y_full == 0)
            other_mask = (y_full != 0)
            
            class_0_indices = np.where(class_0_mask)[0][:samples_per_class]
            other_indices = np.where(other_mask)[0][:samples_per_class]
            
            # Create balanced dataset using proper indexing
            balanced_X = np.vstack([
                X_full[class_0_indices],
                X_full[other_indices]
            ])
            
            balanced_y = np.hstack([
                np.zeros(len(class_0_indices), dtype=int),
                np.ones(len(other_indices), dtype=int)
            ])
            
        else:
            unique_classes, class_counts = np.unique(y_full, return_counts=True)
            top_classes = unique_classes[np.argsort(class_counts)[-n_classes:]]
            
            balanced_X_list = []
            balanced_y_list = []
            
            for new_label, original_class in enumerate(top_classes):
                class_mask = (y_full == original_class)
                class_indices = np.where(class_mask)[0][:samples_per_class]
                
                balanced_X_list.append(X_full[class_indices])
                balanced_y_list.append(np.full(len(class_indices), new_label, dtype=int))
            
            balanced_X = np.vstack(balanced_X_list)
            balanced_y = np.hstack(balanced_y_list)
        
        # Normalize to [0,1]
        feat_min = balanced_X.min()
        feat_max = balanced_X.max()
        if feat_min < 0 or feat_max > 1:
            balanced_X = (balanced_X - feat_min) / (feat_max - feat_min)
        
        X_train, X_test, y_train, y_test = train_test_split(
            balanced_X, balanced_y, test_size=0.3, stratify=balanced_y, random_state=42
        )
        
        return {
            'train_features': X_train,
            'train_labels': y_train,
            'test_features': X_test,
            'test_labels': y_test,
            'n_classes': n_classes,
            'feature_dim': X_train.shape[1]
        }
    
    def establish_classical_baselines(self, quantum_data):
        X_train = quantum_data['train_features']
        y_train = quantum_data['train_labels']
        X_test = quantum_data['test_features']
        y_test = quantum_data['test_labels']
        
        baselines = {}
        
        # Majority class baseline
        majority_class = np.bincount(y_train).argmax()
        majority_pred = np.full(len(y_test), majority_class)
        baselines['majority_class'] = accuracy_score(y_test, majority_pred)
        
        # Classical ML models
        classifiers = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        }
        
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            baselines[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        
        self.results['classical_baselines'] = baselines
        return baselines
    
    def run_encoding_comparison(self, quantum_data):
        print("Running encoding strategy comparison...")
        results = []
        best_per_encoding = {e: {'accuracy': -1, 'trial': None} for e in self.encoding_strategies}
        
        for encoding in self.encoding_strategies:
            print(f"Testing {encoding} encoding")
            
            for trial in range(self.trials_per_condition):
                trial_seed = int(np.random.randint(0, 1_000_000))
                encoder = self._create_encoder(encoding, quantum_data['feature_dim'])
                qml_model = QuantumMultiClassifier(
                    encoder, n_classes=quantum_data['n_classes'], 
                    n_layers=2, maxiter=80, shots_final=512
                )
                
                start_time = datetime.now()
                qml_model.train(quantum_data['train_features'], quantum_data['train_labels'])
                training_time = (datetime.now() - start_time).total_seconds()
                
                y_pred, _ = qml_model.predict(quantum_data['test_features'], shots=512)
                y_true = quantum_data['test_labels']
                
                metrics = self._calculate_metrics(y_true, y_pred)
                param_count = qml_model.n_params
                # Probe circuit depth/gates (parameters don't affect topology)
                try:
                    probe_circuit = qml_model.create_variational_circuit(quantum_data['train_features'][0], qml_model.params)
                    circuit_depth = probe_circuit.depth()
                    gate_count = len(probe_circuit.data)
                except Exception:
                    circuit_depth = None
                    gate_count = None
                accuracy_per_qubit = metrics['accuracy'] / encoder.n_qubits if encoder.n_qubits else 0.0
                f1_per_param = metrics['f1_score'] / param_count if param_count else 0.0
                
                record = {
                    'encoding': encoding,
                    'trial': trial,
                    'feature_dim': quantum_data['feature_dim'],
                    'n_qubits': encoder.n_qubits,
                    'training_time': training_time,
                    'param_count': param_count,
                    'circuit_depth_example': circuit_depth,
                    'gate_count_example': gate_count,
                    'accuracy_per_qubit': accuracy_per_qubit,
                    'f1_per_param': f1_per_param,
                    'seed': trial_seed,
                    **metrics
                }
                results.append(record)

                # Persist model artifacts
                try:
                    model_dir = os.path.join(self.results_dir, 'models', encoding, f'trial_{trial}')
                    os.makedirs(model_dir, exist_ok=True)
                    # Save params
                    if not qml_model:
                        print("Could not get model params")
                    np.save(os.path.join(model_dir, 'params.npy'), qml_model.params)
                    # Save classical head
                    if qml_model.classical_head is not None:
                        with open(os.path.join(model_dir, 'classical_head.pkl'), 'wb') as pf:
                            pickle.dump(qml_model.classical_head, pf)
                    # Save metadata
                    metadata = {
                        'encoding': encoding,
                        'trial': trial,
                        'seed': trial_seed,
                        'n_qubits': encoder.n_qubits,
                        'n_layers': 2,
                        'param_count': param_count,
                        'feature_dim': quantum_data['feature_dim'],
                        'accuracy': metrics['accuracy'],
                        'f1_score': metrics['f1_score'],
                        'circuit_depth_example': circuit_depth,
                        'gate_count_example': gate_count
                    }
                    with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as mf:
                        json.dump(metadata, mf, indent=2)
                except Exception as e:  # pragma: no cover
                    print(f"  Warning: failed to persist model artifacts ({e})")

                # Track best
                if metrics['accuracy'] > best_per_encoding[encoding]['accuracy']:
                    best_per_encoding[encoding] = {'accuracy': metrics['accuracy'], 'trial': trial}
        
        self.results['encoding_comparison'] = results
        # Mark best models in metadata (append small summary file)
        try:
            with open(os.path.join(self.results_dir, 'best_models.json'), 'w', encoding='utf-8') as bf:
                json.dump(best_per_encoding, bf, indent=2)
        except Exception:  # pragma: no cover
            pass
        return results
    
    def run_dimension_scaling(self, datasets):
        print("Running dimension scaling analysis...")
        results = []
        
        for dim in self.feature_dimensions:
            processor = MedicalImageProcessor(target_dim=dim)
            quantum_data = processor.prepare_for_quantum(datasets)
            
            # Balance the dataset for this dimension
            X_full = quantum_data['train_features']
            y_full = quantum_data['train_labels']
            
            class_0_indices = np.where(y_full == 0)[0][:100]
            other_indices = np.where(y_full != 0)[0][:100]
            
            X_balanced = np.vstack([X_full[class_0_indices], X_full[other_indices]])
            y_balanced = np.hstack([np.zeros(len(class_0_indices)), np.ones(len(other_indices))])
            
            # Normalize
            X_min, X_max = X_balanced.min(), X_balanced.max()
            if X_min < 0 or X_max > 1:
                X_balanced = (X_balanced - X_min) / (X_max - X_min)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.3, random_state=42
            )
            
            dim_data = {
                'train_features': X_train, 'train_labels': y_train,
                'test_features': X_test, 'test_labels': y_test,
                'n_classes': 2, 'feature_dim': dim
            }
            
            for encoding in self.encoding_strategies:
                for trial in range(self.trials_per_condition):
                    encoder = self._create_encoder(encoding, dim)
                    qml_model = QuantumMultiClassifier(
                        encoder, n_classes=2, n_layers=2, maxiter=30, shots_final=256
                    )
                    
                    qml_model.train(dim_data['train_features'], dim_data['train_labels'])
                    y_pred, _ = qml_model.predict(dim_data['test_features'], shots=256)
                    
                    metrics = self._calculate_metrics(dim_data['test_labels'], y_pred)
                    
                    results.append({
                        'encoding': encoding,
                        'feature_dim': dim,
                        'trial': trial,
                        'n_qubits': encoder.n_qubits,
                        **metrics
                    })
        
        self.results['dimension_scaling'] = results
        return results
    
    def run_depth_analysis(self, quantum_data):
        print("Running circuit depth analysis...")
        results = []
        
        for depth in self.circuit_depths:
            for encoding in self.encoding_strategies:
                for trial in range(self.trials_per_condition):
                    encoder = self._create_encoder(encoding, quantum_data['feature_dim'])
                    qml_model = QuantumMultiClassifier(
                        encoder, n_classes=quantum_data['n_classes'],
                        n_layers=depth, maxiter=30, shots_final=256
                    )
                    
                    qml_model.train(quantum_data['train_features'], quantum_data['train_labels'])
                    y_pred, _ = qml_model.predict(quantum_data['test_features'], shots=256)
                    
                    metrics = self._calculate_metrics(quantum_data['test_labels'], y_pred)
                    
                    results.append({
                        'encoding': encoding,
                        'circuit_depth': depth,
                        'trial': trial,
                        **metrics
                    })
        
        self.results['depth_analysis'] = results
        return results
    
    def run_cross_validation(self, quantum_data, n_splits=3, max_folds=None):
        print("Running cross-validation study...")
        results = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(skf.split(quantum_data['train_features'], quantum_data['train_labels']))
        if max_folds is not None:
            cv_splits = cv_splits[:max_folds]

        for encoding in self.encoding_strategies:
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                X_train = quantum_data['train_features'][train_idx]
                y_train = quantum_data['train_labels'][train_idx]
                X_val = quantum_data['train_features'][val_idx]
                y_val = quantum_data['train_labels'][val_idx]

                encoder = self._create_encoder(encoding, quantum_data['feature_dim'])
                qml_model = QuantumMultiClassifier(
                    encoder, n_classes=quantum_data['n_classes'],
                    n_layers=2, maxiter=20, shots_final=256
                )

                qml_model.train(X_train, y_train, n_epochs=2, show_progress=False)
                y_pred, _ = qml_model.predict(X_val, shots=256)

                metrics = self._calculate_metrics(y_val, y_pred)

                results.append({
                    'encoding': encoding,
                    'fold': fold,
                    **metrics
                })

        self.results['cross_validation'] = results
        return results
    
    def _create_encoder(self, encoding_type, n_features):
        if encoding_type == 'amplitude':
            return AmplitudeEncoder(n_features)
        elif encoding_type == 'angle':
            return AngleEncoder(n_features)
        elif encoding_type == 'basis':
            return BasisEncoder(n_features, n_bits_per_feature=1)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _calculate_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def save_results(self, filename, results):
        filepath = os.path.join(self.results_dir, f"{filename}.json")
        clean_results = []
        for result in results:
            clean_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    clean_result[key] = float(value)
                else:
                    clean_result[key] = value
            clean_results.append(clean_result)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
    
    def generate_analysis_report(self):
        print("Generating comprehensive analysis...")
        
        # Statistical summaries
        analysis = {
            'classical_baselines': self.results['classical_baselines'],
            'quantum_performance': {},
            'comparative_analysis': {},
            'pairwise_tests': {}
        }
        
        # Encoding comparison analysis
        if self.results['encoding_comparison']:
            df = pd.DataFrame(self.results['encoding_comparison'])
            # Aggregate per encoding
            for encoding in df['encoding'].unique():
                enc_data = df[df['encoding'] == encoding]
                n = len(enc_data)
                acc_std = enc_data['accuracy'].std(ddof=1) if n > 1 else 0.0
                f1_std = enc_data['f1_score'].std(ddof=1) if n > 1 else 0.0
                acc_ci95 = 1.96 * acc_std / np.sqrt(n) if n > 1 else 0.0
                f1_ci95 = 1.96 * f1_std / np.sqrt(n) if n > 1 else 0.0
                mean_params = enc_data['param_count'].mean() if 'param_count' in enc_data else np.nan
                analysis['quantum_performance'][encoding] = {
                    'accuracy_mean': float(enc_data['accuracy'].mean()),
                    'accuracy_std': float(acc_std),
                    'accuracy_ci95': float(acc_ci95),
                    'f1_mean': float(enc_data['f1_score'].mean()),
                    'f1_std': float(f1_std),
                    'f1_ci95': float(f1_ci95),
                    'n_qubits_mean': float(enc_data['n_qubits'].mean()),
                    'param_count_mean': float(mean_params) if not np.isnan(mean_params) else None,
                    'accuracy_per_qubit_mean': float(enc_data.get('accuracy_per_qubit', pd.Series(dtype=float)).mean()) if 'accuracy_per_qubit' in enc_data else None,
                    'f1_per_param_mean': float(enc_data.get('f1_per_param', pd.Series(dtype=float)).mean()) if 'f1_per_param' in enc_data else None,
                    'trials': n
                }

            # Pairwise Welch t-tests & Cohen's d
            try:
                from scipy import stats as scipy_stats  # type: ignore
                encodings = sorted(df['encoding'].unique())
                for i in range(len(encodings)):
                    for j in range(i + 1, len(encodings)):
                        e1, e2 = encodings[i], encodings[j]
                        a1 = df[df.encoding == e1]['accuracy']
                        a2 = df[df.encoding == e2]['accuracy']
                        # Welch t-test
                        t_res = scipy_stats.ttest_ind(a1, a2, equal_var=False)
                        # Prefer attribute access; fall back to sequence only if needed
                        t_stat_attr = getattr(t_res, 'statistic', None)
                        p_val_attr = getattr(t_res, 'pvalue', None)
                        if t_stat_attr is not None and p_val_attr is not None:
                            t_stat = float(t_stat_attr)
                            p_val = float(p_val_attr)
                        else:  # fallback for older scipy returning tuple
                            try:
                                t_stat = float(t_res[0])  # type: ignore[index]
                                p_val = float(t_res[1])   # type: ignore[index]
                            except Exception:  # pragma: no cover
                                t_stat = float('nan')
                                p_val = float('nan')
                        # Cohen's d
                        s1, s2 = a1.std(ddof=1), a2.std(ddof=1)
                        n1, n2 = len(a1), len(a2)
                        pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / max(1, (n1 + n2 - 2))) if (n1 + n2) > 2 else 0.0
                        cohend = float((a1.mean() - a2.mean()) / pooled) if pooled > 0 else 0.0
                        analysis['pairwise_tests'][f'{e1}_vs_{e2}'] = {
                            't_stat': t_stat,
                            'p_value': p_val,
                            'cohens_d': cohend
                        }
            except Exception:
                analysis['pairwise_tests']['warning'] = 'SciPy not available'
                analysis['quantum_performance'][encoding] = {
                    'accuracy_mean': float(enc_data['accuracy'].mean()),
                    'accuracy_std': float(enc_data['accuracy'].std()),
                    'f1_mean': float(enc_data['f1_score'].mean()),
                    'f1_std': float(enc_data['f1_score'].std()),
                    'n_qubits': int(enc_data['n_qubits'].iloc[0])
                }
        
        # Dimension scaling analysis
        if self.results['dimension_scaling']:
            df = pd.DataFrame(self.results['dimension_scaling'])
            scaling_data = {}
            for dim in df['feature_dim'].unique():
                dim_data = df[df['feature_dim'] == dim]
                dim_summary = {}
                for encoding in dim_data['encoding'].unique():
                    enc_dim_data = dim_data[dim_data['encoding'] == encoding]
                    dim_summary[encoding] = {
                        'accuracy_mean': float(enc_dim_data['accuracy'].mean()),
                        'f1_mean': float(enc_dim_data['f1_score'].mean())
                    }
                scaling_data[f'dim_{dim}'] = dim_summary
            analysis['dimension_scaling'] = scaling_data
        
        # Depth analysis
        if self.results['depth_analysis']:
            df = pd.DataFrame(self.results['depth_analysis'])
            depth_data = {}
            for depth in df['circuit_depth'].unique():
                depth_subset = df[df['circuit_depth'] == depth]
                depth_summary = {}
                for encoding in depth_subset['encoding'].unique():
                    enc_depth_data = depth_subset[depth_subset['encoding'] == encoding]
                    depth_summary[encoding] = {
                        'accuracy_mean': float(enc_depth_data['accuracy'].mean()),
                        'f1_mean': float(enc_depth_data['f1_score'].mean())
                    }
                depth_data[f'depth_{depth}'] = depth_summary
            analysis['depth_analysis'] = depth_data
        
        # Cross-validation analysis
        if self.results['cross_validation']:
            df = pd.DataFrame(self.results['cross_validation'])
            cv_data = {}
            for encoding in df['encoding'].unique():
                enc_data = df[df['encoding'] == encoding]
                cv_data[encoding] = {
                    'accuracy_mean': float(enc_data['accuracy'].mean()),
                    'accuracy_std': float(enc_data['accuracy'].std()),
                    'f1_mean': float(enc_data['f1_score'].mean()),
                    'f1_std': float(enc_data['f1_score'].std())
                }
            analysis['cross_validation'] = cv_data
        
        # Performance ranking
        if self.results['encoding_comparison']:
            df = pd.DataFrame(self.results['encoding_comparison'])
            ranking_base = df.groupby('encoding').agg({
                'accuracy': 'mean',
                'f1_score': 'mean',
                'param_count': 'mean'
            }).rename(columns={'param_count': 'param_mean'})
            # Parameter penalty term (scaled 0..1)
            if ranking_base['param_mean'].max() > ranking_base['param_mean'].min():
                param_norm = (ranking_base['param_mean'] - ranking_base['param_mean'].min()) / (ranking_base['param_mean'].max() - ranking_base['param_mean'].min())
            else:
                param_norm = ranking_base['param_mean'] * 0
            ranking_base['performance_score'] = 0.45 * ranking_base['accuracy'] + 0.45 * ranking_base['f1_score'] + 0.10 * (1 - param_norm)
            ranking_base = ranking_base.sort_values('performance_score', ascending=False)
            analysis['performance_ranking'] = {
                enc: {
                    'accuracy': float(row['accuracy']),
                    'f1_score': float(row['f1_score']),
                    'param_mean': float(row['param_mean']),
                    'performance_score': float(row['performance_score']),
                    'rank': idx + 1
                } for idx, (enc, row) in enumerate(ranking_base.iterrows())
            }
        
        # Save analysis
        analysis_file = os.path.join(self.results_dir, 'comprehensive_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save individual result files
        for experiment_name, results in self.results.items():
            if results and experiment_name != 'classical_baselines':
                self.save_results(experiment_name, results)
        
        return analysis
    
    def run_complete_phase1(self, datasets, n_classes=2):
        print("Phase 1: Quantum ML Baseline Experiment")
        print(f"Target runtime: 6-10 hours")
        
        start_time = datetime.now()
        
        # Prepare dataset
        quantum_data = self.prepare_dataset(datasets, n_classes=n_classes)
        print(f"Dataset prepared: {len(quantum_data['train_features'])} train, {len(quantum_data['test_features'])} test")
        
        # Run all experiments
        self.establish_classical_baselines(quantum_data)
        self.run_encoding_comparison(quantum_data)
        self.run_dimension_scaling(datasets)
        self.run_depth_analysis(quantum_data)
        self.run_cross_validation(quantum_data)
        
        # Generate analysis
        analysis = self.generate_analysis_report()
        
        total_time = datetime.now() - start_time
        print(f"Phase 1 completed in {total_time}")
        print(f"Results saved in: {self.results_dir}")
        
        return analysis

    def run_medium_scale(self, datasets, n_classes=2):
        """Run medium-scale experiment optimized for 16GB RAM laptops (3-5 hours)."""
        original_encodings = self.encoding_strategies
        original_trials = self.trials_per_condition
        original_dims = self.feature_dimensions
        original_depths = self.circuit_depths

        # Medium-scale configuration for laptops
        self.encoding_strategies = ['amplitude', 'angle', 'basis']  # all encodings
        self.trials_per_condition = 2  # maintain statistical validity
        self.feature_dimensions = [2, 4, 6]  # skip 8D to save memory
        self.circuit_depths = [2, 3, 4]  # skip 8-layer circuits

        print("🔄 Medium-scale run (≈ 3-5 hours) optimized for 16GB RAM")
        print("Reduced: 8D features → 6D max, 8-layer circuits → 4 max")
        
        start_time = datetime.now()
        
        # Prepare dataset with reduced samples
        quantum_data = self.prepare_dataset(datasets, n_classes=n_classes, samples_per_class=100)
        print(f"Dataset: {len(quantum_data['train_features'])} train, {len(quantum_data['test_features'])} test")

        # Patch classifier for memory efficiency
        orig_init = QuantumMultiClassifier.__init__
        def memory_efficient_init(self_q, encoder, n_classes=2, n_layers=2, optimizer='SPSA', learning_rate=0.05, maxiter=50, shots_final=512):  # type: ignore
            orig_init(self_q, encoder, n_classes=n_classes, n_layers=n_layers, optimizer=optimizer, learning_rate=learning_rate, maxiter=maxiter, shots_final=shots_final)
        QuantumMultiClassifier.__init__ = memory_efficient_init  # type: ignore
        
        orig_train = QuantumMultiClassifier.train
        def memory_efficient_train(self_q, X, y, batch_size=15, n_epochs=2, show_progress=True):  # type: ignore
            return orig_train(self_q, X, y, batch_size=batch_size, n_epochs=n_epochs, show_progress=show_progress)
        QuantumMultiClassifier.train = memory_efficient_train  # type: ignore

        # Patch cost function to use lower shot counts during training
        orig_cost = QuantumMultiClassifier.cost_function
        def memory_efficient_cost(self_q, parameters, X, y, shots):  # type: ignore
            # Cap shots at 256 to save memory
            efficient_shots = min(shots, 256)
            return orig_cost(self_q, parameters, X, y, efficient_shots)
        QuantumMultiClassifier.cost_function = memory_efficient_cost  # type: ignore

        try:
            # Run all experiments with memory-efficient settings
            print("Establishing classical baselines...")
            self.establish_classical_baselines(quantum_data)
            
            print("Running encoding comparison (reduced shots)...")
            self.run_encoding_comparison(quantum_data)
            
            print("Running dimension scaling analysis...")
            self.run_dimension_scaling(datasets)
            
            print("Running depth analysis...")
            self.run_depth_analysis(quantum_data)
            
            print("Running cross-validation...")
            self.run_cross_validation(quantum_data, n_splits=3, max_folds=3)

            # Generate analysis
            analysis = self.generate_analysis_report()
            
            total_time = datetime.now() - start_time
            estimated_models = len(self.encoding_strategies) * self.trials_per_condition * (1 + len(self.feature_dimensions) + len(self.circuit_depths)) + 9
            print(f"✅ Medium-scale experiment completed in {total_time}")
            print(f"Total quantum models trained: ~{estimated_models}")
            print(f"Results saved in: {self.results_dir}")
            
        finally:
            # Always restore methods
            QuantumMultiClassifier.__init__ = orig_init  # type: ignore
            QuantumMultiClassifier.train = orig_train  # type: ignore
            QuantumMultiClassifier.cost_function = orig_cost  # type: ignore

            # Restore original configuration
            self.encoding_strategies = original_encodings
            self.trials_per_condition = original_trials
            self.feature_dimensions = original_dims
            self.circuit_depths = original_depths

        return analysis

    def run_quick_demo(self, datasets, n_classes=2):
        """Run a fast end-to-end demo of all analyses with sharply reduced scope."""
        original_encodings = self.encoding_strategies
        original_trials = self.trials_per_condition
        original_dims = self.feature_dimensions
        original_depths = self.circuit_depths

        # Narrow scope for speed
        self.encoding_strategies = ['amplitude', 'angle']  # two encodings
        self.trials_per_condition = 1
        self.feature_dimensions = [original_dims[0]]  # single dimension
        self.circuit_depths = [original_depths[0]]    # single depth

        print("⚡ Quick demo run (≈ few minutes) starting...")
        quantum_data = self.prepare_dataset(datasets, n_classes=n_classes, samples_per_class=30)
        print(f"Demo dataset: {len(quantum_data['train_features'])} train / {len(quantum_data['test_features'])} test")

        # Patch classifier to reduce layers/epochs
        orig_init = QuantumMultiClassifier.__init__
        def patched_init(self_q, encoder, n_classes=2, n_layers=1, optimizer='SPSA', learning_rate=0.05, maxiter=20, shots_final=256):  # type: ignore
            orig_init(self_q, encoder, n_classes=n_classes, n_layers=1, optimizer=optimizer, learning_rate=learning_rate, maxiter=20, shots_final=256)
        QuantumMultiClassifier.__init__ = patched_init  # type: ignore
        orig_train = QuantumMultiClassifier.train
        def patched_train(self_q, X, y, batch_size=15, n_epochs=1, show_progress=False):  # type: ignore
            return orig_train(self_q, X, y, batch_size=batch_size, n_epochs=1, show_progress=show_progress)
        QuantumMultiClassifier.train = patched_train  # type: ignore

        # Run reduced experiments
        self.establish_classical_baselines(quantum_data)
        self.run_encoding_comparison(quantum_data)
        self.run_dimension_scaling(datasets)
        self.run_depth_analysis(quantum_data)
        self.run_cross_validation(quantum_data, n_splits=2, max_folds=2)

        # Restore methods
        QuantumMultiClassifier.__init__ = orig_init  # type: ignore
        QuantumMultiClassifier.train = orig_train  # type: ignore

        analysis = self.generate_analysis_report()
        print("✅ Quick demo complete. See results in", self.results_dir)

        # Restore original configuration
        self.encoding_strategies = original_encodings
        self.trials_per_condition = original_trials
        self.feature_dimensions = original_dims
        self.circuit_depths = original_depths

        return analysis


# Usage
if __name__ == "__main__":
    experiment = Phase1BaselineExperiment()
    print("Phase 1 framework ready for execution")
    print("Call experiment.run_complete_phase1(datasets) to start")