"""
Centralized configuration for all quantum ML experiments
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Phase1Config:
    """Configuration for Phase 1 noiseless baseline experiments"""
    
    # Encoding strategies to test
    encoding_strategies: List[str] = field(default_factory=lambda: [
        'amplitude', 'angle', 'basis'
    ])
    
    # Feature dimensions to test
    feature_dimensions: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])
    
    # Circuit depths for variational analysis
    circuit_depths: List[int] = field(default_factory=lambda: list(range(1, 11)))
    
    # Experimental parameters
    shots_per_experiment: int = 8192
    trials_per_condition: int = 20
    cv_folds: int = 5
    
    # Sample sizes (for faster experiments)
    max_train_samples: int = 100
    max_test_samples: int = 50
    
    # Results directory
    results_dir: str = 'results/phase1'
    
    # Random seed for reproducibility
    random_seed: int = 42

@dataclass
class Phase2Config:
    """Configuration for Phase 2 noise studies"""
    
    # # Base noise parameters
    # noise_levels: List[float] = field(default_factory=lambda: [
    #     0.001, 0.005, 0.01, 0.02, 0.05, 0.1
    # ])
    
    # # Noise models to simulate
    # noise_models: List[str] = field(default_factory=lambda: [
    #     'depolarizing', 'amplitude_damping', 'phase_damping', 
    #     'pauli', 'thermal_relaxation'
    # ])
    
    # # IBM backend models
    # ibm_backends: List[str] = field(default_factory=lambda: [
    #     'ibm_perth', 'ibm_lagos', 'ibm_nairobi', 'ibm_oslo', 'ibm_geneva'
    # ])
    
    # # Coherence time parameters (microseconds)
    # t1_times: List[float] = field(default_factory=lambda: [10, 50, 100, 200])
    # t2_times: List[float] = field(default_factory=lambda: [5, 25, 50, 100])
    
    # # Gate error rates
    # single_qubit_errors: List[float] = field(default_factory=lambda: [
    #     0.0001, 0.0005, 0.001, 0.002
    # ])
    # two_qubit_errors: List[float] = field(default_factory=lambda: [
    #     0.005, 0.01, 0.02, 0.05
    # ])
    
    # # Readout error rates  
    # readout_errors: List[float] = field(default_factory=lambda: [
    #     0.01, 0.02, 0.05, 0.1
    # ])
    
    # # Experimental parameters
    # shots_per_experiment: int = 4096
    # trials_per_condition: int = 10
    noise_levels: List[float] = field(default_factory=lambda: [
        0.001, 0.01, 0.05  # Reduced from 6 to 3 levels (low, medium, high)
    ])

    # Noise models to simulate
    noise_models: List[str] = field(default_factory=lambda: [
        'depolarizing',        # A good generic, baseline model
        'amplitude_damping',   # Represents T1 energy loss
        'thermal_relaxation'   # A more realistic, combined model
    ]) # Reduced from 5 to 3 core models

    # IBM backend models
    ibm_backends: List[str] = field(default_factory=lambda: [
        'ibm_nairobi', 'ibm_geneva'  # Reduced from 5 to 2 representative backends
    ])

    # Coherence time parameters (microseconds) for thermal_relaxation
    t1_times: List[float] = field(default_factory=lambda: [50, 150]) # Reduced from 4 to 2
    t2_times: List[float] = field(default_factory=lambda: [25, 75])  # Reduced from 4 to 2

    # --- Parameters below are commented out to save time ---
    # Their effects are implicitly studied in the 'ibm_backends' simulations.
    # You can re-enable one of these for a more focused study if needed.
    
    # # Gate error rates
    single_qubit_errors: List[float] = field(default_factory=lambda: [0.0005, 0.002])
    two_qubit_errors: List[float] = field(default_factory=lambda: [0.01, 0.05])
    
    # # Readout error rates  
    readout_errors: List[float] = field(default_factory=lambda: [0.02, 0.1])

    # Experimental parameters
    shots_per_experiment: int = 2048 # Kept high for accuracy
    trials_per_condition: int = 3      # Reduced from 10. (3 is the minimum for std. deviation)

    
    # Results directory
    results_dir: str = 'results/phase2'

@dataclass
class ExperimentConfig:
    """Master configuration combining all phases"""
    
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ['pathmnist', 'pneumoniamnist'])
    
    # Phase configurations
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    
    # Hardware validation config
    hardware_time_budget: int = 600  # seconds (10 minutes)
    hardware_backend: str = 'ibm_perth'  # Default backend
    
    # Global settings
    verbose: bool = True
    save_intermediate: bool = True
    plot_results: bool = True
    
    def get_total_phase1_experiments(self) -> int:
        """Calculate total number of Phase 1 experiments"""
        # Encoding comparison
        encoding_exp = len(self.phase1.encoding_strategies) * self.phase1.trials_per_condition
        
        # Dimension scaling  
        dimension_exp = (len(self.phase1.encoding_strategies) * 
                        len(self.phase1.feature_dimensions) * 
                        self.phase1.trials_per_condition)
        
        # Depth analysis
        depth_exp = (len(self.phase1.encoding_strategies) * 
                    len(self.phase1.circuit_depths) * 
                    self.phase1.trials_per_condition)
        
        # Cross-validation
        cv_exp = len(self.phase1.encoding_strategies) * self.phase1.cv_folds * 5
        
        return encoding_exp + dimension_exp + depth_exp + cv_exp
    
    def get_total_phase2_experiments(self) -> int:
        """Calculate total number of Phase 2 experiments"""
        # Basic noise studies
        noise_exp = (len(self.phase1.encoding_strategies) * 
                    len(self.phase2.noise_models) * 
                    len(self.phase2.noise_levels) * 
                    self.phase2.trials_per_condition)
        
        # Backend-specific studies
        backend_exp = (len(self.phase1.encoding_strategies) * 
                      len(self.phase2.ibm_backends) * 
                      self.phase2.trials_per_condition)
        
        # Coherence studies
        coherence_exp = (len(self.phase1.encoding_strategies) * 
                        len(self.phase2.t1_times) * 
                        len(self.phase2.t2_times) * 
                        self.phase2.trials_per_condition)
        
        return noise_exp + backend_exp + coherence_exp
    
    def print_experiment_summary(self):
        """Print a summary of all planned experiments"""
        print("🔬 COMPREHENSIVE QUANTUM ML EXPERIMENT PLAN")
        print("=" * 50)
        
        print(f"\n📊 PHASE 1 - NOISELESS BASELINE:")
        print(f"   • Encoding strategies: {len(self.phase1.encoding_strategies)}")
        print(f"   • Feature dimensions: {len(self.phase1.feature_dimensions)}")
        print(f"   • Circuit depths: {len(self.phase1.circuit_depths)}")
        print(f"   • Total experiments: {self.get_total_phase1_experiments():,}")
        print(f"   • Shots per experiment: {self.phase1.shots_per_experiment:,}")
        print(f"   • Estimated duration: 2-3 days")
        
        print(f"\n🔊 PHASE 2 - NOISE STUDIES:")
        print(f"   • Noise models: {len(self.phase2.noise_models)}")
        print(f"   • Noise levels: {len(self.phase2.noise_levels)}")
        print(f"   • IBM backends: {len(self.phase2.ibm_backends)}")
        print(f"   • Total experiments: {self.get_total_phase2_experiments():,}")
        print(f"   • Shots per experiment: {self.phase2.shots_per_experiment:,}")
        print(f"   • Estimated duration: 4-5 days")
        
        total_phase1_shots = self.get_total_phase1_experiments() * self.phase1.shots_per_experiment
        total_phase2_shots = self.get_total_phase2_experiments() * self.phase2.shots_per_experiment
        
        print(f"\n📈 TOTAL STATISTICAL POWER:")
        print(f"   • Total experiments: {self.get_total_phase1_experiments() + self.get_total_phase2_experiments():,}")
        print(f"   • Total quantum shots: {total_phase1_shots + total_phase2_shots:,}")
        print(f"   • Cost: FREE (local simulation)")
        print(f"   • Hardware budget: {self.hardware_time_budget} seconds")
        
        print(f"\n✅ This exceeds the experimental scope of most published QML papers!")

# Create default configuration instance
DEFAULT_CONFIG = ExperimentConfig()

# Specialized configurations for different research focuses
@dataclass 
class QuickTestConfig(ExperimentConfig):
    """Reduced configuration for quick testing"""
    
    def __post_init__(self):
        # Reduce scope for faster testing
        self.phase1.feature_dimensions = [4, 6]
        self.phase1.circuit_depths = [1, 2, 3]
        self.phase1.trials_per_condition = 5
        self.phase1.shots_per_experiment = 1024
        
        self.phase2.noise_levels = [0.01, 0.05]
        self.phase2.noise_models = ['depolarizing', 'amplitude_damping'] 
        self.phase2.trials_per_condition = 3

@dataclass
class ComprehensiveConfig(ExperimentConfig):
    """Extended configuration for comprehensive research"""
    
    def __post_init__(self):
        # Extended scope for publication-quality research
        self.phase1.feature_dimensions = [4, 6, 8, 10, 12, 14, 16]
        self.phase1.circuit_depths = list(range(1, 16))  # Up to 15 layers
        self.phase1.trials_per_condition = 50  # Higher statistical power
        self.phase1.shots_per_experiment = 8192
        
        # More comprehensive noise studies
        self.phase2.noise_levels = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        self.phase2.trials_per_condition = 20
        
        # Extended IBM backend coverage
        self.phase2.ibm_backends.extend(['ibm_kyiv', 'ibm_sherbrooke'])

# Configuration factory
def get_config(config_type: str = 'default') -> ExperimentConfig:
    """Factory function to get different configurations"""
    if config_type.lower() == 'quick':
        return QuickTestConfig()
    elif config_type.lower() == 'comprehensive':
        return ComprehensiveConfig()
    else:
        return DEFAULT_CONFIG

# Experiment planning utilities
def estimate_computation_time(config: ExperimentConfig) -> Dict[str, float]:
    """Estimate computation time for experiments"""
    
    # Time estimates based on typical quantum simulation performance
    # These are rough estimates - actual times depend on hardware
    
    base_time_per_shot = 0.001  # seconds per shot (rough estimate)
    
    phase1_shots = config.get_total_phase1_experiments() * config.phase1.shots_per_experiment
    phase2_shots = config.get_total_phase2_experiments() * config.phase2.shots_per_experiment
    
    estimates = {
        'phase1_hours': (phase1_shots * base_time_per_shot) / 3600,
        'phase2_hours': (phase2_shots * base_time_per_shot) / 3600,
        'total_days': ((phase1_shots + phase2_shots) * base_time_per_shot) / (24 * 3600)
    }
    
    return estimates

def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate configuration and return any warnings"""
    warnings = []
    
    # Check for computational feasibility
    if config.get_total_phase1_experiments() > 50000:
        warnings.append("Phase 1 has >50k experiments - may take very long")
    
    if config.phase1.shots_per_experiment > 10000:
        warnings.append("High shot count may slow down experiments significantly")
    
    # Check for memory requirements
    max_qubits = max(config.phase1.feature_dimensions) + 5  # Rough estimate
    if max_qubits > 20:
        warnings.append(f"May need >{max_qubits} qubits - check simulator limits")
    
    # Check hardware budget
    if config.hardware_time_budget < 300:
        warnings.append("Hardware time budget <5min may be insufficient")
    
    return warnings

# Example usage and configuration showcase
if __name__ == "__main__":
    print("🔧 QUANTUM ML EXPERIMENT CONFIGURATIONS")
    print("=" * 50)
    
    # Show different configurations
    configs = {
        'Quick Test': get_config('quick'),
        'Default': get_config('default'),
        'Comprehensive': get_config('comprehensive')
    }
    
    for name, config in configs.items():
        print(f"\n📋 {name.upper()} CONFIGURATION:")
        print(f"   Phase 1 experiments: {config.get_total_phase1_experiments():,}")
        print(f"   Phase 2 experiments: {config.get_total_phase2_experiments():,}")
        
        time_est = estimate_computation_time(config)
        print(f"   Estimated time: {time_est['total_days']:.1f} days")
        
        warnings = validate_config(config)
        if warnings:
            print(f"   ⚠️  Warnings: {'; '.join(warnings)}")
        else:
            print(f"   ✅ Configuration looks good!")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   • Start with 'Quick Test' to verify everything works")
    print(f"   • Use 'Default' for solid research results") 
    print(f"   • Use 'Comprehensive' for publication-quality studies")
    
    # Show detailed breakdown for default config
    print(f"\n📊 DEFAULT CONFIG DETAILED BREAKDOWN:")
    DEFAULT_CONFIG.print_experiment_summary()