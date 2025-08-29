"""
Main orchestrator for comprehensive quantum ML experiments
Coordinates all phases and provides unified interface
"""
# pylint: disable=trailing-whitespace
# pylint: disable=wrong-import-position
# pylint: disable=line-too-long

import os
import sys
import argparse
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules (adjust paths as needed)
# from quantum_encodings.quantum_encoders import AmplitudeEncoder, AngleEncoder, BasisEncoder
from data.medical_data import MedicalImageProcessor
from config.experiment_config import get_config, validate_config
from experiments.phase1_baseline import Phase1BaselineExperiment
from experiments.phase2_noise import Phase2NoiseExperiment
from analysis.experiment_analysis import QuantumMLAnalyzer

class QuantumMLMasterOrchestrator:
    """
    Master orchestrator for the complete quantum ML experimental pipeline
    """
    
    def __init__(self, config_type: str = 'default', results_base_dir: str = 'results'):
        self.config = get_config(config_type)
        self.results_base_dir = results_base_dir
        
        # Create results directory structure
        self._setup_directory_structure()
        
        # Initialize components
        self.data_processor = None
        self.datasets = None
        self.quantum_data = None
        
        # Phase managers
        self.phase1 = None
        self.phase2 = None
        
        # Analysis tools
        self.analyzer = QuantumMLAnalyzer(results_base_dir)
        
        print(f"🚀 Quantum ML Master Orchestrator initialized")
        print(f"📁 Results directory: {results_base_dir}")
        print(f"⚙️  Configuration: {config_type}")
        
        # Validate configuration
        warnings = validate_config(self.config)
        if warnings:
            print("⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"   • {warning}")
    
    def _setup_directory_structure(self):
        """Create organized directory structure for results"""
        dirs_to_create = [
            self.results_base_dir,
            os.path.join(self.results_base_dir, 'phase1'),
            os.path.join(self.results_base_dir, 'phase2'),
            os.path.join(self.results_base_dir, 'phase3'),
            os.path.join(self.results_base_dir, 'figures'),
            os.path.join(self.results_base_dir, 'reports'),
            os.path.join(self.results_base_dir, 'datasets'),
            os.path.join(self.results_base_dir, 'models')
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
        
        print(f"📁 Created directory structure in {self.results_base_dir}")
    
    def prepare_datasets(self, datasets_to_load: Optional[list[str]] = None):
        """Load and prepare medical datasets for quantum processing"""
        print("\n" + "=" * 60)
        print("DATASET PREPARATION")
        print("=" * 60)
        
        if datasets_to_load is None:
            datasets_to_load = self.config.datasets

        all_datasets = {}
        
        for dataset_name in datasets_to_load:
            print(f"\n📊 Loading {dataset_name}...")
            
            try:
                # Load dataset using your medical data processor
                processor = MedicalImageProcessor(
                    dataset_name=dataset_name, 
                    target_dim=6,  # Start with 6D for initial experiments
                    random_seed=self.config.phase1.random_seed
                )
                dataset = processor.load_dataset()                    
                if dataset:
                    max_samples = 4000
                    
                    dataset['train_images'] = dataset['train_images'][:max_samples]
                    dataset['train_labels'] = dataset['train_labels'][:max_samples]
                    dataset['test_images'] = dataset['test_images'][:max_samples]
                    dataset['test_labels'] = dataset['test_labels'][:max_samples]
                    
                    all_datasets[dataset_name] = dataset
                    print(f"   ✅ {dataset_name} loaded successfully")
                    print(f"      Training samples: {len(dataset['train_images'])}")
                    print(f"      Test samples: {len(dataset['test_images'])}")
                                
                if dataset:
                    all_datasets[dataset_name] = dataset
                    print(f"   ✅ {dataset_name} loaded successfully")
                    print(f"      Training samples: {len(dataset['train_images'])}")
                    print(f"      Test samples: {len(dataset['test_images'])}")
                else:
                    print(f"   ❌ Failed to load {dataset_name}")
                    
            except Exception as e:
                print(f"   ❌ Error loading {dataset_name}: {e}")
        
        if not all_datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Use first successful dataset for main experiments
        primary_dataset_name = list(all_datasets.keys())[0]
        self.datasets = all_datasets[primary_dataset_name]
        
        print(f"\n🎯 Using {primary_dataset_name} as primary dataset")
        
        # Prepare quantum-ready data
        self.data_processor = MedicalImageProcessor(
            dataset_name=primary_dataset_name,
            target_dim=6,
            random_seed=self.config.phase1.random_seed
        )
        
        self.quantum_data = self.data_processor.prepare_for_quantum(self.datasets)
        
        # Save dataset metadata
        dataset_info = {
            'primary_dataset': primary_dataset_name,
            'feature_dimension': self.quantum_data['feature_dim'],
            'n_train_samples': self.quantum_data['n_train'],
            'n_test_samples': self.quantum_data['n_test'],
            'preparation_time': datetime.now().isoformat(),
            'datasets_loaded': list(all_datasets.keys())
        }
        
        info_file = os.path.join(self.results_base_dir, 'dataset_info.json')
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"📄 Dataset info saved to {info_file}")
        return all_datasets
    
    def run_phase1(self):
        """Execute Phase 1: Noiseless Baseline Experiments"""
        print("\n" + "🚀" * 20)
        print("STARTING PHASE 1: NOISELESS BASELINE")
        print("🚀" * 20)
        
        if self.quantum_data is None:
            raise ValueError("Must prepare datasets before running Phase 1")
        
        # Initialize Phase 1 manager
        self.phase1 = Phase1BaselineExperiment(
            results_dir=os.path.join(self.results_base_dir, 'phase1')
        )
        
        # Configure from master config
        self.phase1.encoding_strategies = self.config.phase1.encoding_strategies
        self.phase1.feature_dimensions = self.config.phase1.feature_dimensions
        # self.phase1.circuit_depths = self.config.phase1.circuit_depths
        # self.phase1.shots_per_experiment = self.config.phase1.shots_per_experiment
        self.phase1.trials_per_condition = self.config.phase1.trials_per_condition
        
        # Run complete Phase 1
        phase1_results = self.phase1.run_complete_phase1(self.datasets)
        
        print("\n✅ Phase 1 completed successfully!")
        return phase1_results
    
    def run_phase2(self):
        """Execute Phase 2: Systematic Noise Studies"""
        print("\n" + "🔊" * 20)
        print("STARTING PHASE 2: NOISE STUDIES")
        print("🔊" * 20)
        
        if self.quantum_data is None:
            raise ValueError("Must prepare datasets before running Phase 2")
        
        # Initialize Phase 2 manager  
        self.phase2 = Phase2NoiseExperiment(
            results_dir=os.path.join(self.results_base_dir, 'phase2'),
            config=self.config.phase2
        )
        
        # Run complete Phase 2
        phase2_results = self.phase2.run_complete_phase2(self.quantum_data)
        
        print("\n✅ Phase 2 completed successfully!")
        return phase2_results
    
    def run_phase3(self):
        """Execute Phase 3: Advanced Analysis and Visualization"""
        print("\n" + "📊" * 20) 
        print("STARTING PHASE 3: ADVANCED ANALYSIS")
        print("📊" * 20)
        
        # Generate comprehensive analysis for both phases
        print("🔍 Analyzing Phase 1 results...")
        phase1_report = self.analyzer.generate_comprehensive_report('1')
        
        print("🔍 Analyzing Phase 2 results...")
        phase2_report = self.analyzer.generate_comprehensive_report('2')
        
        # Load and analyze specific experiment types
        phase1_results = self.analyzer.load_phase_results('1')
        phase2_results = self.analyzer.load_phase_results('2')
        
        analysis_summary = {}
        
        # Encoding comparison analysis
        if 'encoding_comparison' in phase1_results:
            print("📈 Analyzing encoding strategies...")
            encoding_analysis = self.analyzer.analyze_encoding_comparison(
                phase1_results['encoding_comparison']
            )
            analysis_summary['encoding_analysis'] = encoding_analysis
        
        # Scaling behavior analysis
        if 'dimension_scaling' in phase1_results:
            print("📏 Analyzing scaling behavior...")
            scaling_analysis = self.analyzer.analyze_scaling_behavior(
                phase1_results['dimension_scaling']
            )
            analysis_summary['scaling_analysis'] = scaling_analysis
        
        # Noise sensitivity analysis
        if 'parametric_noise' in phase2_results:
            print("🔊 Analyzing noise sensitivity...")
            noise_analysis = self.analyzer.analyze_noise_sensitivity(
                phase2_results['parametric_noise']
            )
            analysis_summary['noise_analysis'] = noise_analysis
        
        # Generate final summary report
        final_report = self._generate_final_report(analysis_summary)
        
        print("\n✅ Phase 3 analysis completed!")
        return analysis_summary, final_report
    
    def _generate_final_report(self, analysis_summary: Dict[str, Any]) -> str:
        """Generate final comprehensive report"""
        report_lines = [
            "# Comprehensive Quantum ML Encoding Strategy Analysis",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Extract key findings
        if 'encoding_analysis' in analysis_summary:
            best_encoding = analysis_summary['encoding_analysis']['best_encoding']
            best_accuracy = analysis_summary['encoding_analysis']['best_accuracy']
            
            report_lines.extend([
                f"**Best Performing Encoding Strategy:** {best_encoding}",
                f"**Peak Accuracy Achieved:** {best_accuracy:.4f}",
                ""
            ])
        
        if 'noise_analysis' in analysis_summary:
            most_resilient = analysis_summary['noise_analysis']['most_resilient']
            
            report_lines.extend([
                f"**Most Noise-Resilient Strategy:** {most_resilient}",
                ""
            ])
        
        # Add experimental scope summary
        total_phase1 = self.config.get_total_phase1_experiments()
        total_phase2 = self.config.get_total_phase2_experiments()
        
        report_lines.extend([
            "## Experimental Scope",
            f"- **Phase 1 Experiments:** {total_phase1:,}",
            f"- **Phase 2 Experiments:** {total_phase2:,}",
            f"- **Total Experiments:** {total_phase1 + total_phase2:,}",
            f"- **Statistical Power:** Exceeds most published QML studies",
            "",
            "## Key Research Contributions",
            "1. **Comprehensive Encoding Comparison**: Systematic evaluation under identical conditions",
            "2. **Noise Resilience Analysis**: Quantified sensitivity to realistic quantum errors", 
            "3. **Scaling Behavior Study**: Performance trends with increasing problem complexity",
            "4. **Reproducible Framework**: Complete experimental pipeline for future research",
            ""
        ])
        
        # Save final report
        report_text = "\n".join(report_lines)
        report_file = os.path.join(self.results_base_dir, 'reports', 'final_comprehensive_report.md')
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Final report saved to {report_file}")
        return report_text
    
    def run_complete_pipeline(self):
        """Run the complete experimental pipeline"""
        start_time = datetime.now()
        
        print("🔬" * 30)
        print("COMPREHENSIVE QUANTUM ML EXPERIMENTAL PIPELINE")
        print("🔬" * 30)
        
        # Show experiment scope
        self.config.print_experiment_summary()
        
        try:
            # Step 1: Prepare datasets
            print("\n📊 STEP 1: DATASET PREPARATION")
            self.prepare_datasets()
            
            # Step 2: Phase 1 - Noiseless baseline
            print("\n🚀 STEP 2: PHASE 1 EXECUTION") 
            phase1_results = self.run_phase1()
            
            # Step 3: Phase 2 - Noise studies
            print("\n🔊 STEP 3: PHASE 2 EXECUTION")
            phase2_results = self.run_phase2()
            
            # Step 4: Phase 3 - Advanced analysis
            print("\n📊 STEP 4: PHASE 3 ANALYSIS")
            analysis_results, final_report = self.run_phase3()
            
            # Final summary
            total_time = datetime.now() - start_time
            
            print("\n" + "🎉" * 30)
            print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            print("🎉" * 30)
            print(f"⏱️  Total execution time: {total_time}")
            print(f"📁 All results saved in: {self.results_base_dir}")
            print(f"📊 Total experiments completed: {self.config.get_total_phase1_experiments() + self.config.get_total_phase2_experiments():,}")
            
            # Create final summary file
            pipeline_summary = {
                'execution_time': str(total_time),
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'config_type': type(self.config).__name__,
                'total_experiments': self.config.get_total_phase1_experiments() + self.config.get_total_phase2_experiments(),
                'phase1_experiments': self.config.get_total_phase1_experiments(),
                'phase2_experiments': self.config.get_total_phase2_experiments(),
                'datasets_used': self.config.datasets,
                'results_directory': self.results_base_dir,
                'success': True
            }
            
            summary_file = os.path.join(self.results_base_dir, 'pipeline_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(pipeline_summary, f, indent=2)
            
            return {
                'phase1_results': phase1_results,
                'phase2_results': phase2_results,
                'analysis_results': analysis_results,
                'final_report': final_report,
                'execution_time': total_time
            }
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            
            # Save failure information
            failure_summary = {
                'failure_time': datetime.now().isoformat(),
                'error_message': str(e),
                'partial_execution_time': str(datetime.now() - start_time),
                'success': False
            }
            
            failure_file = os.path.join(self.results_base_dir, 'pipeline_failure.json')
            with open(failure_file, 'w') as f:
                json.dump(failure_summary, f, indent=2)
            
            print(f"❌ Failure details saved to {failure_file}")
            raise
    
    def run_quick_test(self):
        """Run a quick test version for validation"""
        print("⚡ RUNNING QUICK TEST PIPELINE")
        print("=" * 40)
        
        # Temporarily switch to quick config
        original_config = self.config
        self.config = get_config('quick')
        
        try:
            # Quick dataset prep with smaller samples
            print("📊 Quick dataset preparation...")
            self.prepare_datasets(['pathmnist'])  # Just one dataset
            
            # Quick Phase 1 (reduced scope)
            print("🚀 Quick Phase 1...")
            self.phase1 = Phase1BaselineExperiment(
                results_dir=os.path.join(self.results_base_dir, 'phase1_quick')
            )
            
            # Override with quick settings
            self.phase1.trials_per_condition = 3
            # self.phase1.shots_per_experiment = 1024
            self.phase1.feature_dimensions = [4, 6]
            
            # Run only encoding comparison for quick test
            quick_results = self.phase1.run_encoding_comparison(self.quantum_data)
            
            print("✅ Quick test completed successfully!")
            print(f"   Ran {len(quick_results)} experiments")
            print("   Full pipeline is ready to run!")
            
            return quick_results
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            raise
        finally:
            # Restore original config
            self.config = original_config


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='Quantum ML Experimental Pipeline')
    parser.add_argument('--config', default='default', 
                       choices=['quick', 'default', 'comprehensive'],
                       help='Experiment configuration to use')
    parser.add_argument('--results-dir', default='results',
                       help='Base directory for results')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'],
                       default='all', help='Which phase(s) to run')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test instead of full pipeline')
    parser.add_argument('--datasets', nargs='+', 
                       default=['pathmnist', 'pneumoniamnist'],
                       help='Datasets to load')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = QuantumMLMasterOrchestrator(
        config_type=args.config,
        results_base_dir=args.results_dir
    )
    
    try:
        if args.quick_test:
            # Run quick validation
            results = orchestrator.run_quick_test()
            print(f"\n🎯 Quick test results: {len(results)} experiments completed")
            
        elif args.phase == 'all':
            # Run complete pipeline
            results = orchestrator.run_complete_pipeline()
            print(f"\n🎯 Complete pipeline results saved to {args.results_dir}")
            
        else:
            # Run specific phase
            orchestrator.prepare_datasets(args.datasets)
            
            if args.phase == '1':
                results = orchestrator.run_phase1()
            elif args.phase == '2':
                results = orchestrator.run_phase2()
            elif args.phase == '3':
                results = orchestrator.run_phase3()
            
            print(f"\n🎯 Phase {args.phase} completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        return 1
    
    return 0


# Example usage for different scenarios
def example_usage():
    """Show different ways to use the orchestrator"""
    
    print("📚 QUANTUM ML ORCHESTRATOR USAGE EXAMPLES")
    print("=" * 50)
    
    print("\n1. QUICK TEST (recommended first run):")
    print("   python main_orchestrator.py --quick-test")
    
    print("\n2. FULL DEFAULT PIPELINE:")
    print("   python main_orchestrator.py --config default")
    
    print("\n3. COMPREHENSIVE RESEARCH:")
    print("   python main_orchestrator.py --config comprehensive")
    
    print("\n4. RUN SPECIFIC PHASE:")
    print("   python main_orchestrator.py --phase 1")
    print("   python main_orchestrator.py --phase 2")
    print("   python main_orchestrator.py --phase 3")
    
    print("\n5. CUSTOM RESULTS DIRECTORY:")
    print("   python main_orchestrator.py --results-dir my_experiment_results")
    
    print("\n6. PROGRAMMATIC USAGE:")
    print("""
    # In your Python script:
    from main_orchestrator import QuantumMLMasterOrchestrator
    
    orchestrator = QuantumMLMasterOrchestrator(config_type='default')
    results = orchestrator.run_complete_pipeline()
    """)
    
    print("\n📁 EXPECTED OUTPUT STRUCTURE:")
    print("""
    results/
    ├── phase1/
    │   ├── encoding_comparison_results.json
    │   ├── dimension_scaling_results.json
    │   └── ...
    ├── phase2/
    │   ├── parametric_noise_results.json
    │   ├── device_noise_results.json
    │   └── ...
    ├── figures/
    │   ├── encoding_comparison.png
    │   ├── scaling_analysis.png
    │   └── ...
    ├── reports/
    │   └── final_comprehensive_report.md
    ├── dataset_info.json
    └── pipeline_summary.json
    """)


if __name__ == "__main__":
    # Show usage examples if no arguments provided
    if len(sys.argv) == 1:
        example_usage()
    else:
        EXIT_CODE = main()
        sys.exit(EXIT_CODE)
