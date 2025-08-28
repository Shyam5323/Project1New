"""
Comprehensive analysis framework for quantum ML experiments
Handles visualization, statistical analysis, and report generation
"""

# pylint: disable=trailing-whitespace
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=line-too-long


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class QuantumMLAnalyzer:
    """
    Comprehensive analysis tool for quantum ML experiment results
    """
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.results_cache = {}
        
    def load_phase_results(self, phase: str) -> Dict[str, pd.DataFrame]:
        """Load all results from a specific phase"""
        phase_dir = os.path.join(self.results_dir, f'phase{phase}')
        
        if not os.path.exists(phase_dir):
            print(f"Warning: {phase_dir} does not exist")
            return {}
        
        results = {}
        
        for filename in os.listdir(phase_dir):
            if filename.endswith('_results.json'):
                experiment_name = filename.replace('_results.json', '')
                filepath = os.path.join(phase_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[experiment_name] = pd.DataFrame(data)
                    
        print(f"Loaded {len(results)} result sets from Phase {phase}")
        return results
    
    def analyze_encoding_comparison(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed analysis of encoding strategy comparison"""
        print("\n📊 ENCODING STRATEGY ANALYSIS")
        print("=" * 40)
        
        # Statistical summary
        summary_stats = results_df.groupby('encoding').agg({
            'accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std'],
            'precision': ['mean', 'std'], 
            'recall': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'n_qubits': 'first'
        }).round(4)
        
        print("\n📈 Performance Summary:")
        print(summary_stats)
        
        # Statistical significance testing
        encodings = results_df['encoding'].unique()
        significance_results = {}
        
        for i, enc1 in enumerate(encodings):
            for enc2 in encodings[i+1:]:
                group1 = results_df[results_df['encoding'] == enc1]['accuracy']
                group2 = results_df[results_df['encoding'] == enc2]['accuracy']
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(group1, group2)
                significance_results[f"{enc1}_vs_{enc2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        print("\n🔬 Statistical Significance (α=0.05):")
        for comparison, result in significance_results.items():
            status = "✅ Significant" if result['significant'] else "❌ Not significant"
            print(f"   {comparison}: p={result['p_value']:.4f} {status}")
        
        # Create visualization
        self._plot_encoding_comparison(results_df)
        
        # Fix the tuple access issue
        best_encoding = summary_stats[('accuracy', 'mean')].idxmax()
        best_accuracy = summary_stats[('accuracy', 'mean')].max()
        
        return {
            'summary_stats': summary_stats,
            'significance_tests': significance_results,
            'best_encoding': best_encoding,
            'best_accuracy': best_accuracy
        }
    
    def analyze_scaling_behavior(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how performance scales with problem size"""
        print("\n📏 SCALING ANALYSIS")
        print("=" * 40)
        
        # Analyze scaling with feature dimensions
        scaling_summary = results_df.groupby(['encoding', 'feature_dim']).agg({
            'accuracy': ['mean', 'std'],
            'n_qubits': 'first',
            'execution_time': 'mean'
        }).round(4)
        
        print("\n📊 Scaling Summary:")
        print(scaling_summary)
        
        # Calculate scaling coefficients for each encoding
        scaling_analysis = {}
        
        for encoding in results_df['encoding'].unique():
            encoding_data = results_df[results_df['encoding'] == encoding]
            
            # Fit linear regression: accuracy ~ feature_dim
            x = encoding_data['feature_dim'].values
            y = encoding_data['accuracy'].values
            
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                scaling_analysis[encoding] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value
                }
        
        print("\n📈 Scaling Coefficients:")
        for encoding, coeffs in scaling_analysis.items():
            print(f"   {encoding}: slope={coeffs['slope']:.4f}, R²={coeffs['r_squared']:.3f}")
        
        # Create scaling visualization
        self._plot_scaling_analysis(results_df)
        
        return {
            'scaling_summary': scaling_summary,
            'scaling_coefficients': scaling_analysis
        }
    
    def analyze_noise_sensitivity(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sensitivity to quantum noise"""
        print("\n🔊 NOISE SENSITIVITY ANALYSIS") 
        print("=" * 40)
        
        if 'noise_level' not in results_df.columns:
            print("No noise level data found - skipping noise analysis")
            return {}
        
        # Calculate noise degradation for each encoding
        noise_analysis = {}
        
        for encoding in results_df['encoding'].unique():
            encoding_data = results_df[results_df['encoding'] == encoding]
            
            # Get noiseless baseline (assuming noise_level = 0 or minimal)
            min_noise = encoding_data['noise_level'].min()
            baseline_acc = encoding_data[
                encoding_data['noise_level'] == min_noise
            ]['accuracy'].mean()
            
            # Calculate degradation at each noise level
            degradation = []
            for noise_level in sorted(encoding_data['noise_level'].unique()):
                noisy_acc = encoding_data[
                    encoding_data['noise_level'] == noise_level
                ]['accuracy'].mean()
                
                degradation.append({
                    'noise_level': noise_level,
                    'accuracy': noisy_acc,
                    'degradation': baseline_acc - noisy_acc,
                    'relative_degradation': (baseline_acc - noisy_acc) / baseline_acc
                })
            
            noise_analysis[encoding] = degradation
        
        # Find most noise-resilient encoding
        resilience_scores = {}
        for encoding, degradation_data in noise_analysis.items():
            # Average relative degradation as resilience metric
            avg_degradation = np.mean([d['relative_degradation'] for d in degradation_data])
            resilience_scores[encoding] = 1 - avg_degradation  # Higher is better
        
        most_resilient = max(resilience_scores.keys(), key=lambda k: resilience_scores[k])
        
        print(f"\n🛡️  Most noise-resilient: {most_resilient}")
        print("   Resilience scores:")
        for encoding, score in resilience_scores.items():
            print(f"     {encoding}: {score:.3f}")
        
        # Create noise visualization
        self._plot_noise_analysis(results_df)
        
        return {
            'noise_analysis': noise_analysis,
            'resilience_scores': resilience_scores,
            'most_resilient': most_resilient
        }
    
    def generate_comprehensive_report(self, phase: str) -> str:
        """Generate a comprehensive analysis report"""
        print(f"\n📝 GENERATING COMPREHENSIVE REPORT FOR PHASE {phase}")
        print("=" * 50)
        
        # Load results
        results = self.load_phase_results(phase)
        
        if not results:
            return "No results found to analyze"
        
        report_lines = []
        report_lines.append(f"# Phase {phase} Quantum ML Experiment Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Analyze each experiment type
        for experiment_name, df in results.items():
            report_lines.append(f"## {experiment_name.replace('_', ' ').title()}")
            report_lines.append("")
            
            # Initialize analysis variable
            analysis = {}
            
            if 'encoding' in df.columns:
                if experiment_name == 'encoding_comparison':
                    analysis = self.analyze_encoding_comparison(df)
                elif 'feature_dim' in df.columns:
                    analysis = self.analyze_scaling_behavior(df)
                elif 'noise_level' in df.columns:
                    analysis = self.analyze_noise_sensitivity(df)
                else:
                    # Default analysis for other encoding experiments
                    print(f"Performing basic analysis for {experiment_name}")
                    analysis = self._perform_basic_analysis(df)
            else:
                # Handle non-encoding experiments
                print(f"Performing basic analysis for {experiment_name}")
                analysis = self._perform_basic_analysis(df)
                
            # Add key findings to report
            report_lines.append("### Key Findings:")
            if 'best_encoding' in analysis:
                report_lines.append(f"- Best performing encoding: **{analysis['best_encoding']}**")
                report_lines.append(f"- Best accuracy achieved: **{analysis['best_accuracy']:.4f}**")
            elif 'summary' in analysis:
                for key, value in analysis['summary'].items():
                    report_lines.append(f"- {key}: **{value}**")
            
            report_lines.append("")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_file = os.path.join(self.results_dir, f'phase{phase}_comprehensive_report.md')
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Report saved to: {report_file}")
        return report_text
    
    def _perform_basic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic analysis for experiments that don't fit other categories"""
        summary = {}
        
        # Basic statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['accuracy', 'f1_score', 'precision', 'recall']:
                summary[f'avg_{col}'] = f"{df[col].mean():.4f}"
                summary[f'std_{col}'] = f"{df[col].std():.4f}"
        
        if 'accuracy' in df.columns:
            best_idx = df['accuracy'].idxmax()
            summary['best_accuracy'] = f"{df.loc[best_idx, 'accuracy']:.4f}"
            
        return {'summary': summary}
    
    def _plot_encoding_comparison(self, results_df: pd.DataFrame):
        """Create encoding comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        sns.boxplot(data=results_df, x='encoding', y='accuracy', ax=axes[0,0])
        axes[0,0].set_title('Accuracy by Encoding Strategy')
        axes[0,0].set_ylabel('Accuracy')
        
        # F1 score comparison
        sns.boxplot(data=results_df, x='encoding', y='f1_score', ax=axes[0,1])
        axes[0,1].set_title('F1 Score by Encoding Strategy')
        axes[0,1].set_ylabel('F1 Score')
        
        # Execution time comparison
        sns.boxplot(data=results_df, x='encoding', y='execution_time', ax=axes[1,0])
        axes[1,0].set_title('Execution Time by Encoding Strategy')
        axes[1,0].set_ylabel('Time (seconds)')
        
        # Qubit requirements
        qubit_data = results_df.groupby('encoding')['n_qubits'].first()
        axes[1,1].bar(qubit_data.index, qubit_data.values)
        axes[1,1].set_title('Qubit Requirements')
        axes[1,1].set_ylabel('Number of Qubits')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'encoding_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_scaling_analysis(self, results_df: pd.DataFrame):
        """Create scaling analysis visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance vs Feature Dimension
        for encoding in results_df['encoding'].unique():
            encoding_data = results_df[results_df['encoding'] == encoding]
            grouped = encoding_data.groupby('feature_dim')['accuracy'].agg(['mean', 'std'])
            
            axes[0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                           marker='o', label=encoding, capsize=5)
        
        axes[0].set_xlabel('Feature Dimension')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy vs Feature Dimension')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Qubits vs Feature Dimension
        for encoding in results_df['encoding'].unique():
            encoding_data = results_df[results_df['encoding'] == encoding]
            grouped = encoding_data.groupby('feature_dim')['n_qubits'].first()
            
            axes[1].plot(grouped.index, grouped.values, marker='s', label=encoding)
        
        axes[1].set_xlabel('Feature Dimension')
        axes[1].set_ylabel('Number of Qubits')
        axes[1].set_title('Qubit Requirements vs Feature Dimension')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'scaling_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_noise_analysis(self, results_df: pd.DataFrame):
        """Create noise sensitivity visualization"""
        if 'noise_level' not in results_df.columns:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs Noise Level
        for encoding in results_df['encoding'].unique():
            encoding_data = results_df[results_df['encoding'] == encoding]
            grouped = encoding_data.groupby('noise_level')['accuracy'].agg(['mean', 'std'])
            
            axes[0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           marker='o', label=encoding, capsize=5)
        
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Accuracy') 
        axes[0].set_title('Accuracy vs Noise Level')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Relative Performance Degradation
        for encoding in results_df['encoding'].unique():
            encoding_data = results_df[results_df['encoding'] == encoding]
            grouped = encoding_data.groupby('noise_level')['accuracy'].mean()
            baseline = grouped.iloc[0]  # First (lowest noise) as baseline
            
            relative_performance = grouped / baseline
            axes[1].plot(grouped.index, relative_performance, marker='s', label=encoding)
        
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('Relative Performance')
        axes[1].set_title('Performance Degradation with Noise')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'noise_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = QuantumMLAnalyzer('results')
    
    print("📊 QUANTUM ML ANALYSIS FRAMEWORK")
    print("=" * 40)
    print("Available analysis functions:")
    print("  • analyze_encoding_comparison()")
    print("  • analyze_scaling_behavior()")  
    print("  • analyze_noise_sensitivity()")
    print("  • generate_comprehensive_report()")
    print("\nFramework ready for analyzing experiment results!")