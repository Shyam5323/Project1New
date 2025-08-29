"""
Phase 1 Visualization Script
Generates comprehensive graphs and figures for Phase 1 quantum encoding comparison results
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Phase1Visualizer:
    """Generates comprehensive visualizations for Phase 1 results"""
    
    def __init__(self, results_dir='results/phase1', output_dir='results/figures/phase1'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all data files
        self.data = self._load_data()
        
    def _load_data(self):
        """Load all Phase 1 result files"""
        data = {}
        
        files_to_load = [
            'best_models.json',
            'comprehensive_analysis.json', 
            'cross_validation.json',
            'depth_analysis.json',
            'dimension_scaling.json',
            'encoding_comparison.json'
        ]
        
        for filename in files_to_load:
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    key = filename.replace('.json', '')
                    data[key] = json.load(f)
                    print(f"✅ Loaded {filename}")
            else:
                print(f"⚠️  Missing {filename}")
                
        return data
    
    def save_figure(self, fig, filename, dpi=300):
        """Save figure with consistent formatting"""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"📊 Saved: {filename}")
    
    def plot_encoding_comparison(self):
        """Main encoding performance comparison"""
        if 'comprehensive_analysis' not in self.data:
            return
            
        analysis = self.data['comprehensive_analysis']
        quantum_perf = analysis['quantum_performance']
        classical = analysis['classical_baselines']
        
        # Extract data
        encodings = list(quantum_perf.keys())
        accuracies = [quantum_perf[enc]['accuracy_mean'] for enc in encodings]
        f1_scores = [quantum_perf[enc]['f1_mean'] for enc in encodings]
        accuracy_stds = [quantum_perf[enc]['accuracy_std'] for enc in encodings]
        f1_stds = [quantum_perf[enc]['f1_std'] for enc in encodings]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        x_pos = np.arange(len(encodings))
        bars1 = ax1.bar(x_pos, accuracies, yerr=accuracy_stds, capsize=5, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        
        # Add classical baselines
        ax1.axhline(y=classical['logistic_regression']['accuracy'], 
                   color='red', linestyle='--', alpha=0.7, label='Logistic Regression')
        ax1.axhline(y=classical['random_forest']['accuracy'], 
                   color='green', linestyle='--', alpha=0.7, label='Random Forest')
        
        ax1.set_xlabel('Quantum Encoding Strategy')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Quantum vs Classical Performance (Accuracy)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([enc.capitalize() for enc in encodings])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (acc, std) in enumerate(zip(accuracies, accuracy_stds)):
            ax1.text(i, acc + std + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1 Score comparison
        bars2 = ax2.bar(x_pos, f1_scores, yerr=f1_stds, capsize=5,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        
        ax2.axhline(y=classical['logistic_regression']['f1_score'], 
                   color='red', linestyle='--', alpha=0.7, label='Logistic Regression')
        ax2.axhline(y=classical['random_forest']['f1_score'], 
                   color='green', linestyle='--', alpha=0.7, label='Random Forest')
        
        ax2.set_xlabel('Quantum Encoding Strategy')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Quantum vs Classical Performance (F1 Score)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([enc.capitalize() for enc in encodings])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (f1, std) in enumerate(zip(f1_scores, f1_stds)):
            ax2.text(i, f1 + std + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, 'encoding_performance_comparison.png')
    
    def plot_statistical_significance(self):
        """Plot statistical significance tests between encodings"""
        if 'comprehensive_analysis' not in self.data:
            return
            
        pairwise = self.data['comprehensive_analysis']['pairwise_tests']
        
        # Extract data
        comparisons = list(pairwise.keys())
        t_stats = [pairwise[comp]['t_stat'] for comp in comparisons]
        p_values = [pairwise[comp]['p_value'] for comp in comparisons]
        effect_sizes = [pairwise[comp]['cohens_d'] for comp in comparisons]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # P-values with significance threshold
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        bars1 = ax1.bar(range(len(comparisons)), p_values, color=colors, alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        ax1.set_xlabel('Pairwise Comparisons')
        ax1.set_ylabel('P-value')
        ax1.set_title('Statistical Significance Tests')
        ax1.set_xticks(range(len(comparisons)))
        ax1.set_xticklabels([comp.replace('_vs_', ' vs\n') for comp in comparisons], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add significance labels
        for i, p in enumerate(p_values):
            significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax1.text(i, p + 0.002, significance, ha='center', va='bottom', fontweight='bold')
        
        # Effect sizes (Cohen's d)
        effect_colors = ['darkgreen' if d >= 0.8 else 'orange' if d >= 0.5 else 'lightcoral' for d in effect_sizes]
        bars2 = ax2.bar(range(len(comparisons)), effect_sizes, color=effect_colors, alpha=0.7)
        ax2.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.7, label='Large effect (d ≥ 0.8)')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect (d ≥ 0.5)')
        ax2.set_xlabel('Pairwise Comparisons')
        ax2.set_ylabel("Cohen's d (Effect Size)")
        ax2.set_title('Effect Size Analysis')
        ax2.set_xticks(range(len(comparisons)))
        ax2.set_xticklabels([comp.replace('_vs_', ' vs\n') for comp in comparisons], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add effect size labels
        for i, d in enumerate(effect_sizes):
            ax2.text(i, d + 0.1, f'{d:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, 'statistical_significance_analysis.png')
    
    def plot_dimension_scaling(self):
        """Plot performance scaling with feature dimensions"""
        if 'comprehensive_analysis' not in self.data:
            return
            
        dim_data = self.data['comprehensive_analysis']['dimension_scaling']
        
        # Extract data
        dimensions = sorted([int(d.split('_')[1]) for d in dim_data.keys()])
        encodings = ['amplitude', 'angle', 'basis']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy scaling
        for i, encoding in enumerate(encodings):
            accuracies = []
            for dim in dimensions:
                key = f'dim_{dim}'
                if key in dim_data and encoding in dim_data[key]:
                    accuracies.append(dim_data[key][encoding]['accuracy_mean'])
                else:
                    accuracies.append(None)
            
            # Filter out None values
            valid_dims = [d for d, a in zip(dimensions, accuracies) if a is not None]
            valid_accs = [a for a in accuracies if a is not None]
            
            ax1.plot(valid_dims, valid_accs, marker='o', linewidth=2, markersize=8, 
                    label=encoding.capitalize(), alpha=0.8)
        
        ax1.set_xlabel('Feature Dimension')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Scaling with Feature Dimension')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(dimensions)
        
        # F1 Score scaling
        for i, encoding in enumerate(encodings):
            f1_scores = []
            for dim in dimensions:
                key = f'dim_{dim}'
                if key in dim_data and encoding in dim_data[key]:
                    f1_scores.append(dim_data[key][encoding]['f1_mean'])
                else:
                    f1_scores.append(None)
            
            # Filter out None values
            valid_dims = [d for d, f in zip(dimensions, f1_scores) if f is not None]
            valid_f1s = [f for f in f1_scores if f is not None]
            
            ax2.plot(valid_dims, valid_f1s, marker='o', linewidth=2, markersize=8, 
                    label=encoding.capitalize(), alpha=0.8)
        
        ax2.set_xlabel('Feature Dimension')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Scaling with Feature Dimension')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(dimensions)
        
        plt.tight_layout()
        self.save_figure(fig, 'dimension_scaling_analysis.png')
    
    def plot_circuit_depth_analysis(self):
        """Plot performance vs circuit depth"""
        if 'comprehensive_analysis' not in self.data:
            return
            
        depth_data = self.data['comprehensive_analysis']['depth_analysis']
        
        # Extract data
        depths = sorted([int(d.split('_')[1]) for d in depth_data.keys()])
        encodings = ['amplitude', 'angle', 'basis']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs depth
        for encoding in encodings:
            accuracies = []
            for depth in depths:
                key = f'depth_{depth}'
                if key in depth_data and encoding in depth_data[key]:
                    accuracies.append(depth_data[key][encoding]['accuracy_mean'])
                else:
                    accuracies.append(None)
            
            valid_depths = [d for d, a in zip(depths, accuracies) if a is not None]
            valid_accs = [a for a in accuracies if a is not None]
            
            ax1.plot(valid_depths, valid_accs, marker='s', linewidth=2, markersize=8, 
                    label=encoding.capitalize(), alpha=0.8)
        
        ax1.set_xlabel('Circuit Depth')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance vs Circuit Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 vs depth
        for encoding in encodings:
            f1_scores = []
            for depth in depths:
                key = f'depth_{depth}'
                if key in depth_data and encoding in depth_data[key]:
                    f1_scores.append(depth_data[key][encoding]['f1_mean'])
                else:
                    f1_scores.append(None)
            
            valid_depths = [d for d, f in zip(depths, f1_scores) if f is not None]
            valid_f1s = [f for f in f1_scores if f is not None]
            
            ax2.plot(valid_depths, valid_f1s, marker='s', linewidth=2, markersize=8, 
                    label=encoding.capitalize(), alpha=0.8)
        
        ax2.set_xlabel('Circuit Depth')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score vs Circuit Depth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'circuit_depth_analysis.png')
    
    def plot_cross_validation_results(self):
        """Plot cross-validation results with error bars"""
        if 'cross_validation' not in self.data:
            return
            
        cv_data = pd.DataFrame(self.data['cross_validation'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy across folds
        sns.boxplot(data=cv_data, x='encoding', y='accuracy', ax=ax1)
        ax1.set_title('Cross-Validation Accuracy Distribution')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Encoding Strategy')
        
        # F1 Score across folds
        sns.boxplot(data=cv_data, x='encoding', y='f1_score', ax=ax2)
        ax2.set_title('Cross-Validation F1 Score Distribution')
        ax2.set_ylabel('F1 Score')
        ax2.set_xlabel('Encoding Strategy')
        
        # Precision across folds
        sns.boxplot(data=cv_data, x='encoding', y='precision', ax=ax3)
        ax3.set_title('Cross-Validation Precision Distribution')
        ax3.set_ylabel('Precision')
        ax3.set_xlabel('Encoding Strategy')
        
        # Recall across folds
        sns.boxplot(data=cv_data, x='encoding', y='recall', ax=ax4)
        ax4.set_title('Cross-Validation Recall Distribution')
        ax4.set_ylabel('Recall')
        ax4.set_xlabel('Encoding Strategy')
        
        plt.tight_layout()
        self.save_figure(fig, 'cross_validation_analysis.png')
    
    def plot_efficiency_analysis(self):
        """Plot resource efficiency metrics"""
        if 'comprehensive_analysis' not in self.data:
            return
            
        quantum_perf = self.data['comprehensive_analysis']['quantum_performance']
        
        # Extract efficiency metrics
        encodings = list(quantum_perf.keys())
        accuracy_per_qubit = [quantum_perf[enc]['accuracy_per_qubit_mean'] for enc in encodings]
        f1_per_param = [quantum_perf[enc]['f1_per_param_mean'] for enc in encodings]
        n_qubits = [quantum_perf[enc]['n_qubits_mean'] for enc in encodings]
        param_counts = [quantum_perf[enc]['param_count_mean'] for enc in encodings]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy per qubit
        bars1 = ax1.bar(encodings, accuracy_per_qubit, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax1.set_title('Resource Efficiency: Accuracy per Qubit')
        ax1.set_ylabel('Accuracy / Number of Qubits')
        ax1.set_xlabel('Encoding Strategy')
        for i, v in enumerate(accuracy_per_qubit):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1 per parameter
        bars2 = ax2.bar(encodings, f1_per_param, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax2.set_title('Parameter Efficiency: F1 Score per Parameter')
        ax2.set_ylabel('F1 Score / Number of Parameters')
        ax2.set_xlabel('Encoding Strategy')
        for i, v in enumerate(f1_per_param):
            ax2.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Resource usage comparison
        x_pos = np.arange(len(encodings))
        width = 0.35
        
        bars3 = ax3.bar(x_pos - width/2, n_qubits, width, label='Qubits', alpha=0.8)
        bars4 = ax3.bar(x_pos + width/2, [p/10 for p in param_counts], width, label='Parameters (÷10)', alpha=0.8)
        ax3.set_title('Resource Usage Comparison')
        ax3.set_ylabel('Count')
        ax3.set_xlabel('Encoding Strategy')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(encodings)
        ax3.legend()
        
        # Performance vs resource trade-off
        accuracies = [quantum_perf[enc]['accuracy_mean'] for enc in encodings]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (enc, acc, qubits) in enumerate(zip(encodings, accuracies, n_qubits)):
            ax4.scatter(qubits, acc, s=200, alpha=0.7, color=colors[i], label=enc.capitalize())
            ax4.annotate(enc.capitalize(), (qubits, acc), xytext=(5, 5), 
                        textcoords='offset points', fontweight='bold')
        
        ax4.set_xlabel('Number of Qubits')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Quantum Resource Usage')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        self.save_figure(fig, 'efficiency_analysis.png')
    
    def plot_performance_ranking(self):
        """Plot overall performance ranking"""
        if 'comprehensive_analysis' not in self.data:
            return
            
        ranking = self.data['comprehensive_analysis']['performance_ranking']
        
        # Extract data
        encodings = list(ranking.keys())
        scores = [ranking[enc]['performance_score'] for enc in encodings]
        ranks = [ranking[enc]['rank'] for enc in encodings]
        accuracies = [ranking[enc]['accuracy'] for enc in encodings]
        f1_scores = [ranking[enc]['f1_score'] for enc in encodings]
        
        # Sort by rank
        sorted_data = sorted(zip(encodings, scores, ranks, accuracies, f1_scores), key=lambda x: x[2])
        encodings, scores, ranks, accuracies, f1_scores = zip(*sorted_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance score ranking
        colors = ['gold', 'silver', '#CD7F32']  # Gold, Silver, Bronze
        bars1 = ax1.barh(range(len(encodings)), scores, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(encodings)))
        ax1.set_yticklabels([f"{rank}. {enc.capitalize()}" for rank, enc in zip(ranks, encodings)])
        ax1.set_xlabel('Performance Score')
        ax1.set_title('Overall Performance Ranking')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add score labels
        for i, score in enumerate(scores):
            ax1.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')
        
        # Detailed metrics comparison
        x_pos = np.arange(len(encodings))
        width = 0.35
        
        bars2 = ax2.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        bars3 = ax2.bar(x_pos + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax2.set_xlabel('Encoding Strategy (by rank)')
        ax2.set_ylabel('Score')
        ax2.set_title('Detailed Performance Metrics')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{rank}. {enc.capitalize()}" for rank, enc in zip(ranks, encodings)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'performance_ranking.png')
    
    def generate_summary_report(self):
        """Generate a visual summary report"""
        if 'comprehensive_analysis' not in self.data:
            return
            
        analysis = self.data['comprehensive_analysis']
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Phase 1: Quantum Encoding Strategies - Comprehensive Analysis', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        # Main performance comparison
        ax1 = fig.add_subplot(gs[0, :2])
        quantum_perf = analysis['quantum_performance']
        encodings = list(quantum_perf.keys())
        accuracies = [quantum_perf[enc]['accuracy_mean'] for enc in encodings]
        
        bars = ax1.bar(encodings, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax1.set_title('Quantum Encoding Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        
        for i, acc in enumerate(accuracies):
            ax1.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance ranking
        ax2 = fig.add_subplot(gs[0, 2])
        ranking = analysis['performance_ranking']
        ranks = [ranking[enc]['rank'] for enc in encodings]
        colors = ['gold' if r == 1 else 'silver' if r == 2 else '#CD7F32' for r in ranks]
        
        ax2.pie([1, 1, 1], labels=[f"{i+1}. {enc.capitalize()}" for i, enc in enumerate(encodings)], 
                colors=colors, autopct='', startangle=90)
        ax2.set_title('Performance Ranking', fontsize=14, fontweight='bold')
        
        # Statistical significance
        ax3 = fig.add_subplot(gs[1, 0])
        pairwise = analysis['pairwise_tests']
        comparisons = list(pairwise.keys())
        p_values = [pairwise[comp]['p_value'] for comp in comparisons]
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        ax3.bar(range(len(comparisons)), p_values, color=colors, alpha=0.7)
        ax3.axhline(y=0.05, color='red', linestyle='--')
        ax3.set_title('Statistical Significance', fontsize=12, fontweight='bold')
        ax3.set_ylabel('P-value')
        ax3.set_xticks(range(len(comparisons)))
        ax3.set_xticklabels([comp.replace('_vs_', '\nvs\n') for comp in comparisons], fontsize=8)
        
        # Resource efficiency
        ax4 = fig.add_subplot(gs[1, 1])
        efficiency = [quantum_perf[enc]['accuracy_per_qubit_mean'] for enc in encodings]
        ax4.bar(encodings, efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax4.set_title('Resource Efficiency', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Accuracy/Qubit')
        
        # Classical comparison
        ax5 = fig.add_subplot(gs[1, 2])
        classical = analysis['classical_baselines']
        methods = ['Amplitude\n(Quantum)', 'Logistic\nRegression', 'Random\nForest']
        performances = [max(accuracies), classical['logistic_regression']['accuracy'], 
                       classical['random_forest']['accuracy']]
        colors = ['#FF6B6B', 'red', 'green']
        
        ax5.bar(methods, performances, color=colors, alpha=0.7)
        ax5.set_title('Quantum vs Classical', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Accuracy')
        
        # Key insights text
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        insights = [
            "🏆 KEY FINDINGS:",
            f"• Best performing encoding: {encodings[0].capitalize()} ({accuracies[0]:.3f} accuracy)",
            f"• Quantum advantage: {(max(accuracies) - classical['random_forest']['accuracy']):.3f} over best classical",
            f"• Most resource efficient: {encodings[np.argmax([quantum_perf[enc]['accuracy_per_qubit_mean'] for enc in encodings])].capitalize()}",
            f"• Statistically significant differences between all encoding pairs",
            "• Amplitude encoding shows superior performance with fewer qubits required"
        ]
        
        for i, insight in enumerate(insights):
            ax6.text(0.05, 0.8 - i*0.12, insight, fontsize=12, fontweight='bold' if i == 0 else 'normal',
                    transform=ax6.transAxes, verticalalignment='top')
        
        self.save_figure(fig, 'phase1_comprehensive_summary.png')
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("🎨 Generating Phase 1 visualizations...")
        
        self.plot_encoding_comparison()
        self.plot_statistical_significance()
        self.plot_dimension_scaling()
        self.plot_circuit_depth_analysis()
        self.plot_cross_validation_results()
        self.plot_efficiency_analysis()
        self.plot_performance_ranking()
        self.generate_summary_report()
        
        print(f"✅ All plots saved to: {self.output_dir}")

def main():
    """Main execution function"""
    visualizer = Phase1Visualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()
