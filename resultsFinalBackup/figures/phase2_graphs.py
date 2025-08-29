import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(output_dir, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(os.path.join(output_dir, name), bbox_inches='tight')
    plt.close(fig)

# 1. Accuracy vs Noise Level for each encoding/noise type
def plot_noise_progression():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase2/research_noise_progression.csv')
    df = pd.read_csv(csv_path)
    for encoding in df['encoding'].unique():
        fig, ax = plt.subplots()
        subset = df[df['encoding'] == encoding]
        sns.lineplot(data=subset, x='noise_level', y='accuracy_mean', hue='noise_type', marker='o', ax=ax)
        ax.set_title(f'Accuracy vs Noise Level ({encoding})')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Noise Level')
        save_fig(fig, f'accuracy_vs_noise_{encoding}.png')

        fig, ax = plt.subplots()
        sns.lineplot(data=subset, x='noise_level', y='f1_mean', hue='noise_type', marker='o', ax=ax)
        ax.set_title(f'F1 Score vs Noise Level ({encoding})')
        ax.set_ylabel('F1 Score')
        ax.set_xlabel('Noise Level')
        save_fig(fig, f'f1_vs_noise_{encoding}.png')

# 2. Encoding comparison for each error type (from error_type_comparison.csv)
def plot_error_type_comparison():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase2/error_type_comparison.csv')
    df = pd.read_csv(csv_path, header=[0,1,2])
    # Extract error types from the first column (third-level header 'encoding' under unnamed parents)
    first_col_key = None
    for col in df.columns:
        if col[2] == 'encoding':
            first_col_key = col
            break
    if first_col_key is None:
        raise ValueError('Could not locate the error type column ("encoding" third-level header). Columns found: ' + str(df.columns))
    error_types = df[first_col_key].values
    # Ensure plain python list of labels for matplotlib (avoid ExtensionArray issues)
    error_types_list = [str(x) for x in error_types]
    # Encodings are the third-level entries under mean/accuracy
    encodings = sorted({c[2] for c in df.columns if c[0] == 'mean' and c[1] == 'accuracy'})

    # Accuracy plot
    fig, ax = plt.subplots()
    for encoding in encodings:
        y_vals = df[('mean','accuracy',encoding)].to_numpy()
        ax.plot(error_types_list, y_vals, marker='o', label=encoding)
    ax.set_title('Encoding Accuracy Comparison by Error Type')
    ax.set_xlabel('Error Type')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    save_fig(fig, 'encoding_accuracy_by_error_type.png')

    # F1 Score plot
    fig, ax = plt.subplots()
    for encoding in encodings:
        y_vals = df[('mean','f1_score',encoding)].to_numpy()
        ax.plot(error_types_list, y_vals, marker='o', label=encoding)
    ax.set_title('Encoding F1 Score Comparison by Error Type')
    ax.set_xlabel('Error Type')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    save_fig(fig, 'encoding_f1_by_error_type.png')

# 3. Device comparison (from device_comparison.csv)
def plot_device_comparison():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase2/device_comparison.csv')
    df = pd.read_csv(csv_path, header=[0,1,2])
    # Infer encodings from mean/accuracy third-level labels
    encodings = sorted({c[2] for c in df.columns if c[0] == 'mean' and c[1] == 'accuracy'})
    if not encodings:
        raise ValueError('Could not infer encodings from columns. Columns: ' + str(df.columns))
    # Choose first row (assumed single backend or first backend record)
    first_row = df.iloc[0]
    backend_label = None
    # Try to find a column whose third level suggests backend identifier
    for col in df.columns:
        if col[2].lower() in ('backend','device','system'):
            backend_label = first_row[col]
            break
    if backend_label is None:
        backend_label = 'Backend'

    acc_values = [first_row[('mean','accuracy',enc)] for enc in encodings]
    fig, ax = plt.subplots()
    ax.bar(encodings, acc_values)
    ax.set_title(f'Device Accuracy Comparison ({backend_label})')
    ax.set_ylabel('Accuracy')
    save_fig(fig, 'device_accuracy_comparison.png')

    f1_values = [first_row[('mean','f1_score',enc)] for enc in encodings]
    fig, ax = plt.subplots()
    ax.bar(encodings, f1_values)
    ax.set_title(f'Device F1 Score Comparison ({backend_label})')
    ax.set_ylabel('F1 Score')
    save_fig(fig, 'device_f1_comparison.png')

# 4. Coherence analysis (from coherence_analysis.csv)
def plot_coherence_analysis():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase2/coherence_analysis.csv')
    df = pd.read_csv(csv_path, header=[0,1,2])
    # Find t1_time column dynamically; fallback: use first column if patterns absent
    t1_col = next((c for c in df.columns if c[2].lower() in ('t1_time','t1','t1time')), None)
    if t1_col is not None:
        t1_times_list = list(df[t1_col].values)
    else:
        # Fallback: treat the first column (likely labelled 'encoding' due to csv structure) as t1 values
        first_col = df.columns[0]
        t1_times_list = list(df[first_col].values)
    encodings = sorted({c[2] for c in df.columns if c[0]=='mean' and c[1]=='accuracy'})
    if not encodings:
        raise ValueError('Could not infer encodings for coherence plot. Columns: ' + str(df.columns))

    fig, ax = plt.subplots()
    for encoding in encodings:
        y_vals = df[('mean','accuracy',encoding)].to_numpy()
        ax.plot(t1_times_list, y_vals, marker='o', label=encoding)
    ax.set_title('Coherence Study: Accuracy vs T1 Time')
    ax.set_xlabel('T1 Time')
    ax.set_ylabel('Accuracy')
    ax.legend()
    save_fig(fig, 'coherence_accuracy_vs_t1.png')

    fig, ax = plt.subplots()
    for encoding in encodings:
        y_vals = df[('mean','f1_score',encoding)].to_numpy()
        ax.plot(t1_times_list, y_vals, marker='o', label=encoding)
    ax.set_title('Coherence Study: F1 Score vs T1 Time')
    ax.set_xlabel('T1 Time')
    ax.set_ylabel('F1 Score')
    ax.legend()
    save_fig(fig, 'coherence_f1_vs_t1.png')

# 5. Resilience analysis (from phase2_noise_summary.json)
def plot_resilience_analysis():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase2/phase2_noise_summary.json')
    with open(json_path, 'r') as f:
        summary = json.load(f)
    resilience = summary['resilience_analysis']
    fig, ax = plt.subplots()
    noise_types = list(resilience.keys())
    performances = [resilience[n]['performance'] for n in noise_types]
    encodings = [resilience[n]['best_encoding'] for n in noise_types]
    ax.bar(noise_types, performances, color='skyblue')
    for i, encoding in enumerate(encodings):
        ax.text(i, performances[i], encoding, ha='center', va='bottom')
    ax.set_title('Best Encoding per Noise Type (Resilience Analysis)')
    ax.set_ylabel('Performance')
    save_fig(fig, 'resilience_best_encoding.png')

# 6. Parametric noise: accuracy mean/std vs encoding/noise type (from research_statistical_comparison.csv)
def plot_parametric_noise_stats():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase2/research_statistical_comparison.csv')
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='encoding', y='accuracy_mean', hue='noise_type', ax=ax)
    ax.set_title('Parametric Noise: Accuracy Mean by Encoding/Noise Type')
    save_fig(fig, 'parametric_noise_accuracy_mean.png')

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='encoding', y='accuracy_std', hue='noise_type', ax=ax)
    ax.set_title('Parametric Noise: Accuracy Std by Encoding/Noise Type')
    save_fig(fig, 'parametric_noise_accuracy_std.png')

if __name__ == '__main__':
    plot_noise_progression()
    plot_error_type_comparison()
    plot_device_comparison()
    plot_coherence_analysis()
    plot_resilience_analysis()
    plot_parametric_noise_stats()
