"""
Thesis Results Generator - Comprehensive Analysis and Visualization
Parses comparison results and generates publication-ready charts and tables

SAVE AS: tests/analysis/generate_thesis_results.py
RUN: python tests/analysis/generate_thesis_results.py modular_comparison_simple_medium_complex_*.json

Creates:
- thesis_results/charts/ - All visualizations (PNG, PDF)
- thesis_results/tables/ - LaTeX and CSV tables
- thesis_results/summary.txt - Executive summary
- thesis_results/analysis.json - Detailed statistics
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create output directories
OUTPUT_DIR = Path("thesis_results")
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"

def setup_output_dirs():
    """Create output directory structure"""
    for dir_path in [OUTPUT_DIR, CHARTS_DIR, TABLES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directories created in: {OUTPUT_DIR.absolute()}")

def load_results(json_file):
    """Load and parse results JSON"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def calculate_metrics(results, filter_dict):
    """Calculate accuracy metrics with filters"""
    filtered = [r for r in results if all(r.get(k) == v for k, v in filter_dict.items())]
    
    if not filtered:
        return None
    
    # Separate by model
    metrics_by_model = {}
    
    # VAE metrics
    vae_results = [r for r in filtered if r.get('vae_available') and r.get('vae_correct') is not None]
    if vae_results:
        vae_correct = sum(1 for r in vae_results if r['vae_correct'])
        vae_acc = vae_correct / len(vae_results) * 100
        
        # Calculate precision/recall
        tp = sum(1 for r in vae_results if not r['expected_normal'] and not r['vae_predicted_normal'])
        tn = sum(1 for r in vae_results if r['expected_normal'] and r['vae_predicted_normal'])
        fp = sum(1 for r in vae_results if r['expected_normal'] and not r['vae_predicted_normal'])
        fn = sum(1 for r in vae_results if not r['expected_normal'] and r['vae_predicted_normal'])
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_by_model['VAE'] = {
            'count': len(vae_results),
            'accuracy': vae_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # LLM metrics
    for model_id in ['gpt-4o', 'claude']:
        model_name = 'GPT-4o' if model_id == 'gpt-4o' else 'Claude Sonnet 3.5'
        llm_results = [r for r in filtered if r.get('llm_model') == model_id and 
                       r.get('llm_available') and r.get('llm_correct') is not None]
        
        if llm_results:
            llm_correct = sum(1 for r in llm_results if r['llm_correct'])
            llm_acc = llm_correct / len(llm_results) * 100
            
            tp = sum(1 for r in llm_results if not r['expected_normal'] and not r['llm_predicted_normal'])
            tn = sum(1 for r in llm_results if r['expected_normal'] and r['llm_predicted_normal'])
            fp = sum(1 for r in llm_results if r['expected_normal'] and not r['llm_predicted_normal'])
            fn = sum(1 for r in llm_results if not r['expected_normal'] and r['llm_predicted_normal'])
            
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_by_model[model_name] = {
                'count': len(llm_results),
                'accuracy': llm_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    return metrics_by_model

def generate_complexity_comparison_chart(data):
    """Generate main complexity comparison chart"""
    results = data['detailed_results']
    
    # Calculate metrics by complexity
    complexities = ['simple', 'medium', 'complex']
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    
    accuracy_data = {model: [] for model in models}
    
    for complexity in complexities:
        metrics = calculate_metrics(results, {'complexity': complexity})
        if metrics:
            for model in models:
                if model in metrics:
                    accuracy_data[model].append(metrics[model]['accuracy'])
                else:
                    accuracy_data[model].append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(complexities))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (model, color) in enumerate(zip(models, colors)):
        offset = width * (idx - 1)
        bars = ax.bar(x + offset, accuracy_data[model], width, label=model, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Circuit Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Hardware Verification Accuracy by Circuit Complexity', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Simple\n(26 circuits)', 'Medium\n(48 circuits)', 'Complex\n(12 circuits)'])
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(CHARTS_DIR / 'complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(CHARTS_DIR / 'complexity_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: complexity_comparison chart")

def generate_trend_line_chart(data):
    """Generate line chart showing performance trends"""
    results = data['detailed_results']
    
    complexities = ['simple', 'medium', 'complex']
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'VAE': '#FF6B6B', 'GPT-4o': '#4ECDC4', 'Claude Sonnet 3.5': '#45B7D1'}
    markers = {'VAE': 'o', 'GPT-4o': 's', 'Claude Sonnet 3.5': '^'}
    
    for model in models:
        accuracies = []
        for complexity in complexities:
            metrics = calculate_metrics(results, {'complexity': complexity})
            if metrics and model in metrics:
                accuracies.append(metrics[model]['accuracy'])
            else:
                accuracies.append(0)
        
        ax.plot(complexities, accuracies, marker=markers[model], 
                label=model, linewidth=3, markersize=12, 
                color=colors[model])
        
        # Annotate points
        for x, y in zip(complexities, accuracies):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Circuit Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Trends Across Complexity Levels', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.set_ylim(50, 90)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'trend_lines.png', dpi=300, bbox_inches='tight')
    plt.savefig(CHARTS_DIR / 'trend_lines.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: trend_lines chart")

def generate_precision_recall_chart(data):
    """Generate precision vs recall scatter plot"""
    results = data['detailed_results']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    complexities = ['simple', 'medium', 'complex']
    
    colors = {'simple': '#FF6B6B', 'medium': '#4ECDC4', 'complex': '#45B7D1'}
    markers = {'VAE': 'o', 'GPT-4o': 's', 'Claude Sonnet 3.5': '^'}
    
    for model in models:
        for complexity in complexities:
            metrics = calculate_metrics(results, {'complexity': complexity})
            if metrics and model in metrics:
                m = metrics[model]
                ax.scatter(m['recall'], m['precision'], 
                          s=300, marker=markers[model], 
                          color=colors[complexity], alpha=0.7,
                          edgecolors='black', linewidth=2,
                          label=f'{model} - {complexity}' if model == 'VAE' else '')
    
    ax.set_xlabel('Recall (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision (%)', fontsize=14, fontweight='bold')
    ax.set_title('Precision vs Recall by Model and Complexity', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='VAE', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='GPT-4o', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markersize=10, label='Claude', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
               markersize=10, label='Simple', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
               markersize=10, label='Medium', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', 
               markersize=10, label='Complex', markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'precision_recall.png', dpi=300, bbox_inches='tight')
    plt.savefig(CHARTS_DIR / 'precision_recall.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: precision_recall chart")

def generate_f1_score_chart(data):
    """Generate F1 score comparison"""
    results = data['detailed_results']
    
    complexities = ['simple', 'medium', 'complex']
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    
    f1_data = {model: [] for model in models}
    
    for complexity in complexities:
        metrics = calculate_metrics(results, {'complexity': complexity})
        if metrics:
            for model in models:
                if model in metrics:
                    f1_data[model].append(metrics[model]['f1'])
                else:
                    f1_data[model].append(0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(complexities))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (model, color) in enumerate(zip(models, colors)):
        offset = width * (idx - 1)
        bars = ax.bar(x + offset, f1_data[model], width, label=model, color=color, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Circuit Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title('F1-Score Comparison Across Complexity Levels', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Simple', 'Medium', 'Complex'])
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'f1_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig(CHARTS_DIR / 'f1_scores.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: f1_scores chart")

def generate_category_breakdown(data):
    """Generate category-wise performance breakdown"""
    results = data['detailed_results']
    
    # Group by category
    categories = set(r['category'] for r in results)
    
    category_metrics = {}
    for category in sorted(categories):
        metrics = calculate_metrics(results, {'category': category})
        if metrics:
            category_metrics[category] = metrics
    
    # Create heatmap data
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    heatmap_data = []
    category_labels = []
    
    for category in sorted(category_metrics.keys()):
        row = []
        for model in models:
            if model in category_metrics[category]:
                row.append(category_metrics[category][model]['accuracy'])
            else:
                row.append(0)
        heatmap_data.append(row)
        category_labels.append(category.replace(' - ', '\n'))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(category_labels)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(category_labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(category_labels)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Accuracy by Circuit Category', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'category_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(CHARTS_DIR / 'category_breakdown.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated: category_breakdown chart")

def generate_latex_table(data):
    """Generate LaTeX formatted table"""
    results = data['detailed_results']
    
    complexities = ['simple', 'medium', 'complex']
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Hardware Verification Accuracy by Complexity Level}")
    latex.append("\\label{tab:complexity_results}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model} & \\textbf{Simple} & \\textbf{Medium} & \\textbf{Complex} \\\\")
    latex.append("\\midrule")
    
    for model in models:
        row = [model]
        for complexity in complexities:
            metrics = calculate_metrics(results, {'complexity': complexity})
            if metrics and model in metrics:
                acc = metrics[model]['accuracy']
                row.append(f"{acc:.1f}\\%")
            else:
                row.append("--")
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(TABLES_DIR / 'complexity_table.tex', 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"✅ Generated: LaTeX table")

def generate_csv_tables(data):
    """Generate CSV tables for analysis"""
    results = data['detailed_results']
    
    # Main results table
    complexities = ['simple', 'medium', 'complex']
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    
    rows = []
    for model in models:
        for complexity in complexities:
            metrics = calculate_metrics(results, {'complexity': complexity})
            if metrics and model in metrics:
                m = metrics[model]
                rows.append({
                    'Model': model,
                    'Complexity': complexity,
                    'Accuracy': f"{m['accuracy']:.2f}",
                    'Precision': f"{m['precision']:.2f}",
                    'Recall': f"{m['recall']:.2f}",
                    'F1-Score': f"{m['f1']:.2f}",
                    'Count': m['count']
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'main_results.csv', index=False)
    
    print(f"✅ Generated: CSV tables")

def generate_summary_report(data):
    """Generate executive summary"""
    results = data['detailed_results']
    
    summary = []
    summary.append("=" * 80)
    summary.append("THESIS RESULTS SUMMARY")
    summary.append("Hardware Verification Using LLMs vs Traditional ML")
    summary.append("=" * 80)
    summary.append("")
    
    # Overall statistics
    total_circuits = len(set(r['circuit_name'] for r in results))
    total_tests = len(results)
    
    summary.append(f"Total Circuits Tested: {total_circuits}")
    summary.append(f"Total Tests Conducted: {total_tests}")
    summary.append(f"Test Date: {data['summary'].get('test_timestamp', 'N/A')}")
    summary.append("")
    
    # Performance by complexity
    summary.append("PERFORMANCE BY COMPLEXITY LEVEL")
    summary.append("-" * 80)
    
    complexities = ['simple', 'medium', 'complex']
    models = ['VAE', 'GPT-4o', 'Claude Sonnet 3.5']
    
    for complexity in complexities:
        summary.append(f"\n{complexity.upper()} CIRCUITS:")
        metrics = calculate_metrics(results, {'complexity': complexity})
        if metrics:
            for model in models:
                if model in metrics:
                    m = metrics[model]
                    summary.append(f"  {model:20s}: {m['accuracy']:5.1f}% "
                                 f"(P:{m['precision']:5.1f}% R:{m['recall']:5.1f}% F1:{m['f1']:5.1f})")
    
    # Key findings
    summary.append("\n" + "=" * 80)
    summary.append("KEY FINDINGS")
    summary.append("=" * 80)
    
    # Calculate trends
    claude_metrics = {}
    for complexity in complexities:
        metrics = calculate_metrics(results, {'complexity': complexity})
        if metrics and 'Claude Sonnet 3.5' in metrics:
            claude_metrics[complexity] = metrics['Claude Sonnet 3.5']['accuracy']
    
    if 'simple' in claude_metrics and 'complex' in claude_metrics:
        claude_change = claude_metrics['complex'] - claude_metrics['simple']
        summary.append(f"\n1. Claude Sonnet 3.5 Performance Trend:")
        summary.append(f"   - Simple: {claude_metrics.get('simple', 0):.1f}%")
        summary.append(f"   - Medium: {claude_metrics.get('medium', 0):.1f}%")
        summary.append(f"   - Complex: {claude_metrics.get('complex', 0):.1f}%")
        summary.append(f"   - Overall Change: {claude_change:+.1f}% ({'IMPROVEMENT' if claude_change > 0 else 'DECLINE'})")
    
    summary.append("\n2. Model Rankings by Complexity:")
    for complexity in complexities:
        metrics = calculate_metrics(results, {'complexity': complexity})
        if metrics:
            sorted_models = sorted(metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            summary.append(f"   {complexity.capitalize():8s}: {sorted_models[0][0]} ({sorted_models[0][1]['accuracy']:.1f}%)")
    
    summary.append("\n3. Best Overall Model:")
    overall_metrics = calculate_metrics(results, {})
    if overall_metrics:
        best_model = max(overall_metrics.items(), key=lambda x: x[1]['accuracy'])
        summary.append(f"   {best_model[0]}: {best_model[1]['accuracy']:.1f}% accuracy")
    
    summary.append("\n" + "=" * 80)
    
    with open(OUTPUT_DIR / 'summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"✅ Generated: summary report")
    return '\n'.join(summary)

def generate_detailed_analysis(data):
    """Generate detailed JSON analysis"""
    results = data['detailed_results']
    
    analysis = {
        'metadata': {
            'total_circuits': len(set(r['circuit_name'] for r in results)),
            'total_tests': len(results),
            'test_date': data['summary'].get('test_timestamp', 'N/A'),
            'models_tested': data['summary'].get('models_tested', [])
        },
        'by_complexity': {},
        'by_category': {},
        'overall': {}
    }
    
    # By complexity
    for complexity in ['simple', 'medium', 'complex']:
        metrics = calculate_metrics(results, {'complexity': complexity})
        if metrics:
            analysis['by_complexity'][complexity] = metrics
    
    # By category
    categories = set(r['category'] for r in results)
    for category in categories:
        metrics = calculate_metrics(results, {'category': category})
        if metrics:
            analysis['by_category'][category] = metrics
    
    # Overall
    overall_metrics = calculate_metrics(results, {})
    if overall_metrics:
        analysis['overall'] = overall_metrics
    
    with open(OUTPUT_DIR / 'analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Generated: detailed analysis JSON")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_thesis_results.py <json_file>")
        print("\nExample:")
        print("  python tests/analysis/generate_thesis_results.py modular_comparison_simple_medium_complex_20251016_014108.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"❌ Error: File not found: {json_file}")
        sys.exit(1)
    
    print("=" * 80)
    print("THESIS RESULTS GENERATOR")
    print("=" * 80)
    print(f"Input file: {json_file}\n")
    
    # Setup
    setup_output_dirs()
    
    # Load data
    print("\n📖 Loading results...")
    data = load_results(json_file)
    print(f"✅ Loaded {len(data['detailed_results'])} test results")
    
    # Generate all outputs
    print("\n📊 Generating visualizations...")
    generate_complexity_comparison_chart(data)
    generate_trend_line_chart(data)
    generate_precision_recall_chart(data)
    generate_f1_score_chart(data)
    generate_category_breakdown(data)
    
    print("\n📋 Generating tables...")
    generate_latex_table(data)
    generate_csv_tables(data)
    
    print("\n📝 Generating reports...")
    summary_text = generate_summary_report(data)
    generate_detailed_analysis(data)
    
    print("\n" + "=" * 80)
    print("✅ ALL RESULTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📁 Output location: {OUTPUT_DIR.absolute()}")
    print(f"   📊 Charts (PNG/PDF): {CHARTS_DIR.absolute()}")
    print(f"   📋 Tables (LaTeX/CSV): {TABLES_DIR.absolute()}")
    print(f"   📄 Summary: {OUTPUT_DIR / 'summary.txt'}")
    print(f"   📄 Analysis: {OUTPUT_DIR / 'analysis.json'}")
    
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(summary_text.split("KEY FINDINGS")[1] if "KEY FINDINGS" in summary_text else "")

if __name__ == "__main__":
    main()