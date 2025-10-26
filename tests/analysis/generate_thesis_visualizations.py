"""
Thesis Visualization Generator for Multi-Modal Test Results
Creates publication-quality charts and graphs for thesis

SAVE AS: tests/analysis/generate_thesis_visualizations.py
RUN: python tests/analysis/generate_thesis_visualizations.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
INPUT_DIR = ROOT_DIR / "thesis_generation_results" / "multimodal_mermaid"
OUTPUT_DIR = INPUT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

def load_aggregated_data():
    """Load the most recent aggregated summary"""
    files = sorted(INPUT_DIR.glob("aggregated_summary_*.json"))
    if not files:
        print("❌ No aggregated summary found! Run the batch test first.")
        return None
    
    latest = files[-1]
    print(f"📂 Loading: {latest.name}")
    
    with open(latest, 'r') as f:
        return json.load(f)

def load_all_detailed_results():
    """Load all individual test results"""
    files = sorted(INPUT_DIR.glob("three_way_results_*.json"))
    if not files:
        print("❌ No result files found!")
        return []
    
    # Get last 5 files
    recent_files = files[-5:]
    all_results = []
    
    for file in recent_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_results.extend(data['results'])
    
    return all_results

def plot_1_overall_winner_comparison(data):
    """Bar chart: Overall wins across all runs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
    wins = [
        data['overall_stats']['prompt_only_wins'],
        data['overall_stats']['mermaid_only_wins'],
        data['overall_stats']['combined_wins']
    ]
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(approaches, wins, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Number of Circuit Wins', fontweight='bold')
    ax.set_xlabel('Approach', fontweight='bold')
    ax.set_title(f'Overall Winner Comparison Across {data["num_runs"]} Runs\n(Total: {data["total_tests"]} tests)', 
                 fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(wins) * 1.15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_overall_winner_comparison.png', bbox_inches='tight')
    print("✅ Created: 1_overall_winner_comparison.png")
    plt.close()

def plot_2_correctness_rates_per_run(data):
    """Line chart: Correctness rates across runs"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    runs = [r['run'] for r in data['per_run_results']]
    
    # Parse percentages
    prompt_rates = [float(r['prompt_only_rate'].strip('%')) for r in data['per_run_results']]
    mermaid_rates = [float(r['mermaid_only_rate'].strip('%')) for r in data['per_run_results']]
    combined_rates = [float(r['combined_rate'].strip('%')) for r in data['per_run_results']]
    
    ax.plot(runs, prompt_rates, marker='o', linewidth=2.5, markersize=8, 
            label='Prompt Only', color='#2ecc71')
    ax.plot(runs, mermaid_rates, marker='s', linewidth=2.5, markersize=8, 
            label='Mermaid Only', color='#e74c3c')
    ax.plot(runs, combined_rates, marker='^', linewidth=2.5, markersize=8, 
            label='Combined', color='#3498db')
    
    ax.set_xlabel('Run Number', fontweight='bold')
    ax.set_ylabel('Functional Correctness Rate (%)', fontweight='bold')
    ax.set_title('Correctness Rate Stability Across Multiple Runs', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    ax.set_xticks(runs)
    
    # Add variance annotation
    prompt_std = np.std(prompt_rates)
    mermaid_std = np.std(mermaid_rates)
    combined_std = np.std(combined_rates)
    
    ax.text(0.02, 0.98, 
            f'Std Dev:\nPrompt: {prompt_std:.1f}%\nMermaid: {mermaid_std:.1f}%\nCombined: {combined_std:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_correctness_rates_per_run.png', bbox_inches='tight')
    print("✅ Created: 2_correctness_rates_per_run.png")
    plt.close()

def plot_3_average_correctness_with_error_bars(data):
    """Bar chart with error bars showing mean and variance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate means and std devs
    prompt_rates = [float(r['prompt_only_rate'].strip('%')) for r in data['per_run_results']]
    mermaid_rates = [float(r['mermaid_only_rate'].strip('%')) for r in data['per_run_results']]
    combined_rates = [float(r['combined_rate'].strip('%')) for r in data['per_run_results']]
    
    means = [np.mean(prompt_rates), np.mean(mermaid_rates), np.mean(combined_rates)]
    stds = [np.std(prompt_rates), np.std(mermaid_rates), np.std(combined_rates)]
    
    approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars = ax.bar(approaches, means, yerr=stds, capsize=10, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2., mean + std + 2,
                f'{mean:.1f}%\n±{std:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Functional Correctness Rate (%)', fontweight='bold')
    ax.set_xlabel('Approach', fontweight='bold')
    ax.set_title(f'Average Correctness Rates with Variance\n({data["num_runs"]} runs, {data["total_tests"]} total tests)', 
                 fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_average_correctness_with_error_bars.png', bbox_inches='tight')
    print("✅ Created: 3_average_correctness_with_error_bars.png")
    plt.close()

def plot_4_consistency_pie_chart(data):
    """Pie chart: Winner distribution"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    winner_dist = data['consistency']['winner_distribution']
    labels = list(winner_dist.keys())
    sizes = list(winner_dist.values())
    
    colors_map = {
        'Prompt Only': '#2ecc71',
        'Mermaid Only': '#e74c3c',
        'Combined': '#3498db'
    }
    colors = [colors_map.get(label, '#95a5a6') for label in labels]
    
    explode = [0.1 if size == max(sizes) else 0 for size in sizes]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, explode=explode,
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    ax.set_title(f'Winner Distribution Across {data["num_runs"]} Runs\n(Consistency: {data["consistency"]["consistency_rate"]})', 
                 fontweight='bold', pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_consistency_pie_chart.png', bbox_inches='tight')
    print("✅ Created: 4_consistency_pie_chart.png")
    plt.close()

def plot_5_complexity_breakdown(all_results):
    """Grouped bar chart: Performance by circuit complexity"""
    if not all_results:
        print("⚠️  Skipping complexity breakdown - no detailed data")
        return
    
    df = pd.DataFrame(all_results)
    
    # Group by complexity
    complexities = ['simple', 'medium', 'complex']
    approaches = ['prompt_only', 'mermaid_only', 'combined']
    
    data_by_complexity = {}
    for comp in complexities:
        comp_df = df[df['complexity'] == comp]
        data_by_complexity[comp] = {
            'prompt_only': (comp_df['prompt_only_consensus_anomalous'] == False).sum() / len(comp_df) * 100 if len(comp_df) > 0 else 0,
            'mermaid_only': (comp_df['mermaid_only_consensus_anomalous'] == False).sum() / len(comp_df) * 100 if len(comp_df) > 0 else 0,
            'combined': (comp_df['combined_consensus_anomalous'] == False).sum() / len(comp_df) * 100 if len(comp_df) > 0 else 0
        }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(complexities))
    width = 0.25
    
    bars1 = ax.bar(x - width, [data_by_complexity[c]['prompt_only'] for c in complexities], 
                   width, label='Prompt Only', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, [data_by_complexity[c]['mermaid_only'] for c in complexities], 
                   width, label='Mermaid Only', color='#e74c3c', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, [data_by_complexity[c]['combined'] for c in complexities], 
                   width, label='Combined', color='#3498db', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Functional Correctness Rate (%)', fontweight='bold')
    ax.set_xlabel('Circuit Complexity', fontweight='bold')
    ax.set_title('Performance by Circuit Complexity', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_complexity_breakdown.png', bbox_inches='tight')
    print("✅ Created: 5_complexity_breakdown.png")
    plt.close()

def plot_6_model_comparison(all_results):
    """Model comparison: Claude vs GPT-4o"""
    if not all_results:
        print("⚠️  Skipping model comparison - no detailed data")
        return
    
    df = pd.DataFrame(all_results)
    
    models = df['model'].unique()
    approaches = ['prompt_only', 'mermaid_only', 'combined']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model]
        
        correctness = [
            (model_df['prompt_only_consensus_anomalous'] == False).sum() / len(model_df) * 100,
            (model_df['mermaid_only_consensus_anomalous'] == False).sum() / len(model_df) * 100,
            (model_df['combined_consensus_anomalous'] == False).sum() / len(model_df) * 100
        ]
        
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        bars = ax.bar(['Prompt Only', 'Mermaid Only', 'Combined'], correctness, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Correctness Rate (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('Model Comparison: Approach Performance', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '6_model_comparison.png', bbox_inches='tight')
    print("✅ Created: 6_model_comparison.png")
    plt.close()

def plot_7_generation_time_comparison(all_results):
    """Box plot: Generation time by approach"""
    if not all_results:
        print("⚠️  Skipping time comparison - no detailed data")
        return
    
    df = pd.DataFrame(all_results)
    
    # Prepare data for box plot
    time_data = [
        df['prompt_only_generation_time'].dropna(),
        df['mermaid_only_generation_time'].dropna(),
        df['combined_generation_time'].dropna()
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(time_data, labels=['Prompt Only', 'Mermaid Only', 'Combined'],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Generation Time (seconds)', fontweight='bold')
    ax.set_xlabel('Approach', fontweight='bold')
    ax.set_title('Generation Time Distribution by Approach', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add mean values as text
    for i, data in enumerate(time_data, 1):
        mean_val = data.mean()
        median_val = data.median()
        ax.text(i, ax.get_ylim()[1] * 0.95, 
                f'Mean: {mean_val:.2f}s\nMedian: {median_val:.2f}s',
                ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '7_generation_time_comparison.png', bbox_inches='tight')
    print("✅ Created: 7_generation_time_comparison.png")
    plt.close()

def plot_8_success_rates_stacked(all_results):
    """Stacked bar: Generation vs Simulation vs Correctness success"""
    if not all_results:
        print("⚠️  Skipping success rates - no detailed data")
        return
    
    df = pd.DataFrame(all_results)
    
    approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
    
    # Calculate success rates
    gen_success = [
        df['prompt_only_generation_success'].sum() / len(df) * 100,
        df['mermaid_only_generation_success'].sum() / len(df) * 100,
        df['combined_generation_success'].sum() / len(df) * 100
    ]
    
    sim_success = [
        df['prompt_only_simulation_success'].sum() / len(df) * 100,
        df['mermaid_only_simulation_success'].sum() / len(df) * 100,
        df['combined_simulation_success'].sum() / len(df) * 100
    ]
    
    correct = [
        (df['prompt_only_consensus_anomalous'] == False).sum() / len(df) * 100,
        (df['mermaid_only_consensus_anomalous'] == False).sum() / len(df) * 100,
        (df['combined_consensus_anomalous'] == False).sum() / len(df) * 100
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(approaches))
    width = 0.6
    
    p1 = ax.bar(x, correct, width, label='Functionally Correct', 
                color='#27ae60', alpha=0.9, edgecolor='black')
    p2 = ax.bar(x, [s - c for s, c in zip(sim_success, correct)], width,
                bottom=correct, label='Simulated (but incorrect)', 
                color='#f39c12', alpha=0.9, edgecolor='black')
    p3 = ax.bar(x, [g - s for g, s in zip(gen_success, sim_success)], width,
                bottom=sim_success, label='Generated (but sim failed)', 
                color='#e74c3c', alpha=0.9, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xlabel('Approach', fontweight='bold')
    ax.set_title('Success Pipeline: Generation → Simulation → Correctness', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for i, (g, s, c) in enumerate(zip(gen_success, sim_success, correct)):
        if c > 5:
            ax.text(i, c/2, f'{c:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=10)
        if (s-c) > 5:
            ax.text(i, c + (s-c)/2, f'{s-c:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=9)
        if (g-s) > 5:
            ax.text(i, s + (g-s)/2, f'{g-s:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '8_success_pipeline_stacked.png', bbox_inches='tight')
    print("✅ Created: 8_success_pipeline_stacked.png")
    plt.close()

def generate_summary_table(data):
    """Generate a summary table as an image"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    prompt_rates = [float(r['prompt_only_rate'].strip('%')) for r in data['per_run_results']]
    mermaid_rates = [float(r['mermaid_only_rate'].strip('%')) for r in data['per_run_results']]
    combined_rates = [float(r['combined_rate'].strip('%')) for r in data['per_run_results']]
    
    table_data = [
        ['Metric', 'Prompt Only', 'Mermaid Only', 'Combined'],
        ['', '', '', ''],
        ['Average Correctness', f'{np.mean(prompt_rates):.1f}%', 
         f'{np.mean(mermaid_rates):.1f}%', f'{np.mean(combined_rates):.1f}%'],
        ['Std Deviation', f'±{np.std(prompt_rates):.1f}%', 
         f'±{np.std(mermaid_rates):.1f}%', f'±{np.std(combined_rates):.1f}%'],
        ['Total Wins', str(data['overall_stats']['prompt_only_wins']), 
         str(data['overall_stats']['mermaid_only_wins']), 
         str(data['overall_stats']['combined_wins'])],
        ['Consistency', data['consistency']['consistency_rate'], '-', '-'],
        ['', '', '', ''],
        ['Overall Winner', data['consistency']['most_consistent'], '', '']
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Style winner row
    cell = table[(7, 0)]
    cell.set_facecolor('#2ecc71')
    cell.set_text_props(weight='bold')
    cell = table[(7, 1)]
    cell.set_facecolor('#2ecc71')
    cell.set_text_props(weight='bold', fontsize=13)
    
    # Color code the metrics
    colors = ['#ecf0f1', 'white']
    for i in range(2, 7):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i % 2])
    
    plt.title(f'Summary Statistics - {data["num_runs"]} Runs ({data["total_tests"]} Total Tests)', 
              fontweight='bold', fontsize=14, pad=20)
    
    plt.savefig(OUTPUT_DIR / '9_summary_table.png', bbox_inches='tight', dpi=300)
    print("✅ Created: 9_summary_table.png")
    plt.close()

def main():
    print("="*80)
    print("📊 THESIS VISUALIZATION GENERATOR")
    print("="*80)
    
    # Load data
    aggregated_data = load_aggregated_data()
    if not aggregated_data:
        return
    
    all_results = load_all_detailed_results()
    
    print(f"\n✅ Loaded data from {aggregated_data['num_runs']} runs")
    print(f"✅ Total tests: {aggregated_data['total_tests']}")
    print(f"\n📈 Generating visualizations...\n")
    
    # Generate all plots
    plot_1_overall_winner_comparison(aggregated_data)
    plot_2_correctness_rates_per_run(aggregated_data)
    plot_3_average_correctness_with_error_bars(aggregated_data)
    plot_4_consistency_pie_chart(aggregated_data)
    plot_5_complexity_breakdown(all_results)
    plot_6_model_comparison(all_results)
    plot_7_generation_time_comparison(all_results)
    plot_8_success_rates_stacked(all_results)
    generate_summary_table(aggregated_data)
    
    print(f"\n{'='*80}")
    print(f"✅ ALL VISUALIZATIONS CREATED!")
    print(f"{'='*80}")
    print(f"\n📂 Saved to: {OUTPUT_DIR.relative_to(ROOT_DIR)}")
    print(f"\n📊 Generated 9 publication-quality figures:")
    print(f"   1. Overall Winner Comparison (Bar Chart)")
    print(f"   2. Correctness Rates Per Run (Line Chart)")
    print(f"   3. Average Correctness with Error Bars (Bar Chart)")
    print(f"   4. Winner Distribution Consistency (Pie Chart)")
    print(f"   5. Performance by Complexity (Grouped Bar)")
    print(f"   6. Model Comparison (Side-by-side Bar)")
    print(f"   7. Generation Time Distribution (Box Plot)")
    print(f"   8. Success Pipeline (Stacked Bar)")
    print(f"   9. Summary Statistics Table")
    print(f"\n✅ All images are 300 DPI, publication-ready!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()