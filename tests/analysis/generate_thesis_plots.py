"""
Comprehensive Analysis and Visualization for Verilog Generation Results
Generates plots and insights comparing Claude vs GPT-4o

SAVE AS: tests/analysis/generate_thesis_plots.py
RUN FROM: tests/analysis/ directory
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = ROOT_DIR / "thesis_generation_results"
OUTPUT_DIR = RESULTS_DIR / "analysis_plots"

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = {'claude': '#7B68EE', 'gpt-4o': '#FF6B6B'}
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
DPI = 300

def load_latest_results() -> Tuple[pd.DataFrame, Dict]:
    """Load the most recent comprehensive test results"""
    json_files = list((RESULTS_DIR / "comprehensive").glob("comprehensive_results_*.json"))
    
    if not json_files:
        print("❌ No results found! Run test_comprehensive_suite.py first.")
        sys.exit(1)
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['results'])
    metadata = {k: v for k, v in data.items() if k != 'results'}
    
    return df, metadata

def load_multiple_runs() -> List[Tuple[pd.DataFrame, Dict]]:
    """Load all test runs for variability analysis"""
    json_files = sorted(
        (RESULTS_DIR / "comprehensive").glob("comprehensive_results_*.json"),
        key=lambda p: p.stat().st_mtime
    )
    
    if len(json_files) < 2:
        return None
    
    print(f"📂 Found {len(json_files)} test runs for variability analysis")
    
    runs = []
    for f in json_files:
        with open(f, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data['results'])
            metadata = {k: v for k, v in data.items() if k != 'results'}
            runs.append((df, metadata))
    
    return runs

def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive statistics"""
    stats = {}
    
    for model in ['claude', 'gpt-4o']:
        model_df = df[df['generator_model'] == model]
        
        stats[model] = {
            'total_circuits': len(model_df),
            'generation_success': (model_df['generation_success'].sum() / len(model_df) * 100),
            'simulation_success': (model_df['simulation_success'].sum() / len(model_df) * 100),
            'functional_correct': (model_df['consensus_anomalous'] == False).sum(),
            'functional_correct_pct': ((model_df['consensus_anomalous'] == False).sum() / len(model_df) * 100),
            'has_testbench_pct': (model_df['has_testbench'].sum() / len(model_df) * 100),
            'avg_loc': model_df['lines_of_code'].mean(),
            'avg_gen_time': model_df['generation_time'].mean(),
            'avg_sim_time': model_df['simulation_time'].mean(),
        }
        
        # By complexity
        for complexity in ['medium', 'complex']:
            comp_df = model_df[model_df['complexity'] == complexity]
            stats[model][f'{complexity}_correct'] = (comp_df['consensus_anomalous'] == False).sum()
            stats[model][f'{complexity}_total'] = len(comp_df)
            stats[model][f'{complexity}_pct'] = ((comp_df['consensus_anomalous'] == False).sum() / len(comp_df) * 100) if len(comp_df) > 0 else 0
    
    return stats

def plot_overall_comparison(df: pd.DataFrame, stats: Dict):
    """Plot 1: Overall Performance Comparison"""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    
    metrics = [
        ('Functional\nCorrectness', 'functional_correct_pct'),
        ('Generation\nSuccess', 'generation_success'),
        ('Simulation\nSuccess', 'simulation_success')
    ]
    
    for ax, (label, key) in zip(axes, metrics):
        claude_val = stats['claude'][key]
        gpt4o_val = stats['gpt-4o'][key]
        
        bars = ax.bar(['Claude Sonnet 3.5', 'GPT-4o'], 
                      [claude_val, gpt4o_val],
                      color=[COLORS['claude'], COLORS['gpt-4o']],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_overall_comparison.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 01_overall_comparison.png")
    plt.close()

def plot_complexity_breakdown(df: pd.DataFrame):
    """Plot 2: Performance by Complexity Level"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    complexity_data = []
    for model in ['claude', 'gpt-4o']:
        for complexity in ['medium', 'complex']:
            subset = df[(df['generator_model'] == model) & (df['complexity'] == complexity)]
            correct = (subset['consensus_anomalous'] == False).sum()
            total = len(subset)
            pct = (correct / total * 100) if total > 0 else 0
            complexity_data.append({
                'Model': 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o',
                'Complexity': complexity.capitalize(),
                'Percentage': pct,
                'Count': f'{correct}/{total}'
            })
    
    complexity_df = pd.DataFrame(complexity_data)
    
    x = np.arange(len(['Medium', 'Complex']))
    width = 0.35
    
    claude_vals = complexity_df[complexity_df['Model'] == 'Claude Sonnet 3.5']['Percentage'].values
    gpt4o_vals = complexity_df[complexity_df['Model'] == 'GPT-4o']['Percentage'].values
    
    bars1 = ax.bar(x - width/2, claude_vals, width, label='Claude Sonnet 3.5', 
                   color=COLORS['claude'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, gpt4o_vals, width, label='GPT-4o',
                   color=COLORS['gpt-4o'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Functional Correctness (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Circuit Complexity', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Medium (20 circuits)', 'Complex (20 circuits)'], fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_complexity_breakdown.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 02_complexity_breakdown.png")
    plt.close()

def plot_category_performance(df: pd.DataFrame):
    """Plot 3: Performance by Circuit Category"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = df['category'].unique()
    category_data = []
    
    for model in ['claude', 'gpt-4o']:
        for category in sorted(categories):
            subset = df[(df['generator_model'] == model) & (df['category'] == category)]
            if len(subset) > 0:
                correct = (subset['consensus_anomalous'] == False).sum()
                total = len(subset)
                pct = (correct / total * 100)
                category_data.append({
                    'Model': 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o',
                    'Category': category,
                    'Percentage': pct,
                    'Count': f'{correct}/{total}'
                })
    
    category_df = pd.DataFrame(category_data)
    
    # Create grouped bar chart
    categories_sorted = sorted(categories)
    x = np.arange(len(categories_sorted))
    width = 0.35
    
    claude_vals = [category_df[(category_df['Model'] == 'Claude Sonnet 3.5') & 
                               (category_df['Category'] == cat)]['Percentage'].values[0] 
                   if len(category_df[(category_df['Model'] == 'Claude Sonnet 3.5') & 
                                     (category_df['Category'] == cat)]) > 0 else 0
                   for cat in categories_sorted]
    
    gpt4o_vals = [category_df[(category_df['Model'] == 'GPT-4o') & 
                              (category_df['Category'] == cat)]['Percentage'].values[0]
                  if len(category_df[(category_df['Model'] == 'GPT-4o') & 
                                    (category_df['Category'] == cat)]) > 0 else 0
                  for cat in categories_sorted]
    
    bars1 = ax.bar(x - width/2, claude_vals, width, label='Claude Sonnet 3.5',
                   color=COLORS['claude'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, gpt4o_vals, width, label='GPT-4o',
                   color=COLORS['gpt-4o'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Functional Correctness (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Circuit Category', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories_sorted, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.0f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_category_performance.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 03_category_performance.png")
    plt.close()

def plot_code_metrics(df: pd.DataFrame):
    """Plot 4: Code Quality Metrics"""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    # Lines of Code
    claude_loc = df[df['generator_model'] == 'claude']['lines_of_code']
    gpt4o_loc = df[df['generator_model'] == 'gpt-4o']['lines_of_code']
    
    axes[0].boxplot([claude_loc, gpt4o_loc], labels=['Claude Sonnet 3.5', 'GPT-4o'],
                    patch_artist=True,
                    boxprops=dict(facecolor=COLORS['claude'], alpha=0.6),
                    medianprops=dict(color='black', linewidth=2))
    axes[0].set_ylabel('Lines of Code', fontsize=11, fontweight='bold')
    axes[0].set_title('Code Length Distribution', fontsize=13, fontweight='bold', pad=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Testbench inclusion
    testbench_data = []
    for model in ['claude', 'gpt-4o']:
        model_df = df[df['generator_model'] == model]
        pct = (model_df['has_testbench'].sum() / len(model_df) * 100)
        testbench_data.append(pct)
    
    bars = axes[1].bar(['Claude Sonnet 3.5', 'GPT-4o'], testbench_data,
                       color=[COLORS['claude'], COLORS['gpt-4o']],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Testbench Inclusion Rate', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_code_metrics.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 04_code_metrics.png")
    plt.close()

def plot_verifier_agreement(df: pd.DataFrame):
    """Plot 5: Verifier Agreement Analysis"""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    # Agreement count distribution
    for model_idx, model in enumerate(['claude', 'gpt-4o']):
        model_df = df[df['generator_model'] == model]
        agreement_counts = model_df['agreement_count'].value_counts().sort_index()
        
        axes[0].bar(agreement_counts.index + (model_idx * 0.35), 
                   agreement_counts.values,
                   width=0.35,
                   label='Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o',
                   color=COLORS[model], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[0].set_xlabel('Number of Agreeing Verifiers', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Circuit Count', fontsize=11, fontweight='bold')
    axes[0].set_title('Verifier Agreement Distribution', fontsize=13, fontweight='bold', pad=10)
    axes[0].legend(fontsize=10)
    axes[0].set_xticks([1, 2, 3])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Consensus confidence
    claude_conf = df[df['generator_model'] == 'claude']['consensus_confidence']
    gpt4o_conf = df[df['generator_model'] == 'gpt-4o']['consensus_confidence']
    
    axes[1].hist([claude_conf, gpt4o_conf], bins=10, 
                 label=['Claude Sonnet 3.5', 'GPT-4o'],
                 color=[COLORS['claude'], COLORS['gpt-4o']], 
                 alpha=0.6, edgecolor='black', linewidth=1.2)
    axes[1].set_xlabel('Consensus Confidence', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Circuit Count', fontsize=11, fontweight='bold')
    axes[1].set_title('Verification Confidence Distribution', fontsize=13, fontweight='bold', pad=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_verifier_agreement.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 05_verifier_agreement.png")
    plt.close()

def plot_performance_times(df: pd.DataFrame):
    """Plot 6: Generation and Simulation Times"""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    times = ['generation_time', 'simulation_time']
    titles = ['Code Generation Time', 'Simulation Time']
    
    for ax, time_col, title in zip(axes, times, titles):
        claude_times = df[df['generator_model'] == 'claude'][time_col]
        gpt4o_times = df[df['generator_model'] == 'gpt-4o'][time_col]
        
        bp = ax.boxplot([claude_times, gpt4o_times],
                        labels=['Claude Sonnet 3.5', 'GPT-4o'],
                        patch_artist=True,
                        boxprops=dict(alpha=0.6),
                        medianprops=dict(color='black', linewidth=2))
        
        bp['boxes'][0].set_facecolor(COLORS['claude'])
        bp['boxes'][1].set_facecolor(COLORS['gpt-4o'])
        
        ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean values
        claude_mean = claude_times.mean()
        gpt4o_mean = gpt4o_times.mean()
        ax.text(1, claude_mean, f'μ={claude_mean:.1f}s', 
               ha='right', va='bottom', fontsize=9, fontweight='bold')
        ax.text(2, gpt4o_mean, f'μ={gpt4o_mean:.1f}s',
               ha='left', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_performance_times.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 06_performance_times.png")
    plt.close()

def plot_success_pipeline(df: pd.DataFrame):
    """Plot 7: Success Through Pipeline Stages"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = ['Generated', 'Compiled', 'Simulated', 'Functionally\nCorrect']
    
    for model in ['claude', 'gpt-4o']:
        model_df = df[df['generator_model'] == model]
        total = len(model_df)
        
        values = [
            model_df['generation_success'].sum(),
            model_df['compilation_success'].sum(),
            model_df['simulation_success'].sum(),
            (model_df['consensus_anomalous'] == False).sum()
        ]
        
        percentages = [(v / total * 100) for v in values]
        
        ax.plot(stages, percentages, marker='o', markersize=10, linewidth=2.5,
               label='Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o',
               color=COLORS[model])
        
        # Add value labels
        for i, (stage, pct, val) in enumerate(zip(stages, percentages, values)):
            ax.text(i, pct + 3, f'{val}\n({pct:.1f}%)', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Through Development Pipeline', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_success_pipeline.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 07_success_pipeline.png")
    plt.close()

def plot_top_bottom_circuits(df: pd.DataFrame):
    """Plot 8: Best and Worst Performing Circuits"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate success rate per circuit
    circuit_performance = []
    for circuit in df['circuit_name'].unique():
        circuit_df = df[df['circuit_name'] == circuit]
        success_count = (circuit_df['consensus_anomalous'] == False).sum()
        total = len(circuit_df)
        rate = (success_count / total * 100) if total > 0 else 0
        circuit_performance.append({
            'circuit': circuit,
            'success_rate': rate,
            'complexity': circuit_df['complexity'].iloc[0]
        })
    
    perf_df = pd.DataFrame(circuit_performance).sort_values('success_rate')
    
    # Bottom 10
    bottom_10 = perf_df.head(10)
    colors_bottom = [COLORS['claude'] if c == 'medium' else COLORS['gpt-4o'] 
                     for c in bottom_10['complexity']]
    
    axes[0].barh(range(len(bottom_10)), bottom_10['success_rate'], 
                 color=colors_bottom, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_yticks(range(len(bottom_10)))
    axes[0].set_yticklabels(bottom_10['circuit'], fontsize=9)
    axes[0].set_xlabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Bottom 10 Performing Circuits', fontsize=13, fontweight='bold', pad=10)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Top 10
    top_10 = perf_df.tail(10)
    colors_top = [COLORS['claude'] if c == 'medium' else COLORS['gpt-4o'] 
                  for c in top_10['complexity']]
    
    axes[1].barh(range(len(top_10)), top_10['success_rate'],
                 color=colors_top, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_yticks(range(len(top_10)))
    axes[1].set_yticklabels(top_10['circuit'], fontsize=9)
    axes[1].set_xlabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Top 10 Performing Circuits', fontsize=13, fontweight='bold', pad=10)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_top_bottom_circuits.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 08_top_bottom_circuits.png")
    plt.close()

def plot_llm_variability(runs: List[Tuple[pd.DataFrame, Dict]]):
    """Plot 9: LLM Non-Determinism Analysis"""
    if not runs or len(runs) < 2:
        print("⚠️  Skipping variability analysis (need 2+ runs)")
        return
    
    print(f"\n🔬 Analyzing LLM variability across {len(runs)} runs...")
    
    # For 2 runs: side-by-side comparison
    # For 3+ runs: statistical analysis with confidence intervals
    
    if len(runs) <= 3:
        # Simple comparison plot
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
        plot_simple_variability(runs, axes)
    else:
        # Advanced statistical plot with confidence intervals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plot_advanced_variability(runs, axes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_llm_variability.png', dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: 09_llm_variability.png")
    plt.close()

def plot_simple_variability(runs: List[Tuple[pd.DataFrame, Dict]], axes):
    """Simple 2-3 run comparison"""
    # Track per-circuit variability
    circuit_results = defaultdict(lambda: {'claude': [], 'gpt-4o': []})
    
    for df, _ in runs:
        for circuit in df['circuit_name'].unique():
            for model in ['claude', 'gpt-4o']:
                subset = df[(df['circuit_name'] == circuit) & (df['generator_model'] == model)]
                if len(subset) > 0:
                    correct = (~subset['consensus_anomalous'].fillna(True)).iloc[0]
                    circuit_results[circuit][model].append(correct)
    
    # Calculate consistency
    variability_data = {'claude': [], 'gpt-4o': []}
    for circuit, results in circuit_results.items():
        for model in ['claude', 'gpt-4o']:
            if len(results[model]) > 1:
                unique_results = len(set(results[model]))
                if unique_results > 1:
                    variability_data[model].append(1)
                else:
                    variability_data[model].append(0)
    
    # Plot 1: Consistency rate
    consistency_rates = []
    for model in ['claude', 'gpt-4o']:
        if variability_data[model]:
            consistency = (1 - np.mean(variability_data[model])) * 100
            consistency_rates.append(consistency)
        else:
            consistency_rates.append(0)
    
    bars = axes[0].bar(['Claude Sonnet 3.5', 'GPT-4o'], consistency_rates,
                       color=[COLORS['claude'], COLORS['gpt-4o']],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Consistency Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('LLM Generation Consistency', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # Plot 2: Run-to-run comparison
    run_stats = []
    for idx, (df, meta) in enumerate(runs):
        for model in ['claude', 'gpt-4o']:
            model_df = df[df['generator_model'] == model]
            correct_pct = (~model_df['consensus_anomalous'].fillna(True)).mean() * 100
            run_stats.append({
                'Run': f"Run {idx+1}",
                'Model': 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o',
                'Correctness': correct_pct
            })
    
    stats_df = pd.DataFrame(run_stats)
    
    x_pos = np.arange(len(runs))
    width = 0.35
    
    claude_vals = stats_df[stats_df['Model'] == 'Claude Sonnet 3.5']['Correctness'].values
    gpt4o_vals = stats_df[stats_df['Model'] == 'GPT-4o']['Correctness'].values
    
    bars1 = axes[1].bar(x_pos - width/2, claude_vals, width,
                        label='Claude Sonnet 3.5', color=COLORS['claude'],
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = axes[1].bar(x_pos + width/2, gpt4o_vals, width,
                        label='GPT-4o', color=COLORS['gpt-4o'],
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[1].set_ylabel('Functional Correctness (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Performance Across Runs', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"Run {i+1}" for i in range(len(runs))], fontsize=11)
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
    
    # Print stats
    print(f"\n📊 Variability Analysis:")
    print(f"  Claude consistency: {consistency_rates[0]:.1f}%")
    print(f"  GPT-4o consistency: {consistency_rates[1]:.1f}%")
    if len(runs) >= 2:
        swing_claude = np.std(claude_vals)
        swing_gpt4o = np.std(gpt4o_vals)
        print(f"  Claude std dev: {swing_claude:.1f}%")
        print(f"  GPT-4o std dev: {swing_gpt4o:.1f}%")

def plot_advanced_variability(runs: List[Tuple[pd.DataFrame, Dict]], axes):
    """Advanced analysis for 4+ runs with confidence intervals"""
    print(f"  📊 Computing confidence intervals from {len(runs)} runs...")
    
    # Collect all results
    all_results = []
    for idx, (df, meta) in enumerate(runs):
        for model in ['claude', 'gpt-4o']:
            model_df = df[df['generator_model'] == model]
            
            overall = (~model_df['consensus_anomalous'].fillna(True)).mean() * 100
            medium = (~model_df[model_df['complexity'] == 'medium']['consensus_anomalous'].fillna(True)).mean() * 100
            complex_pct = (~model_df[model_df['complexity'] == 'complex']['consensus_anomalous'].fillna(True)).mean() * 100
            
            all_results.append({
                'run': idx + 1,
                'model': model,
                'model_name': 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o',
                'overall': overall,
                'medium': medium,
                'complex': complex_pct
            })
    
    results_df = pd.DataFrame(all_results)
    
    # Plot 1: Overall with confidence intervals
    for idx, model in enumerate(['claude', 'gpt-4o']):
        model_data = results_df[results_df['model'] == model]['overall']
        mean = model_data.mean()
        std = model_data.std()
        
        model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
        axes[0, 0].bar(idx, mean, color=COLORS[model], alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        axes[0, 0].errorbar(idx, mean, yerr=std, fmt='none', 
                           color='black', capsize=5, linewidth=2)
        
        axes[0, 0].text(idx, mean + std + 2, 
                       f'{mean:.1f}%\n±{std:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['Claude Sonnet 3.5', 'GPT-4o'], fontsize=10)
    axes[0, 0].set_ylabel('Functional Correctness (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'Mean Performance Over {len(runs)} Runs', 
                        fontsize=12, fontweight='bold', pad=10)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Complexity breakdown with CI
    complexities = ['medium', 'complex']
    x = np.arange(len(complexities))
    width = 0.35
    
    for complexity_idx, complexity in enumerate(complexities):
        for model_idx, model in enumerate(['claude', 'gpt-4o']):
            model_data = results_df[results_df['model'] == model][complexity]
            mean = model_data.mean()
            std = model_data.std()
            
            pos = x[complexity_idx] + (model_idx - 0.5) * width
            axes[0, 1].bar(pos, mean, width, color=COLORS[model],
                          alpha=0.8, edgecolor='black', linewidth=1.5,
                          label='Claude Sonnet 3.5' if (complexity_idx == 0 and model == 'claude') else 
                                'GPT-4o' if (complexity_idx == 0 and model == 'gpt-4o') else '')
            axes[0, 1].errorbar(pos, mean, yerr=std, fmt='none',
                               color='black', capsize=3, linewidth=1.5)
    
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Medium', 'Complex'], fontsize=11)
    axes[0, 1].set_ylabel('Functional Correctness (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Performance by Complexity', fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].legend(fontsize=9, loc='upper right')
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Distribution violin plots
    for idx, model in enumerate(['claude', 'gpt-4o']):
        model_data = results_df[results_df['model'] == model]['overall'].values
        
        parts = axes[1, 0].violinplot([model_data], positions=[idx], 
                                      widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS[model])
            pc.set_alpha(0.6)
    
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(['Claude Sonnet 3.5', 'GPT-4o'], fontsize=10)
    axes[1, 0].set_ylabel('Functional Correctness (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Performance Distribution', fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Trajectory over runs
    for model in ['claude', 'gpt-4o']:
        model_data = results_df[results_df['model'] == model].sort_values('run')
        model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
        
        axes[1, 1].plot(model_data['run'], model_data['overall'], 
                       marker='o', linewidth=2, markersize=8,
                       color=COLORS[model], label=model_name, alpha=0.8)
    
    axes[1, 1].set_xlabel('Run Number', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Functional Correctness (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Performance Trajectory', fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim(0, 105)
    
    # Print detailed stats
    print(f"\n📊 Statistical Summary ({len(runs)} runs):")
    for model in ['claude', 'gpt-4o']:
        model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
        overall_data = results_df[results_df['model'] == model]['overall']
        print(f"\n  {model_name}:")
        print(f"    Mean: {overall_data.mean():.1f}%")
        print(f"    Std Dev: {overall_data.std():.1f}%")
        print(f"    Min: {overall_data.min():.1f}%")
        print(f"    Max: {overall_data.max():.1f}%")
        print(f"    Range: {overall_data.max() - overall_data.min():.1f}%")

def generate_summary_report(df: pd.DataFrame, stats: Dict, metadata: Dict, runs=None):
    """Generate text summary report"""
    report_path = OUTPUT_DIR / 'analysis_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE VERILOG GENERATION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Session: {metadata.get('session_id', 'N/A')}\n")
        f.write(f"Total Circuits: {metadata.get('total_circuits', 0)}\n")
        f.write(f"Timestamp: {metadata.get('timestamp', 'N/A')}\n")
        
        if runs and len(runs) >= 2:
            f.write(f"Total Runs Analyzed: {len(runs)}\n")
            f.write(f"⚠️  VARIABILITY DETECTED: Multiple runs show different results\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for model in ['claude', 'gpt-4o']:
            model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
            f.write(f"\n{model_name}:\n")
            f.write(f"  Total Circuits Tested: {stats[model]['total_circuits']}\n")
            f.write(f"  Functional Correctness: {stats[model]['functional_correct']}/{stats[model]['total_circuits']} ({stats[model]['functional_correct_pct']:.1f}%)\n")
            f.write(f"  Generation Success: {stats[model]['generation_success']:.1f}%\n")
            f.write(f"  Simulation Success: {stats[model]['simulation_success']:.1f}%\n")
            f.write(f"  Testbench Inclusion: {stats[model]['has_testbench_pct']:.1f}%\n")
            f.write(f"  Average LOC: {stats[model]['avg_loc']:.1f}\n")
            f.write(f"  Average Generation Time: {stats[model]['avg_gen_time']:.2f}s\n")
            f.write(f"  Average Simulation Time: {stats[model]['avg_sim_time']:.2f}s\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE BY COMPLEXITY\n")
        f.write("="*80 + "\n\n")
        
        for complexity in ['medium', 'complex']:
            f.write(f"\n{complexity.upper()} Circuits:\n")
            for model in ['claude', 'gpt-4o']:
                model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
                correct = stats[model][f'{complexity}_correct']
                total = stats[model][f'{complexity}_total']
                pct = stats[model][f'{complexity}_pct']
                f.write(f"  {model_name}: {correct}/{total} ({pct:.1f}%)\n")
        
        if runs and len(runs) >= 2:
            f.write("\n" + "="*80 + "\n")
            f.write("LLM NON-DETERMINISM ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"⚠️  CRITICAL FINDING: LLMs show significant variability\n\n")
            
            for idx, (run_df, run_meta) in enumerate(runs):
                f.write(f"Run {idx+1} ({run_meta.get('timestamp', 'N/A')}):\n")
                for model in ['claude', 'gpt-4o']:
                    model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
                    model_df = run_df[run_df['generator_model'] == model]
                    correct_pct = (~model_df['consensus_anomalous'].fillna(True)).mean() * 100
                    
                    # By complexity
                    med = model_df[model_df['complexity'] == 'medium']
                    comp = model_df[model_df['complexity'] == 'complex']
                    med_pct = (~med['consensus_anomalous'].fillna(True)).mean() * 100 if len(med) > 0 else 0
                    comp_pct = (~comp['consensus_anomalous'].fillna(True)).mean() * 100 if len(comp) > 0 else 0
                    
                    f.write(f"  {model_name}: {correct_pct:.1f}% (Med: {med_pct:.1f}%, Complex: {comp_pct:.1f}%)\n")
                f.write("\n")
            
            if len(runs) == 2:
                f.write("Implications:\n")
                f.write("  - Same prompt can produce different circuits\n")
                f.write("  - Results are NOT deterministic\n")
                f.write("  - Multiple runs recommended for accurate assessment\n")
                f.write("  - Temperature/sampling settings impact reliability\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        # Calculate insights
        claude_better = stats['claude']['functional_correct_pct'] > stats['gpt-4o']['functional_correct_pct']
        diff = abs(stats['claude']['functional_correct_pct'] - stats['gpt-4o']['functional_correct_pct'])
        
        f.write(f"1. Overall Performance:\n")
        f.write(f"   {'Claude Sonnet 3.5' if claude_better else 'GPT-4o'} performs better ")
        f.write(f"by {diff:.1f} percentage points\n\n")
        
        f.write(f"2. Complexity Impact:\n")
        for model in ['claude', 'gpt-4o']:
            model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
            med_pct = stats[model]['medium_pct']
            comp_pct = stats[model]['complex_pct']
            drop = med_pct - comp_pct
            f.write(f"   {model_name}: {drop:.1f}% drop from medium to complex\n")
        
        f.write(f"\n3. Generation Reliability:\n")
        for model in ['claude', 'gpt-4o']:
            model_name = 'Claude Sonnet 3.5' if model == 'claude' else 'GPT-4o'
            gen = stats[model]['generation_success']
            sim = stats[model]['simulation_success']
            f.write(f"   {model_name}: {gen:.1f}% generation, {sim:.1f}% simulation\n")
    
    print(f"✅ Saved: analysis_summary.txt")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE ANALYSIS & VISUALIZATION")
    print("="*80 + "\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, metadata = load_latest_results()
    print(f"✅ Loaded {len(df)} test results\n")
    
    # Load multiple runs if available
    runs = load_multiple_runs()
    
    # Calculate statistics
    print("📈 Calculating statistics...")
    stats = calculate_statistics(df)
    
    # Generate plots
    print("\n🎨 Generating plots...\n")
    
    plot_overall_comparison(df, stats)
    plot_complexity_breakdown(df)
    plot_category_performance(df)
    plot_code_metrics(df)
    plot_verifier_agreement(df)
    plot_performance_times(df)
    plot_success_pipeline(df)
    plot_top_bottom_circuits(df)
    
    # Variability analysis if multiple runs
    if runs and len(runs) >= 2:
        plot_llm_variability(runs)
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    generate_summary_report(df, stats, metadata, runs)
    
    print("\n" + "="*80)
    print(f"✅ Analysis complete! All outputs saved to:")
    print(f"   {OUTPUT_DIR.relative_to(ROOT_DIR)}")
    print("="*80 + "\n")
    
    # Print quick summary
    print("📊 Quick Summary:")
    print(f"   Claude Sonnet 3.5: {stats['claude']['functional_correct_pct']:.1f}% functional correctness")
    print(f"   GPT-4o: {stats['gpt-4o']['functional_correct_pct']:.1f}% functional correctness")
    print(f"   Total circuits analyzed: {metadata.get('total_circuits', 0)}")
    
    if runs and len(runs) >= 2:
        print(f"\n   🔄 Detected {len(runs)} runs - variability analysis included!")
    print()

if __name__ == "__main__":
    main()