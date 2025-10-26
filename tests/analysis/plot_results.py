"""
Multi-Modal Test Results Visualization & Analysis
Generates comprehensive plots and metrics from test result JSON files

This script analyzes:
1. Accuracy metrics (generation success, compilation, simulation)
2. Code quality metrics (lines of code, testbench presence)
3. Verification metrics (VAE, LLM verdicts, consensus)
4. Model comparisons (GPT-4o vs Claude)
5. Complexity analysis (simple vs medium vs complex)
6. Consistency analysis (variability across runs)

Input: JSON files from test_fair_multimodal.py or test_multimodal_mermaid.py
Output: Publication-quality plots in multiple formats (PNG, PDF, SVG)

SAVE AS: tests/analysis/plot_results.py
RUN: python tests/analysis/plot_results.py path/to/results.json
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def load_results(json_path: Path) -> Tuple[Dict, pd.DataFrame]:
    """Load results from JSON file"""
    print(f"📂 Loading results from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data['results'])
    
    print(f"✅ Loaded {len(df)} test results")
    print(f"   Circuits: {df['circuit_name'].nunique()}")
    print(f"   Models: {df['model'].nunique()}")
    
    return data, df

def plot_generation_success_rates(df: pd.DataFrame, output_dir: Path):
    """Plot 1: Generation success rates by model and complexity"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1a: Overall success rate by model
    success_by_model = df.groupby('model')['generation_success'].agg(['mean', 'std', 'count'])
    success_by_model['mean'] *= 100  # Convert to percentage
    success_by_model['std'] *= 100
    success_by_model['sem'] = success_by_model['std'] / np.sqrt(success_by_model['count'])
    
    ax = axes[0]
    x = np.arange(len(success_by_model))
    bars = ax.bar(x, success_by_model['mean'], 
                  yerr=success_by_model['sem'],
                  capsize=5, alpha=0.8,
                  color=['#2E86AB', '#A23B72'])
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Generation Success Rate (%)', fontweight='bold')
    ax.set_title('Overall Generation Success Rate', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(success_by_model.index)
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 1b: Success rate by complexity
    success_by_complexity = df.groupby(['complexity', 'model'])['generation_success'].mean() * 100
    success_by_complexity = success_by_complexity.unstack()
    
    ax = axes[1]
    success_by_complexity.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax.set_xlabel('Circuit Complexity', fontweight='bold')
    ax.set_ylabel('Generation Success Rate (%)', fontweight='bold')
    ax.set_title('Success Rate by Complexity', fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Model', frameon=True)
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f'generation_success_rates.{fmt}', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: generation_success_rates.[png/pdf/svg]")
    plt.close()

def plot_code_quality_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot 2: Code quality metrics (LOC, testbench presence)"""
    
    # Filter successful generations
    df_success = df[df['generation_success'] == True].copy()
    
    if len(df_success) == 0:
        print("⚠️  No successful generations to plot code quality")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 2a: Lines of code distribution by model
    ax = axes[0, 0]
    for model in df_success['model'].unique():
        model_data = df_success[df_success['model'] == model]['lines_of_code']
        ax.hist(model_data, bins=20, alpha=0.6, label=model, edgecolor='black')
    
    ax.set_xlabel('Lines of Code', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Generated Code Length Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2b: Average LOC by complexity
    ax = axes[0, 1]
    loc_by_complexity = df_success.groupby(['complexity', 'model'])['lines_of_code'].agg(['mean', 'std'])
    
    complexity_order = ['simple', 'medium', 'complex']
    models = df_success['model'].unique()
    x = np.arange(len(complexity_order))
    width = 0.35
    
    for i, model in enumerate(models):
        means = [loc_by_complexity.loc[(c, model), 'mean'] if (c, model) in loc_by_complexity.index else 0 
                for c in complexity_order]
        stds = [loc_by_complexity.loc[(c, model), 'std'] if (c, model) in loc_by_complexity.index else 0 
               for c in complexity_order]
        ax.bar(x + i*width, means, width, yerr=stds, label=model, alpha=0.8, capsize=3)
    
    ax.set_xlabel('Circuit Complexity', fontweight='bold')
    ax.set_ylabel('Average Lines of Code', fontweight='bold')
    ax.set_title('Code Length by Complexity', fontweight='bold', fontsize=12)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(complexity_order)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2c: Testbench presence rate
    ax = axes[1, 0]
    testbench_rate = df_success.groupby('model')['has_testbench'].mean() * 100
    bars = ax.bar(range(len(testbench_rate)), testbench_rate.values, 
                  alpha=0.8, color=['#2E86AB', '#A23B72'])
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Testbench Inclusion Rate (%)', fontweight='bold')
    ax.set_title('Testbench Generation Rate', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(testbench_rate)))
    ax.set_xticklabels(testbench_rate.index)
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2d: LOC vs Complexity scatter
    ax = axes[1, 1]
    complexity_map = {'simple': 1, 'medium': 2, 'complex': 3}
    df_success['complexity_num'] = df_success['complexity'].map(complexity_map)
    
    for model in df_success['model'].unique():
        model_data = df_success[df_success['model'] == model]
        ax.scatter(model_data['complexity_num'], model_data['lines_of_code'],
                  alpha=0.5, label=model, s=50)
    
    ax.set_xlabel('Circuit Complexity', fontweight='bold')
    ax.set_ylabel('Lines of Code', fontweight='bold')
    ax.set_title('Code Length vs Complexity', fontweight='bold', fontsize=12)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Simple', 'Medium', 'Complex'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f'code_quality_metrics.{fmt}', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: code_quality_metrics.[png/pdf/svg]")
    plt.close()

def plot_verification_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot 3: Verification metrics (compilation, simulation, VAE, LLM)"""
    
    df_success = df[df['generation_success'] == True].copy()
    
    if len(df_success) == 0:
        print("⚠️  No successful generations to plot verification metrics")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 3a: Compilation & Simulation success pipeline
    ax = axes[0, 0]
    stages = ['Generated', 'Compiled', 'Simulated']
    
    for model in df_success['model'].unique():
        model_data = df_success[df_success['model'] == model]
        counts = [
            len(model_data),
            model_data['compilation_success'].sum(),
            model_data['simulation_success'].sum()
        ]
        rates = [c / len(model_data) * 100 for c in counts]
        ax.plot(stages, rates, marker='o', linewidth=2, markersize=8, label=model)
    
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Generation → Compilation → Simulation Pipeline', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 3b: VAE vs LLM agreement
    df_verified = df_success[df_success['simulation_success'] == True].copy()
    
    if len(df_verified) > 0:
        ax = axes[0, 1]
        
        # Create confusion matrix
        vae_verdicts = df_verified['vae_verdict'].fillna(-1).astype(int)
        llm_verdicts = df_verified['llm_verdict'].fillna(-1).astype(int)
        
        agreement = (vae_verdicts == llm_verdicts).sum()
        disagreement = (vae_verdicts != llm_verdicts).sum()
        
        labels = ['Agreement', 'Disagreement']
        sizes = [agreement, disagreement]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.1, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('VAE vs LLM Verdict Agreement', fontweight='bold', fontsize=12)
    
    # Plot 3c: Correctness by model
    ax = axes[1, 0]
    
    if len(df_verified) > 0:
        correctness = df_verified.groupby('model').apply(
            lambda x: {
                'Correct': (x['consensus_anomalous'] == False).sum(),
                'Anomalous': (x['consensus_anomalous'] == True).sum(),
                'Uncertain': (x['consensus_anomalous'].isna()).sum()
            }
        )
        
        correctness_df = pd.DataFrame(correctness.tolist(), index=correctness.index)
        correctness_pct = correctness_df.div(correctness_df.sum(axis=1), axis=0) * 100
        
        correctness_pct.plot(kind='bar', stacked=True, ax=ax, 
                            color=['#2ecc71', '#e74c3c', '#95a5a6'],
                            alpha=0.8)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('Correctness Distribution by Model', fontweight='bold', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Verdict', frameon=True)
        ax.set_ylim([0, 100])
    
    # Plot 3d: Correctness by complexity
    ax = axes[1, 1]
    
    if len(df_verified) > 0:
        correctness_complexity = df_verified.groupby(['complexity', 'model']).apply(
            lambda x: (x['consensus_anomalous'] == False).sum() / len(x) * 100
        ).unstack()
        
        correctness_complexity = correctness_complexity.reindex(['simple', 'medium', 'complex'])
        correctness_complexity.plot(kind='bar', ax=ax, alpha=0.8, color=['#2E86AB', '#A23B72'])
        
        ax.set_xlabel('Circuit Complexity', fontweight='bold')
        ax.set_ylabel('Correctness Rate (%)', fontweight='bold')
        ax.set_title('Correctness by Complexity', fontweight='bold', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Model', frameon=True)
        ax.set_ylim([0, 105])
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f'verification_metrics.{fmt}', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: verification_metrics.[png/pdf/svg]")
    plt.close()

def plot_consistency_analysis(df: pd.DataFrame, output_dir: Path):
    """Plot 4: Consistency analysis across multiple runs"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by circuit and model
    grouped = df.groupby(['circuit_name', 'model'])
    
    # Calculate consistency metrics
    consistency_data = []
    
    for (circuit, model), group in grouped:
        if len(group) < 3:
            continue
        
        success_rate = group['generation_success'].mean()
        success_std = group['generation_success'].std()
        
        # For circuits that succeeded
        succeeded = group[group['simulation_success'] == True]
        if len(succeeded) >= 3:
            # Consistency in outcomes
            outcomes = succeeded['consensus_anomalous'].dropna()
            if len(outcomes) > 0:
                outcome_consistency = 1 - outcomes.std()  # Higher = more consistent
            else:
                outcome_consistency = np.nan
        else:
            outcome_consistency = np.nan
        
        consistency_data.append({
            'circuit': circuit,
            'model': model,
            'complexity': group['complexity'].iloc[0],
            'runs': len(group),
            'success_rate': success_rate,
            'success_std': success_std,
            'outcome_consistency': outcome_consistency
        })
    
    cons_df = pd.DataFrame(consistency_data)
    
    # Plot 4a: Success rate variability by model
    ax = axes[0]
    
    for model in cons_df['model'].unique():
        model_data = cons_df[cons_df['model'] == model]
        ax.scatter(model_data['success_rate'], model_data['success_std'],
                  alpha=0.6, s=100, label=model)
    
    ax.set_xlabel('Mean Success Rate', fontweight='bold')
    ax.set_ylabel('Standard Deviation', fontweight='bold')
    ax.set_title('Generation Consistency\n(Lower SD = More Consistent)', 
                fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add quadrant lines
    ax.axhline(y=cons_df['success_std'].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4b: Consistency by complexity
    ax = axes[1]
    
    cons_by_complexity = cons_df.groupby(['complexity', 'model'])['success_std'].mean().unstack()
    cons_by_complexity = cons_by_complexity.reindex(['simple', 'medium', 'complex'])
    
    cons_by_complexity.plot(kind='bar', ax=ax, alpha=0.8, color=['#2E86AB', '#A23B72'])
    
    ax.set_xlabel('Circuit Complexity', fontweight='bold')
    ax.set_ylabel('Average Standard Deviation', fontweight='bold')
    ax.set_title('Consistency by Complexity\n(Lower = More Consistent)', 
                fontweight='bold', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Model', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f'consistency_analysis.{fmt}', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: consistency_analysis.[png/pdf/svg]")
    plt.close()

def plot_timing_analysis(df: pd.DataFrame, output_dir: Path):
    """Plot 5: Timing analysis"""
    
    df_success = df[df['generation_success'] == True].copy()
    
    if len(df_success) == 0:
        print("⚠️  No successful generations to plot timing")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 5a: Generation time by model and complexity
    ax = axes[0]
    
    timing_by_complexity = df_success.groupby(['complexity', 'model'])['generation_time'].mean().unstack()
    timing_by_complexity = timing_by_complexity.reindex(['simple', 'medium', 'complex'])
    
    timing_by_complexity.plot(kind='bar', ax=ax, alpha=0.8, color=['#2E86AB', '#A23B72'])
    
    ax.set_xlabel('Circuit Complexity', fontweight='bold')
    ax.set_ylabel('Average Generation Time (s)', fontweight='bold')
    ax.set_title('Generation Time by Complexity', fontweight='bold', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Model', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Plot 5b: Simulation time distribution
    ax = axes[1]
    
    df_simulated = df_success[df_success['simulation_success'] == True]
    
    if len(df_simulated) > 0:
        for model in df_simulated['model'].unique():
            model_data = df_simulated[df_simulated['model'] == model]['simulation_time']
            ax.hist(model_data, bins=15, alpha=0.6, label=model, edgecolor='black')
        
        ax.set_xlabel('Simulation Time (s)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Simulation Time Distribution', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f'timing_analysis.{fmt}', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: timing_analysis.[png/pdf/svg]")
    plt.close()

def generate_summary_statistics(data: Dict, df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table"""
    
    summary = {
        'Overall Statistics': {},
        'By Model': {},
        'By Complexity': {}
    }
    
    # Overall
    summary['Overall Statistics'] = {
        'Total Tests': len(df),
        'Unique Circuits': df['circuit_name'].nunique(),
        'Models Tested': df['model'].nunique(),
        'Runs per Circuit': len(df) // (df['circuit_name'].nunique() * df['model'].nunique()),
        'Overall Success Rate': f"{df['generation_success'].mean()*100:.1f}%"
    }
    
    # By Model
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        successful = model_data[model_data['generation_success'] == True]
        verified = successful[successful['simulation_success'] == True]
        
        summary['By Model'][model] = {
            'Generation Success': f"{model_data['generation_success'].mean()*100:.1f}%",
            'Compilation Success': f"{successful['compilation_success'].mean()*100:.1f}%" if len(successful) > 0 else "N/A",
            'Simulation Success': f"{successful['simulation_success'].mean()*100:.1f}%" if len(successful) > 0 else "N/A",
            'Avg LOC': f"{successful['lines_of_code'].mean():.0f}" if len(successful) > 0 else "N/A",
            'Testbench Rate': f"{successful['has_testbench'].mean()*100:.1f}%" if len(successful) > 0 else "N/A",
            'Correctness Rate': f"{(verified['consensus_anomalous'] == False).mean()*100:.1f}%" if len(verified) > 0 else "N/A"
        }
    
    # By Complexity
    for complexity in ['simple', 'medium', 'complex']:
        comp_data = df[df['complexity'] == complexity]
        if len(comp_data) > 0:
            successful = comp_data[comp_data['generation_success'] == True]
            verified = successful[successful['simulation_success'] == True]
            
            summary['By Complexity'][complexity] = {
                'Tests': len(comp_data),
                'Success Rate': f"{comp_data['generation_success'].mean()*100:.1f}%",
                'Avg LOC': f"{successful['lines_of_code'].mean():.0f}" if len(successful) > 0 else "N/A",
                'Correctness': f"{(verified['consensus_anomalous'] == False).mean()*100:.1f}%" if len(verified) > 0 else "N/A"
            }
    
    # Save as JSON
    summary_path = output_dir / 'summary_statistics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved: summary_statistics.json")
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for section, stats in summary.items():
        print(f"\n{section}:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

def create_comparison_dashboard(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive dashboard comparing all metrics"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall success rates
    ax1 = fig.add_subplot(gs[0, 0])
    success_by_model = df.groupby('model')['generation_success'].mean() * 100
    bars = ax1.bar(range(len(success_by_model)), success_by_model.values, 
                   alpha=0.8, color=['#2E86AB', '#A23B72'])
    ax1.set_title('Generation Success', fontweight='bold', fontsize=10)
    ax1.set_xticks(range(len(success_by_model)))
    ax1.set_xticklabels(success_by_model.index, fontsize=8)
    ax1.set_ylabel('%', fontsize=8)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Code quality
    ax2 = fig.add_subplot(gs[0, 1])
    df_success = df[df['generation_success'] == True]
    if len(df_success) > 0:
        loc_by_model = df_success.groupby('model')['lines_of_code'].mean()
        ax2.bar(range(len(loc_by_model)), loc_by_model.values,
               alpha=0.8, color=['#2E86AB', '#A23B72'])
        ax2.set_title('Avg Lines of Code', fontweight='bold', fontsize=10)
        ax2.set_xticks(range(len(loc_by_model)))
        ax2.set_xticklabels(loc_by_model.index, fontsize=8)
        ax2.set_ylabel('LOC', fontsize=8)
    
    # 3. Correctness
    ax3 = fig.add_subplot(gs[0, 2])
    df_verified = df_success[df_success['simulation_success'] == True]
    if len(df_verified) > 0:
        correctness = df_verified.groupby('model').apply(
            lambda x: (x['consensus_anomalous'] == False).sum() / len(x) * 100
        )
        bars = ax3.bar(range(len(correctness)), correctness.values,
                      alpha=0.8, color=['#2E86AB', '#A23B72'])
        ax3.set_title('Correctness Rate', fontweight='bold', fontsize=10)
        ax3.set_xticks(range(len(correctness)))
        ax3.set_xticklabels(correctness.index, fontsize=8)
        ax3.set_ylabel('%', fontsize=8)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # 4-6. By Complexity (second row)
    for idx, complexity in enumerate(['simple', 'medium', 'complex']):
        ax = fig.add_subplot(gs[1, idx])
        comp_data = df[df['complexity'] == complexity]
        
        if len(comp_data) > 0:
            metrics = comp_data.groupby('model').agg({
                'generation_success': 'mean',
                'compilation_success': 'mean',
                'simulation_success': 'mean'
            }) * 100
            
            metrics.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
            ax.set_title(f'{complexity.capitalize()} Circuits', fontweight='bold', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('%', fontsize=8)
            ax.legend(['Gen', 'Comp', 'Sim'], fontsize=7, loc='lower right')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_ylim([0, 105])
    
    # 7. Pipeline view (third row, spans 2 columns)
    ax7 = fig.add_subplot(gs[2, :2])
    stages = ['Generated', 'Compiled', 'Simulated', 'Correct']
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        generated = len(model_data[model_data['generation_success'] == True])
        compiled = model_data['compilation_success'].sum()
        simulated = model_data['simulation_success'].sum()
        verified = model_data[model_data['simulation_success'] == True]
        correct = (verified['consensus_anomalous'] == False).sum()
        
        counts = [generated, compiled, simulated, correct]
        rates = [c / len(model_data) * 100 for c in counts]
        
        ax7.plot(stages, rates, marker='o', linewidth=2, markersize=8, label=model)
    
    ax7.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=10)
    ax7.set_title('Full Pipeline: Generation → Verification', fontweight='bold', fontsize=11)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 105])
    
    # 8. Timing comparison (third row, right column)
    ax8 = fig.add_subplot(gs[2, 2])
    if len(df_success) > 0:
        timing = df_success.groupby('model')['generation_time'].mean()
        ax8.barh(range(len(timing)), timing.values, alpha=0.8, color=['#2E86AB', '#A23B72'])
        ax8.set_title('Avg Generation Time', fontweight='bold', fontsize=10)
        ax8.set_yticks(range(len(timing)))
        ax8.set_yticklabels(timing.index, fontsize=8)
        ax8.set_xlabel('Seconds', fontsize=8)
    
    plt.suptitle('Multi-Modal Test Results Dashboard', fontweight='bold', fontsize=16, y=0.995)
    
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f'comparison_dashboard.{fmt}', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved: comparison_dashboard.[png/pdf/svg]")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Generate plots and analysis from test results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', type=str, help='Path to results JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots (default: same as input)')
    parser.add_argument('--format', type=str, default='all', 
                       choices=['png', 'pdf', 'svg', 'all'],
                       help='Output format (default: all)')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MULTI-MODAL TEST RESULTS VISUALIZATION")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load results
    data, df = load_results(input_path)
    
    # Generate all plots
    print("\n📊 Generating plots...")
    
    print("\n1. Generation Success Rates...")
    plot_generation_success_rates(df, output_dir)
    
    print("\n2. Code Quality Metrics...")
    plot_code_quality_metrics(df, output_dir)
    
    print("\n3. Verification Metrics...")
    plot_verification_metrics(df, output_dir)
    
    print("\n4. Consistency Analysis...")
    plot_consistency_analysis(df, output_dir)
    
    print("\n5. Timing Analysis...")
    plot_timing_analysis(df, output_dir)
    
    print("\n6. Comparison Dashboard...")
    create_comparison_dashboard(df, output_dir)
    
    print("\n7. Summary Statistics...")
    generate_summary_statistics(data, df, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Generated {len(list(output_dir.glob('*.png')))} PNG files")
    print(f"📄 Generated {len(list(output_dir.glob('*.pdf')))} PDF files")
    print(f"🎨 Generated {len(list(output_dir.glob('*.svg')))} SVG files")
    print(f"📋 Generated summary_statistics.json")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()