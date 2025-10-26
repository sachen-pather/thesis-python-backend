"""
Advanced Statistical Analysis for Verilog Generation Results
Performs detailed statistical tests and correlation analysis

SAVE AS: tests/analysis/advanced_statistical_analysis.py
RUN FROM: tests/analysis/ directory
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
from typing import Dict, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = ROOT_DIR / "thesis_generation_results"
OUTPUT_DIR = RESULTS_DIR / "analysis_plots"

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {'claude': '#7B68EE', 'gpt-4o': '#FF6B6B'}
DPI = 300

def load_latest_results() -> pd.DataFrame:
    """Load most recent results"""
    json_files = list((RESULTS_DIR / "comprehensive").glob("comprehensive_results_*.json"))
    if not json_files:
        print("❌ No results found!")
        sys.exit(1)
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {latest_file.name}\n")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data['results'])

def chi_square_test(df: pd.DataFrame) -> Dict:
    """Chi-square test for model independence"""
    print("🔬 Chi-Square Test for Model Independence")
    print("-" * 60)
    
    results = {}
    
    # Overall correctness
    contingency = pd.crosstab(
        df['generator_model'],
        df['consensus_anomalous'].fillna(True)  # Treat None as anomalous
    )
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    results['overall'] = {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'significant': p_value < 0.05
    }
    
    print(f"Overall Functional Correctness:")
    print(f"  Chi² = {chi2:.4f}, p-value = {p_value:.4f}")
    print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05\n")
    
    # By complexity
    for complexity in ['medium', 'complex']:
        subset = df[df['complexity'] == complexity]
        contingency = pd.crosstab(
            subset['generator_model'],
            subset['consensus_anomalous'].fillna(True)
        )
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        results[complexity] = {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < 0.05
        }
        
        print(f"{complexity.capitalize()} Circuits:")
        print(f"  Chi² = {chi2:.4f}, p-value = {p_value:.4f}")
        print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05\n")
    
    return results

def mann_whitney_test(df: pd.DataFrame) -> Dict:
    """Mann-Whitney U test for continuous metrics"""
    print("🔬 Mann-Whitney U Tests (Non-parametric)")
    print("-" * 60)
    
    results = {}
    
    metrics = [
        ('lines_of_code', 'Lines of Code'),
        ('generation_time', 'Generation Time'),
        ('simulation_time', 'Simulation Time'),
        ('consensus_confidence', 'Consensus Confidence')
    ]
    
    for metric, label in metrics:
        claude_data = df[df['generator_model'] == 'claude'][metric].dropna()
        gpt4o_data = df[df['generator_model'] == 'gpt-4o'][metric].dropna()
        
        u_stat, p_value = mannwhitneyu(claude_data, gpt4o_data, alternative='two-sided')
        
        results[metric] = {
            'u_statistic': u_stat,
            'p_value': p_value,
            'claude_median': claude_data.median(),
            'gpt4o_median': gpt4o_data.median(),
            'significant': p_value < 0.05
        }
        
        print(f"{label}:")
        print(f"  Claude median: {claude_data.median():.2f}")
        print(f"  GPT-4o median: {gpt4o_data.median():.2f}")
        print(f"  U = {u_stat:.0f}, p-value = {p_value:.4f}")
        print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05\n")
    
    return results

def correlation_analysis(df: pd.DataFrame):
    """Correlation between code metrics and correctness"""
    print("🔬 Correlation Analysis")
    print("-" * 60)
    
    # Convert boolean to numeric
    df_numeric = df.copy()
    df_numeric['is_correct'] = (~df_numeric['consensus_anomalous'].fillna(True)).astype(int)
    
    metrics = ['lines_of_code', 'generation_time', 'simulation_time', 
               'has_testbench', 'consensus_confidence']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, model in enumerate(['claude', 'gpt-4o']):
        model_df = df_numeric[df_numeric['generator_model'] == model]
        
        # Calculate correlations
        correlations = []
        for metric in metrics:
            if metric == 'has_testbench':
                corr = model_df[[metric, 'is_correct']].astype(float).corr().iloc[0, 1]
            else:
                corr = model_df[[metric, 'is_correct']].corr().iloc[0, 1]
            correlations.append(corr)
        
        # Plot
        y_pos = np.arange(len(metrics))
        colors_bar = [COLORS[model] if c >= 0 else '#FF4444' for c in correlations]
        
        axes[idx].barh(y_pos, correlations, color=colors_bar, alpha=0.7, edgecolor='black')
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
        axes[idx].set_xlabel('Correlation with Correctness', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{"Claude Sonnet 3.5" if model == "claude" else "GPT-4o"}',
                           fontsize=12, fontweight='bold')
        axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[idx].set_xlim(-0.5, 0.5)
        axes[idx].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(correlations):
            x_pos = v + (0.02 if v >= 0 else -0.02)
            ha = 'left' if v >= 0 else 'right'
            axes[idx].text(x_pos, i, f'{v:.3f}', va='center', ha=ha, 
                          fontsize=9, fontweight='bold')
        
        # Print to console
        model_name = "Claude Sonnet 3.5" if model == "claude" else "GPT-4o"
        print(f"\n{model_name}:")
        for metric, corr in zip(metrics, correlations):
            print(f"  {metric}: {corr:.3f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_correlation_analysis.png', dpi=DPI, bbox_inches='tight')
    print(f"\n✅ Saved: 09_correlation_analysis.png")
    plt.close()

def effect_size_analysis(df: pd.DataFrame):
    """Calculate effect sizes for key comparisons"""
    print("\n🔬 Effect Size Analysis (Cohen's h)")
    print("-" * 60)
    
    def cohens_h(p1: float, p2: float) -> float:
        """Cohen's h for proportions"""
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        return phi1 - phi2
    
    # Overall correctness
    claude_correct = (~df[df['generator_model'] == 'claude']['consensus_anomalous'].fillna(True)).mean()
    gpt4o_correct = (~df[df['generator_model'] == 'gpt-4o']['consensus_anomalous'].fillna(True)).mean()
    
    h_overall = cohens_h(claude_correct, gpt4o_correct)
    
    print(f"\nOverall Correctness:")
    print(f"  Claude: {claude_correct:.3f}, GPT-4o: {gpt4o_correct:.3f}")
    print(f"  Cohen's h = {h_overall:.3f}")
    print(f"  Effect size: {interpret_cohens_h(h_overall)}")
    
    # By complexity
    for complexity in ['medium', 'complex']:
        subset = df[df['complexity'] == complexity]
        claude_correct = (~subset[subset['generator_model'] == 'claude']['consensus_anomalous'].fillna(True)).mean()
        gpt4o_correct = (~subset[subset['generator_model'] == 'gpt-4o']['consensus_anomalous'].fillna(True)).mean()
        
        h = cohens_h(claude_correct, gpt4o_correct)
        
        print(f"\n{complexity.capitalize()} Circuits:")
        print(f"  Claude: {claude_correct:.3f}, GPT-4o: {gpt4o_correct:.3f}")
        print(f"  Cohen's h = {h:.3f}")
        print(f"  Effect size: {interpret_cohens_h(h)}")

def interpret_cohens_h(h: float) -> str:
    """Interpret Cohen's h effect size"""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "Small"
    elif abs_h < 0.5:
        return "Medium"
    else:
        return "Large"

def failure_mode_analysis(df: pd.DataFrame):
    """Analyze where and why circuits fail"""
    print("\n🔬 Failure Mode Analysis")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, model in enumerate(['claude', 'gpt-4o']):
        model_df = df[df['generator_model'] == model]
        
        # Failure stages
        failure_stages = {
            'Generation Failed': (~model_df['generation_success']).sum(),
            'Compilation Failed': (model_df['generation_success'] & 
                                  ~model_df['compilation_success']).sum(),
            'Simulation Failed': (model_df['compilation_success'] & 
                                 ~model_df['simulation_success']).sum(),
            'Functional Bug': (model_df['simulation_success'] & 
                              model_df['consensus_anomalous'].fillna(True)).sum()
        }
        
        # Pie chart of failure modes
        colors_pie = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90']
        axes[idx, 0].pie(failure_stages.values(), labels=failure_stages.keys(),
                        autopct='%1.1f%%', colors=colors_pie, startangle=90)
        axes[idx, 0].set_title(f'{"Claude Sonnet 3.5" if model == "claude" else "GPT-4o"} - Failure Modes',
                              fontsize=12, fontweight='bold')
        
        # Failure by category
        categories = model_df['category'].unique()
        failure_by_cat = []
        
        for cat in sorted(categories):
            cat_df = model_df[model_df['category'] == cat]
            failures = cat_df['consensus_anomalous'].fillna(True).sum()
            total = len(cat_df)
            failure_by_cat.append((cat, failures, total))
        
        failure_by_cat.sort(key=lambda x: x[1]/x[2] if x[2] > 0 else 0, reverse=True)
        
        cats = [x[0] for x in failure_by_cat]
        failure_rates = [(x[1]/x[2]*100) if x[2] > 0 else 0 for x in failure_by_cat]
        
        axes[idx, 1].barh(range(len(cats)), failure_rates, color=COLORS[model], 
                         alpha=0.7, edgecolor='black')
        axes[idx, 1].set_yticks(range(len(cats)))
        axes[idx, 1].set_yticklabels(cats, fontsize=9)
        axes[idx, 1].set_xlabel('Failure Rate (%)', fontsize=10, fontweight='bold')
        axes[idx, 1].set_title(f'Failures by Category', fontsize=11, fontweight='bold')
        axes[idx, 1].grid(axis='x', alpha=0.3)
        
        # Print to console
        model_name = "Claude Sonnet 3.5" if model == "claude" else "GPT-4o"
        print(f"\n{model_name} Failure Breakdown:")
        for stage, count in failure_stages.items():
            pct = count / len(model_df) * 100
            print(f"  {stage}: {count} ({pct:.1f}%)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_failure_analysis.png', dpi=DPI, bbox_inches='tight')
    print(f"\n✅ Saved: 10_failure_analysis.png")
    plt.close()

def generate_latex_table(df: pd.DataFrame):
    """Generate LaTeX table for thesis"""
    print("\n📝 Generating LaTeX Tables")
    print("-" * 60)
    
    latex_path = OUTPUT_DIR / 'results_table.tex'
    
    with open(latex_path, 'w') as f:
        f.write("% Main Results Table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of LLM Verilog Generation Performance}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Claude Sonnet 3.5} & \\textbf{GPT-4o} \\\\\n")
        f.write("\\hline\n")
        
        # Calculate metrics
        for model in ['claude', 'gpt-4o']:
            model_df = df[df['generator_model'] == model]
        
        claude_df = df[df['generator_model'] == 'claude']
        gpt4o_df = df[df['generator_model'] == 'gpt-4o']
        
        metrics = [
            ('Functional Correctness', 
             f"{(~claude_df['consensus_anomalous'].fillna(True)).sum()}/{len(claude_df)}",
             f"{(~gpt4o_df['consensus_anomalous'].fillna(True)).sum()}/{len(gpt4o_df)}"),
            ('Generation Success Rate',
             f"{claude_df['generation_success'].mean()*100:.1f}\\%",
             f"{gpt4o_df['generation_success'].mean()*100:.1f}\\%"),
            ('Simulation Success Rate',
             f"{claude_df['simulation_success'].mean()*100:.1f}\\%",
             f"{gpt4o_df['simulation_success'].mean()*100:.1f}\\%"),
            ('Avg. Lines of Code',
             f"{claude_df['lines_of_code'].mean():.1f}",
             f"{gpt4o_df['lines_of_code'].mean():.1f}"),
            ('Testbench Inclusion',
             f"{claude_df['has_testbench'].mean()*100:.1f}\\%",
             f"{gpt4o_df['has_testbench'].mean()*100:.1f}\\%"),
        ]
        
        for metric, claude_val, gpt4o_val in metrics:
            f.write(f"{metric} & {claude_val} & {gpt4o_val} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ Saved: results_table.tex")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🔬 ADVANCED STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_latest_results()
    
    # Statistical tests
    chi_results = chi_square_test(df)
    mw_results = mann_whitney_test(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Effect sizes
    effect_size_analysis(df)
    
    # Failure analysis
    failure_mode_analysis(df)
    
    # LaTeX table
    generate_latex_table(df)
    
    print("\n" + "="*80)
    print("✅ Advanced analysis complete!")
    print(f"   All outputs in: {OUTPUT_DIR.relative_to(ROOT_DIR)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()