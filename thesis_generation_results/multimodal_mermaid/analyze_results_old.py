#!/usr/bin/env python3
"""
Multimodal Mermaid Test Results Analysis Script with Visualizations
Analyzes Verilog generation results and generates publication-quality plots
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

# Set publication-quality parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['figure.dpi'] = 300

# Color scheme
COLORS = {
    'prompt_only': '#2E86AB',      # Blue
    'mermaid_only': '#A23B72',     # Purple
    'combined': '#F18F01',          # Orange
    'claude': '#4ECDC4',            # Teal
    'gpt-4o': '#FF6B6B',            # Red/Coral
    'simple': '#06D6A0',            # Green
    'medium': '#FFD166',            # Yellow
    'complex': '#EF476F'            # Pink
}

def load_json_files(directory: str = ".") -> List[Dict]:
    """Load all JSON files from the specified directory"""
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    json_files.append(data)
                    print(f"✓ Loaded: {filename}")
            except json.JSONDecodeError as e:
                print(f"✗ Error loading {filename}: {e}")
            except Exception as e:
                print(f"✗ Unexpected error with {filename}: {e}")
    
    return json_files

def extract_all_results(json_files: List[Dict]) -> List[Dict]:
    """Extract all circuit results from all JSON files"""
    all_results = []
    for data in json_files:
        if "results" in data:
            all_results.extend(data["results"])
    return all_results

def count_circuits_tested(results: List[Dict]) -> Dict[str, int]:
    """Count the number of circuits tested"""
    stats = {
        "total_circuits": len(results),
        "by_complexity": defaultdict(int),
        "by_category": defaultdict(int),
        "by_model": defaultdict(int)
    }
    
    for result in results:
        complexity = result.get("complexity", "unknown")
        category = result.get("category", "unknown")
        model = result.get("model", "unknown")
        
        stats["by_complexity"][complexity] += 1
        stats["by_category"][category] += 1
        stats["by_model"][model] += 1
    
    return dict(stats)

def calculate_functional_correctness(results: List[Dict]) -> Dict[str, Any]:
    """Calculate functional correctness by model and complexity"""
    correctness = {
        "claude": {"simple": [], "medium": [], "complex": []},
        "gpt-4o": {"simple": [], "medium": [], "complex": []}
    }
    
    for result in results:
        model = result.get("model", "unknown")
        complexity = result.get("complexity", "unknown")
        
        if model not in ["claude", "gpt-4o"] or complexity not in ["simple", "medium", "complex"]:
            continue
        
        # Check prompt_only approach
        prompt_correct = (
            result.get("prompt_only_compilation_success", False) and
            result.get("prompt_only_simulation_success", False) and
            not result.get("prompt_only_consensus_anomalous", True)
        )
        
        # Check mermaid_only approach
        mermaid_correct = (
            result.get("mermaid_only_compilation_success", False) and
            result.get("mermaid_only_simulation_success", False) and
            not result.get("mermaid_only_consensus_anomalous", True)
        )
        
        # Check combined approach
        combined_correct = (
            result.get("combined_compilation_success", False) and
            result.get("combined_simulation_success", False)
        )
        
        correctness[model][complexity].append({
            "circuit_name": result.get("circuit_name", "unknown"),
            "prompt_only": prompt_correct,
            "mermaid_only": mermaid_correct,
            "combined": combined_correct
        })
    
    # Calculate statistics
    stats = {
        "claude": {},
        "gpt-4o": {}
    }
    
    for model in ["claude", "gpt-4o"]:
        for complexity in ["simple", "medium", "complex"]:
            circuits = correctness[model][complexity]
            total = len(circuits)
            
            if total > 0:
                prompt_correct = sum(1 for c in circuits if c["prompt_only"])
                mermaid_correct = sum(1 for c in circuits if c["mermaid_only"])
                combined_correct = sum(1 for c in circuits if c["combined"])
                
                stats[model][complexity] = {
                    "total_circuits": total,
                    "prompt_only": {
                        "correct": prompt_correct,
                        "accuracy": (prompt_correct/total)*100
                    },
                    "mermaid_only": {
                        "correct": mermaid_correct,
                        "accuracy": (mermaid_correct/total)*100
                    },
                    "combined": {
                        "correct": combined_correct,
                        "accuracy": (combined_correct/total)*100
                    }
                }
            else:
                stats[model][complexity] = {
                    "total_circuits": 0,
                    "prompt_only": {"correct": 0, "accuracy": 0},
                    "mermaid_only": {"correct": 0, "accuracy": 0},
                    "combined": {"correct": 0, "accuracy": 0}
                }
    
    return {
        "detailed": correctness,
        "statistics": stats
    }

def compare_approaches_accuracy(results: List[Dict]) -> Dict[str, Any]:
    """Determine accuracy of different approaches"""
    approach_stats = {
        "overall": {
            "prompt_only": {"correct": 0, "total": 0},
            "mermaid_only": {"correct": 0, "total": 0},
            "combined": {"correct": 0, "total": 0}
        },
        "by_model": {
            "claude": {
                "prompt_only": {"correct": 0, "total": 0},
                "mermaid_only": {"correct": 0, "total": 0},
                "combined": {"correct": 0, "total": 0}
            },
            "gpt-4o": {
                "prompt_only": {"correct": 0, "total": 0},
                "mermaid_only": {"correct": 0, "total": 0},
                "combined": {"correct": 0, "total": 0}
            }
        }
    }
    
    for result in results:
        model = result.get("model", "unknown")
        
        # Prompt only
        prompt_correct = (
            result.get("prompt_only_compilation_success", False) and
            result.get("prompt_only_simulation_success", False) and
            not result.get("prompt_only_consensus_anomalous", True)
        )
        approach_stats["overall"]["prompt_only"]["total"] += 1
        approach_stats["overall"]["prompt_only"]["correct"] += int(prompt_correct)
        
        if model in ["claude", "gpt-4o"]:
            approach_stats["by_model"][model]["prompt_only"]["total"] += 1
            approach_stats["by_model"][model]["prompt_only"]["correct"] += int(prompt_correct)
        
        # Mermaid only
        mermaid_correct = (
            result.get("mermaid_only_compilation_success", False) and
            result.get("mermaid_only_simulation_success", False) and
            not result.get("mermaid_only_consensus_anomalous", True)
        )
        approach_stats["overall"]["mermaid_only"]["total"] += 1
        approach_stats["overall"]["mermaid_only"]["correct"] += int(mermaid_correct)
        
        if model in ["claude", "gpt-4o"]:
            approach_stats["by_model"][model]["mermaid_only"]["total"] += 1
            approach_stats["by_model"][model]["mermaid_only"]["correct"] += int(mermaid_correct)
        
        # Combined
        combined_correct = (
            result.get("combined_compilation_success", False) and
            result.get("combined_simulation_success", False)
        )
        approach_stats["overall"]["combined"]["total"] += 1
        approach_stats["overall"]["combined"]["correct"] += int(combined_correct)
        
        if model in ["claude", "gpt-4o"]:
            approach_stats["by_model"][model]["combined"]["total"] += 1
            approach_stats["by_model"][model]["combined"]["correct"] += int(combined_correct)
    
    # Calculate percentages
    def add_percentages(stats_dict):
        for approach, data in stats_dict.items():
            if data["total"] > 0:
                data["accuracy"] = (data['correct']/data['total'])*100
            else:
                data["accuracy"] = 0
        return stats_dict
    
    approach_stats["overall"] = add_percentages(approach_stats["overall"])
    for model in ["claude", "gpt-4o"]:
        approach_stats["by_model"][model] = add_percentages(approach_stats["by_model"][model])
    
    return approach_stats

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_rq1a_multimodal_strategy(approach_accuracy: Dict, output_dir: str = "."):
    """
    RQ1a: Optimal multimodal input strategy comparison
    Bar chart comparing Prompt Only, Mermaid Only, and Combined approaches
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
    overall_accuracies = [
        approach_accuracy["overall"]["prompt_only"]["accuracy"],
        approach_accuracy["overall"]["mermaid_only"]["accuracy"],
        approach_accuracy["overall"]["combined"]["accuracy"]
    ]
    
    colors = [COLORS['prompt_only'], COLORS['mermaid_only'], COLORS['combined']]
    
    bars = ax.bar(approaches, overall_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, overall_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Functional Correctness (%)', fontweight='bold')
    ax.set_title('RQ1a: Multimodal Input Strategy Comparison', fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'RQ1a_multimodal_strategy_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_rq1b_complexity_scaling(correctness_stats: Dict, output_dir: str = "."):
    """
    RQ1b: Performance scaling with circuit complexity
    Line plot showing accuracy degradation across complexity levels
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    complexities = ['Simple', 'Medium', 'Complex']
    approaches = ['prompt_only', 'mermaid_only', 'combined']
    approach_labels = ['Prompt Only', 'Mermaid Only', 'Combined']
    
    # Aggregate across both models
    approach_data = {app: {'simple': [], 'medium': [], 'complex': []} for app in approaches}
    
    for model in ['claude', 'gpt-4o']:
        for complexity in ['simple', 'medium', 'complex']:
            stats = correctness_stats[model][complexity]
            for approach in approaches:
                approach_data[approach][complexity].append(stats[approach]['accuracy'])
    
    # Calculate averages
    for approach in approaches:
        accuracies = [
            np.mean(approach_data[approach]['simple']),
            np.mean(approach_data[approach]['medium']),
            np.mean(approach_data[approach]['complex'])
        ]
        
        color = COLORS[approach]
        ax.plot(complexities, accuracies, marker='o', linewidth=2.5, 
                markersize=10, label=approach_labels[approaches.index(approach)],
                color=color)
        
        # Add value labels
        for x, y in zip(complexities, accuracies):
            ax.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color=color)
    
    ax.set_ylabel('Functional Correctness (%)', fontweight='bold')
    ax.set_xlabel('Circuit Complexity', fontweight='bold')
    ax.set_title('RQ1b: Performance Scaling with Circuit Complexity', fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'RQ1b_complexity_scaling.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_rq2_model_comparison(correctness_stats: Dict, approach_accuracy: Dict, output_dir: str = "."):
    """
    RQ2: LLM model comparison (Claude vs GPT-4o)
    Grouped bar chart comparing both models across approaches
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
    claude_accuracies = [
        approach_accuracy["by_model"]["claude"]["prompt_only"]["accuracy"],
        approach_accuracy["by_model"]["claude"]["mermaid_only"]["accuracy"],
        approach_accuracy["by_model"]["claude"]["combined"]["accuracy"]
    ]
    gpt_accuracies = [
        approach_accuracy["by_model"]["gpt-4o"]["prompt_only"]["accuracy"],
        approach_accuracy["by_model"]["gpt-4o"]["mermaid_only"]["accuracy"],
        approach_accuracy["by_model"]["gpt-4o"]["combined"]["accuracy"]
    ]
    
    x = np.arange(len(approaches))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, claude_accuracies, width, label='Claude 3.5 Sonnet',
                   color=COLORS['claude'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, gpt_accuracies, width, label='GPT-4o',
                   color=COLORS['gpt-4o'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Functional Correctness (%)', fontweight='bold')
    ax.set_xlabel('Input Strategy', fontweight='bold')
    ax.set_title('RQ2: Claude 3.5 Sonnet vs GPT-4o Performance', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'RQ2_model_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_comprehensive_heatmap(correctness_stats: Dict, output_dir: str = "."):
    """
    Comprehensive heatmap showing all combinations of model, complexity, and approach
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    complexities = ['Simple', 'Medium', 'Complex']
    approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
    
    # Claude heatmap
    claude_data = []
    for complexity in ['simple', 'medium', 'complex']:
        row = [
            correctness_stats['claude'][complexity]['prompt_only']['accuracy'],
            correctness_stats['claude'][complexity]['mermaid_only']['accuracy'],
            correctness_stats['claude'][complexity]['combined']['accuracy']
        ]
        claude_data.append(row)
    
    im1 = ax1.imshow(claude_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(np.arange(len(approaches)))
    ax1.set_yticks(np.arange(len(complexities)))
    ax1.set_xticklabels(approaches)
    ax1.set_yticklabels(complexities)
    ax1.set_title('Claude 3.5 Sonnet', fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(complexities)):
        for j in range(len(approaches)):
            text = ax1.text(j, i, f'{claude_data[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=10)
    
    # GPT-4o heatmap
    gpt_data = []
    for complexity in ['simple', 'medium', 'complex']:
        row = [
            correctness_stats['gpt-4o'][complexity]['prompt_only']['accuracy'],
            correctness_stats['gpt-4o'][complexity]['mermaid_only']['accuracy'],
            correctness_stats['gpt-4o'][complexity]['combined']['accuracy']
        ]
        gpt_data.append(row)
    
    im2 = ax2.imshow(gpt_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(np.arange(len(approaches)))
    ax2.set_yticks(np.arange(len(complexities)))
    ax2.set_xticklabels(approaches)
    ax2.set_yticklabels(complexities)
    ax2.set_title('GPT-4o', fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(complexities)):
        for j in range(len(approaches)):
            text = ax2.text(j, i, f'{gpt_data[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Functional Correctness (%)', fontweight='bold')
    
    fig.suptitle('Detailed Performance Heatmap: Model × Complexity × Strategy', 
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'comprehensive_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_complexity_by_model(correctness_stats: Dict, output_dir: str = "."):
    """
    RQ1b variant: Complexity scaling separated by model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    complexities = ['Simple', 'Medium', 'Complex']
    approaches = ['prompt_only', 'mermaid_only', 'combined']
    approach_labels = ['Prompt Only', 'Mermaid Only', 'Combined']
    approach_colors = [COLORS['prompt_only'], COLORS['mermaid_only'], COLORS['combined']]
    
    # Claude subplot
    for idx, approach in enumerate(approaches):
        accuracies = [
            correctness_stats['claude']['simple'][approach]['accuracy'],
            correctness_stats['claude']['medium'][approach]['accuracy'],
            correctness_stats['claude']['complex'][approach]['accuracy']
        ]
        ax1.plot(complexities, accuracies, marker='o', linewidth=2.5, 
                markersize=8, label=approach_labels[idx], color=approach_colors[idx])
        
        # Add value labels
        for x, y in zip(complexities, accuracies):
            ax1.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', 
                   fontsize=8, color=approach_colors[idx])
    
    ax1.set_ylabel('Functional Correctness (%)', fontweight='bold')
    ax1.set_xlabel('Circuit Complexity', fontweight='bold')
    ax1.set_title('Claude 3.5 Sonnet', fontweight='bold', pad=15)
    ax1.set_ylim(0, 110)
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # GPT-4o subplot
    for idx, approach in enumerate(approaches):
        accuracies = [
            correctness_stats['gpt-4o']['simple'][approach]['accuracy'],
            correctness_stats['gpt-4o']['medium'][approach]['accuracy'],
            correctness_stats['gpt-4o']['complex'][approach]['accuracy']
        ]
        ax2.plot(complexities, accuracies, marker='o', linewidth=2.5, 
                markersize=8, label=approach_labels[idx], color=approach_colors[idx])
        
        # Add value labels
        for x, y in zip(complexities, accuracies):
            ax2.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', 
                   fontsize=8, color=approach_colors[idx])
    
    ax2.set_ylabel('Functional Correctness (%)', fontweight='bold')
    ax2.set_xlabel('Circuit Complexity', fontweight='bold')
    ax2.set_title('GPT-4o', fontweight='bold', pad=15)
    ax2.set_ylim(0, 110)
    ax2.legend(loc='lower left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle('Performance Scaling with Complexity by Model', 
                 fontweight='bold', fontsize=14, y=1.0)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'complexity_by_model_detailed.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_overall_success_rates(results: List[Dict], output_dir: str = "."):
    """
    Success rate breakdown showing compilation, simulation, and correctness
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    models = ['claude', 'gpt-4o']
    approaches = [('prompt_only', 'Prompt Only'), 
                  ('mermaid_only', 'Mermaid Only'), 
                  ('combined', 'Combined')]
    
    for model_idx, model in enumerate(models):
        for app_idx, (approach, app_label) in enumerate(approaches):
            ax = axes[model_idx, app_idx]
            
            # Calculate metrics
            model_results = [r for r in results if r.get('model') == model]
            total = len(model_results)
            
            if total == 0:
                continue
            
            compiled = sum(1 for r in model_results 
                          if r.get(f'{approach}_compilation_success', False))
            simulated = sum(1 for r in model_results 
                           if r.get(f'{approach}_simulation_success', False))
            
            if approach in ['prompt_only', 'mermaid_only']:
                correct = sum(1 for r in model_results 
                            if r.get(f'{approach}_compilation_success', False) and
                               r.get(f'{approach}_simulation_success', False) and
                               not r.get(f'{approach}_consensus_anomalous', True))
            else:
                correct = sum(1 for r in model_results 
                            if r.get(f'{approach}_compilation_success', False) and
                               r.get(f'{approach}_simulation_success', False))
            
            # Create stacked bar
            categories = ['Compiled', 'Simulated', 'Functionally\nCorrect']
            values = [(compiled/total)*100, (simulated/total)*100, (correct/total)*100]
            colors_bars = ['#3A86FF', '#8338EC', '#06D6A0']
            
            bars = ax.barh(categories, values, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)
            
            ax.set_xlim(0, 105)
            ax.set_xlabel('Success Rate (%)', fontweight='bold')
            
            # Title
            if model_idx == 0:
                ax.set_title(app_label, fontweight='bold', pad=10)
            
            # Y-axis label for first column
            if app_idx == 0:
                model_name = 'Claude 3.5 Sonnet' if model == 'claude' else 'GPT-4o'
                ax.set_ylabel(model_name, fontweight='bold', fontsize=12)
            
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
    
    fig.suptitle('Success Rate Analysis: Compilation → Simulation → Functional Correctness', 
                 fontweight='bold', fontsize=14, y=0.98)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'success_rate_breakdown.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def generate_summary_report(results: List[Dict]) -> str:
    """Generate a comprehensive summary report"""
    
    circuit_stats = count_circuits_tested(results)
    correctness = calculate_functional_correctness(results)
    approach_accuracy = compare_approaches_accuracy(results)
    
    report = []
    report.append("=" * 80)
    report.append("MULTIMODAL MERMAID TEST RESULTS ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    # INSIGHT 1
    report.append("-" * 80)
    report.append("INSIGHT 1: CIRCUIT COUNT AND DISTRIBUTION")
    report.append("-" * 80)
    report.append(f"Total circuits tested: {circuit_stats['total_circuits']}")
    report.append("")
    report.append("By Complexity:")
    for complexity, count in sorted(circuit_stats['by_complexity'].items()):
        report.append(f"  {complexity.capitalize():10s}: {count:3d} circuits")
    report.append("")
    report.append("By Category:")
    for category, count in sorted(circuit_stats['by_category'].items()):
        report.append(f"  {category:15s}: {count:3d} circuits")
    report.append("")
    report.append("By Model:")
    for model, count in sorted(circuit_stats['by_model'].items()):
        report.append(f"  {model:10s}: {count:3d} circuits")
    report.append("")
    
    # INSIGHT 2
    report.append("-" * 80)
    report.append("INSIGHT 2: FUNCTIONAL CORRECTNESS BY MODEL AND COMPLEXITY")
    report.append("-" * 80)
    report.append("Correctness = Compilation Success + Simulation Success + Not Anomalous")
    report.append("")
    
    for model in ["claude", "gpt-4o"]:
        report.append(f"\n{model.upper()}:")
        report.append("-" * 40)
        
        for complexity in ["simple", "medium", "complex"]:
            stats = correctness["statistics"][model][complexity]
            report.append(f"\n  {complexity.upper()} Circuits ({stats['total_circuits']} total):")
            report.append(f"    Prompt Only    : {stats['prompt_only']['correct']:2d}/{stats['total_circuits']:2d} correct ({stats['prompt_only']['accuracy']:.1f}%)")
            report.append(f"    Mermaid Only   : {stats['mermaid_only']['correct']:2d}/{stats['total_circuits']:2d} correct ({stats['mermaid_only']['accuracy']:.1f}%)")
            report.append(f"    Combined       : {stats['combined']['correct']:2d}/{stats['total_circuits']:2d} correct ({stats['combined']['accuracy']:.1f}%)")
    
    report.append("")
    report.append("")
    
    # INSIGHT 3
    report.append("-" * 80)
    report.append("INSIGHT 3: ACCURACY COMPARISON BY APPROACH")
    report.append("-" * 80)
    report.append("")
    
    report.append("OVERALL ACCURACY:")
    report.append("-" * 40)
    overall = approach_accuracy["overall"]
    report.append(f"Prompt Only    : {overall['prompt_only']['correct']:3d}/{overall['prompt_only']['total']:3d} circuits correct ({overall['prompt_only']['accuracy']:.1f}%)")
    report.append(f"Mermaid Only   : {overall['mermaid_only']['correct']:3d}/{overall['mermaid_only']['total']:3d} circuits correct ({overall['mermaid_only']['accuracy']:.1f}%)")
    report.append(f"Combined       : {overall['combined']['correct']:3d}/{overall['combined']['total']:3d} circuits correct ({overall['combined']['accuracy']:.1f}%)")
    report.append("")
    
    report.append("BY MODEL:")
    report.append("-" * 40)
    for model in ["claude", "gpt-4o"]:
        model_stats = approach_accuracy["by_model"][model]
        report.append(f"\n{model.upper()}:")
        report.append(f"  Prompt Only    : {model_stats['prompt_only']['correct']:2d}/{model_stats['prompt_only']['total']:2d} circuits correct ({model_stats['prompt_only']['accuracy']:.1f}%)")
        report.append(f"  Mermaid Only   : {model_stats['mermaid_only']['correct']:2d}/{model_stats['mermaid_only']['total']:2d} circuits correct ({model_stats['mermaid_only']['accuracy']:.1f}%)")
        report.append(f"  Combined       : {model_stats['combined']['correct']:2d}/{model_stats['combined']['total']:2d} circuits correct ({model_stats['combined']['accuracy']:.1f}%)")
    
    report.append("")
    report.append("=" * 80)
    report.append("")
    
    return "\n".join(report)

def save_detailed_json(results: List[Dict], filename: str = "detailed_analysis.json"):
    """Save detailed analysis results to JSON"""
    circuit_stats = count_circuits_tested(results)
    correctness = calculate_functional_correctness(results)
    approach_accuracy = compare_approaches_accuracy(results)
    
    analysis = {
        "insight_1_circuit_count": circuit_stats,
        "insight_2_functional_correctness": correctness["statistics"],
        "insight_3_approach_accuracy": approach_accuracy
    }
    
    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✓ Detailed analysis saved to: {filename}")

def main():
    print("\n" + "=" * 80)
    print("MULTIMODAL MERMAID TEST RESULTS ANALYZER WITH VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    # Load JSON files
    print("Loading JSON files...")
    json_files = load_json_files()
    
    if not json_files:
        print("No JSON files found in the current directory!")
        return
    
    print(f"\nTotal JSON files loaded: {len(json_files)}\n")
    
    # Extract all results
    all_results = extract_all_results(json_files)
    print(f"Total circuit results extracted: {len(all_results)}\n")
    
    # Calculate statistics
    print("Calculating statistics...")
    correctness = calculate_functional_correctness(all_results)
    approach_accuracy = compare_approaches_accuracy(all_results)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    plot_rq1a_multimodal_strategy(approach_accuracy)
    plot_rq1b_complexity_scaling(correctness["statistics"])
    plot_rq2_model_comparison(correctness["statistics"], approach_accuracy)
    plot_comprehensive_heatmap(correctness["statistics"])
    plot_complexity_by_model(correctness["statistics"])
    plot_overall_success_rates(all_results)
    
    print("-" * 80)
    
    # Generate and display report
    report = generate_summary_report(all_results)
    print("\n" + report)
    
    # Save to file
    report_filename = "analysis_report.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to: {report_filename}")
    
    # Save detailed JSON
    save_detailed_json(all_results)
    
    print("\n" + "=" * 80)
    print("Analysis complete! Generated 6 high-quality PNG visualizations:")
    print("  1. RQ1a_multimodal_strategy_comparison.png")
    print("  2. RQ1b_complexity_scaling.png")
    print("  3. RQ2_model_comparison.png")
    print("  4. comprehensive_heatmap.png")
    print("  5. complexity_by_model_detailed.png")
    print("  6. success_rate_breakdown.png")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()