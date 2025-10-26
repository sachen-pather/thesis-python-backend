#!/usr/bin/env python3
"""
Aggregate Multimodal Test Results with Plotting
Parses multiple JSON result files, aggregates statistics, and generates publication-ready plots.
"""

import json
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# Plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Warning: matplotlib not available. Install with: pip install matplotlib")
    print("   Continuing without plotting functionality...\n")


class MultimodalResultsAggregator:
    """Aggregate results from multiple multimodal test runs"""
    
    def __init__(self, results_dir="."):
        self.results_dir = Path(results_dir)
        self.json_files = []
        self.all_results = []
        
    def find_result_files(self, pattern="multimodal_*_results_*.json"):
        """Find all matching result files"""
        search_pattern = str(self.results_dir / pattern)
        self.json_files = sorted(glob.glob(search_pattern))
        
        print(f"🔍 Searching: {search_pattern}")
        print(f"📁 Found {len(self.json_files)} result files")
        
        if self.json_files:
            print("\nFiles found:")
            for f in self.json_files:
                print(f"  • {Path(f).name}")
        
        return self.json_files
    
    def load_results(self):
        """Load all result files"""
        self.all_results = []
        
        for json_file in self.json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.all_results.append({
                        'filename': Path(json_file).name,
                        'data': data
                    })
                    print(f"✓ Loaded: {Path(json_file).name}")
            except Exception as e:
                print(f"✗ Error loading {json_file}: {e}")
        
        print(f"\n✅ Loaded {len(self.all_results)} result files successfully")
        return self.all_results
    
    def aggregate_statistics(self):
        """Aggregate statistics across all result files"""
        
        if not self.all_results:
            print("❌ No results to aggregate")
            return None
        
        # Initialize aggregated stats
        aggregated = {
            'total_files': len(self.all_results),
            'total_circuits_tested': 0,
            'models_tested': set(),
            'test_dates': [],
            'by_model': defaultdict(lambda: {
                'total_circuits': 0,
                'prompt_only': {'correct': 0, 'total': 0, 'success_rate': 0.0},
                'mermaid_only': {'correct': 0, 'total': 0, 'success_rate': 0.0},
                'combined': {'correct': 0, 'total': 0, 'success_rate': 0.0},
                'test_count': 0
            }),
            'overall': {
                'prompt_only': {'correct': 0, 'total': 0, 'success_rate': 0.0},
                'mermaid_only': {'correct': 0, 'total': 0, 'success_rate': 0.0},
                'combined': {'correct': 0, 'total': 0, 'success_rate': 0.0}
            },
            'by_category': defaultdict(lambda: {
                'prompt_only': {'correct': 0, 'total': 0},
                'mermaid_only': {'correct': 0, 'total': 0},
                'combined': {'correct': 0, 'total': 0}
            }),
            'by_complexity': defaultdict(lambda: {
                'prompt_only': {'correct': 0, 'total': 0},
                'mermaid_only': {'correct': 0, 'total': 0},
                'combined': {'correct': 0, 'total': 0}
            })
        }
        
        # Process each result file
        for result_file in self.all_results:
            data = result_file['data']
            metadata = data.get('metadata', {})
            stats = data.get('statistics', {})
            results = data.get('results', [])
            
            # Extract metadata
            model = metadata.get('model', 'unknown')
            test_date = metadata.get('test_date', 'unknown')
            total_circuits = metadata.get('total_circuits', len(results))
            
            aggregated['models_tested'].add(model)
            aggregated['test_dates'].append(test_date)
            aggregated['total_circuits_tested'] += total_circuits
            
            # Aggregate by model
            model_stats = aggregated['by_model'][model]
            model_stats['total_circuits'] += total_circuits
            model_stats['test_count'] += 1
            
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                if approach in stats:
                    model_stats[approach]['correct'] += stats[approach].get('correct', 0)
                    model_stats[approach]['total'] += stats[approach].get('total', 0)
                    
                    # Also aggregate to overall
                    aggregated['overall'][approach]['correct'] += stats[approach].get('correct', 0)
                    aggregated['overall'][approach]['total'] += stats[approach].get('total', 0)
            
            # Aggregate by category and complexity
            for result in results:
                category = result.get('category', 'unknown')
                complexity = result.get('complexity', 'unknown')
                
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    if approach in result:
                        approach_data = result[approach]
                        
                        # Check if correct
                        is_correct = (
                            approach_data.get('compiled', False) and 
                            approach_data.get('simulated', False) and
                            approach_data.get('has_waveform', False) and
                            not approach_data.get('anomalous', False)
                        )
                        
                        # By category
                        aggregated['by_category'][category][approach]['total'] += 1
                        if is_correct:
                            aggregated['by_category'][category][approach]['correct'] += 1
                        
                        # By complexity
                        aggregated['by_complexity'][complexity][approach]['total'] += 1
                        if is_correct:
                            aggregated['by_complexity'][complexity][approach]['correct'] += 1
        
        # Calculate success rates
        for approach in ['prompt_only', 'mermaid_only', 'combined']:
            overall = aggregated['overall'][approach]
            if overall['total'] > 0:
                overall['success_rate'] = (overall['correct'] / overall['total']) * 100
        
        # Calculate success rates by model
        for model, model_stats in aggregated['by_model'].items():
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                if model_stats[approach]['total'] > 0:
                    model_stats[approach]['success_rate'] = (
                        model_stats[approach]['correct'] / model_stats[approach]['total']
                    ) * 100
        
        # Calculate success rates by category
        for category, cat_stats in aggregated['by_category'].items():
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                if cat_stats[approach]['total'] > 0:
                    cat_stats[approach]['success_rate'] = (
                        cat_stats[approach]['correct'] / cat_stats[approach]['total']
                    ) * 100
                else:
                    cat_stats[approach]['success_rate'] = 0.0
        
        # Calculate success rates by complexity
        for complexity, comp_stats in aggregated['by_complexity'].items():
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                if comp_stats[approach]['total'] > 0:
                    comp_stats[approach]['success_rate'] = (
                        comp_stats[approach]['correct'] / comp_stats[approach]['total']
                    ) * 100
                else:
                    comp_stats[approach]['success_rate'] = 0.0
        
        # Convert sets to lists for JSON serialization
        aggregated['models_tested'] = sorted(list(aggregated['models_tested']))
        aggregated['by_model'] = dict(aggregated['by_model'])
        aggregated['by_category'] = dict(aggregated['by_category'])
        aggregated['by_complexity'] = dict(aggregated['by_complexity'])
        
        return aggregated
    
    def print_summary(self, aggregated):
        """Print aggregated summary"""
        
        print("\n" + "="*80)
        print("📊 AGGREGATED MULTIMODAL TEST RESULTS")
        print("="*80)
        
        # Overall summary
        print(f"\n📁 Files Analyzed: {aggregated['total_files']}")
        print(f"🔬 Models Tested: {', '.join(aggregated['models_tested'])}")
        print(f"🧪 Total Circuits Tested: {aggregated['total_circuits_tested']}")
        
        # Overall accuracy
        print(f"\n{'='*80}")
        print("🎯 OVERALL ACCURACY (RQ1a: Multimodal Input Strategy)")
        print("="*80)
        print(f"\n{'Approach':<20} {'Correct':<12} {'Total':<12} {'Success Rate'}")
        print("-"*60)
        
        for approach in ['prompt_only', 'mermaid_only', 'combined']:
            stats = aggregated['overall'][approach]
            name = approach.replace('_', ' ').title()
            print(f"{name:<20} {stats['correct']:<12} {stats['total']:<12} {stats['success_rate']:>6.2f}%")
        
        # Find best approach
        best_approach = max(
            ['prompt_only', 'mermaid_only', 'combined'],
            key=lambda x: aggregated['overall'][x]['success_rate']
        )
        best_rate = aggregated['overall'][best_approach]['success_rate']
        print(f"\n🏆 Best Approach: {best_approach.replace('_', ' ').title()} ({best_rate:.2f}%)")
        
        # By model
        if len(aggregated['models_tested']) > 1:
            print(f"\n{'='*80}")
            print("🤖 RESULTS BY MODEL")
            print("="*80)
            
            for model in sorted(aggregated['by_model'].keys()):
                model_stats = aggregated['by_model'][model]
                print(f"\n{model.upper()} ({model_stats['test_count']} test runs, {model_stats['total_circuits']} circuits):")
                print(f"{'Approach':<20} {'Success Rate'}")
                print("-"*40)
                
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    name = approach.replace('_', ' ').title()
                    rate = model_stats[approach]['success_rate']
                    correct = model_stats[approach]['correct']
                    total = model_stats[approach]['total']
                    print(f"{name:<20} {rate:>6.2f}% ({correct}/{total})")
        
        # By category
        print(f"\n{'='*80}")
        print("📦 RESULTS BY CATEGORY")
        print("="*80)
        
        for category in sorted(aggregated['by_category'].keys()):
            cat_stats = aggregated['by_category'][category]
            print(f"\n{category.upper()}:")
            print(f"{'Approach':<20} {'Success Rate'}")
            print("-"*40)
            
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                name = approach.replace('_', ' ').title()
                rate = cat_stats[approach]['success_rate']
                correct = cat_stats[approach]['correct']
                total = cat_stats[approach]['total']
                print(f"{name:<20} {rate:>6.2f}% ({correct}/{total})")
        
        # By complexity (RQ1b)
        print(f"\n{'='*80}")
        print("🎚️  RESULTS BY COMPLEXITY (RQ1b: Complexity Scaling)")
        print("="*80)
        
        for complexity in sorted(aggregated['by_complexity'].keys()):
            comp_stats = aggregated['by_complexity'][complexity]
            print(f"\n{complexity.upper()}:")
            print(f"{'Approach':<20} {'Success Rate'}")
            print("-"*40)
            
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                name = approach.replace('_', ' ').title()
                rate = comp_stats[approach]['success_rate']
                correct = comp_stats[approach]['correct']
                total = comp_stats[approach]['total']
                print(f"{name:<20} {rate:>6.2f}% ({correct}/{total})")
        
        print("\n" + "="*80)
    
    def generate_plots(self, aggregated, output_dir=None):
        """Generate publication-ready plots"""
        
        if not PLOTTING_AVAILABLE:
            print("\n⚠️  Skipping plot generation (matplotlib not installed)")
            return []
        
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("📈 GENERATING PLOTS")
        print("="*80)
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
        generated_plots = []
        
        # Plot 1: Overall Accuracy Comparison (RQ1a)
        try:
            fig1 = self._plot_overall_accuracy(aggregated)
            plot1_path = output_dir / "plot_overall_accuracy_rq1a.png"
            fig1.savefig(plot1_path, bbox_inches='tight', dpi=300)
            plt.close(fig1)
            generated_plots.append(plot1_path)
            print(f"✓ Generated: {plot1_path.name}")
        except Exception as e:
            print(f"✗ Failed to generate overall accuracy plot: {e}")
        
        # Plot 2: Complexity Scaling (RQ1b)
        try:
            fig2 = self._plot_complexity_scaling(aggregated)
            plot2_path = output_dir / "plot_complexity_scaling_rq1b.png"
            fig2.savefig(plot2_path, bbox_inches='tight', dpi=300)
            plt.close(fig2)
            generated_plots.append(plot2_path)
            print(f"✓ Generated: {plot2_path.name}")
        except Exception as e:
            print(f"✗ Failed to generate complexity scaling plot: {e}")
        
        # Plot 3: By Category
        try:
            fig3 = self._plot_by_category(aggregated)
            plot3_path = output_dir / "plot_by_category.png"
            fig3.savefig(plot3_path, bbox_inches='tight', dpi=300)
            plt.close(fig3)
            generated_plots.append(plot3_path)
            print(f"✓ Generated: {plot3_path.name}")
        except Exception as e:
            print(f"✗ Failed to generate category plot: {e}")
        
        # Plot 4: Model Comparison (if multiple models)
        if len(aggregated['models_tested']) > 1:
            try:
                fig4 = self._plot_model_comparison(aggregated)
                plot4_path = output_dir / "plot_model_comparison.png"
                fig4.savefig(plot4_path, bbox_inches='tight', dpi=300)
                plt.close(fig4)
                generated_plots.append(plot4_path)
                print(f"✓ Generated: {plot4_path.name}")
            except Exception as e:
                print(f"✗ Failed to generate model comparison plot: {e}")
        
        # Plot 5: Heatmap of success rates
        try:
            fig5 = self._plot_heatmap(aggregated)
            plot5_path = output_dir / "plot_heatmap.png"
            fig5.savefig(plot5_path, bbox_inches='tight', dpi=300)
            plt.close(fig5)
            generated_plots.append(plot5_path)
            print(f"✓ Generated: {plot5_path.name}")
        except Exception as e:
            print(f"✗ Failed to generate heatmap: {e}")
        
        # Plot 6: Combined visualization
        try:
            fig6 = self._plot_combined_summary(aggregated)
            plot6_path = output_dir / "plot_combined_summary.png"
            fig6.savefig(plot6_path, bbox_inches='tight', dpi=300)
            plt.close(fig6)
            generated_plots.append(plot6_path)
            print(f"✓ Generated: {plot6_path.name}")
        except Exception as e:
            print(f"✗ Failed to generate combined summary: {e}")
        
        print(f"\n✅ Generated {len(generated_plots)} plots")
        return generated_plots
    
    def _plot_overall_accuracy(self, aggregated):
        """Plot overall accuracy comparison (RQ1a)"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
        approach_keys = ['prompt_only', 'mermaid_only', 'combined']
        
        success_rates = [aggregated['overall'][key]['success_rate'] for key in approach_keys]
        correct = [aggregated['overall'][key]['correct'] for key in approach_keys]
        total = [aggregated['overall'][key]['total'] for key in approach_keys]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax.bar(approaches, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for i, (bar, rate, c, t) in enumerate(zip(bars, success_rates, correct, total)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%\n({c}/{t})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_xlabel('Input Strategy', fontweight='bold')
        ax.set_title('RQ1a: Multimodal Input Strategy Comparison\nOverall Accuracy Across All Circuits',
                    fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50% Baseline')
        
        # Highlight best approach
        best_idx = success_rates.index(max(success_rates))
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
        
        ax.legend()
        plt.tight_layout()
        return fig
    
    def _plot_complexity_scaling(self, aggregated):
        """Plot complexity scaling analysis (RQ1b)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        complexities = sorted(aggregated['by_complexity'].keys())
        approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
        approach_keys = ['prompt_only', 'mermaid_only', 'combined']
        
        x = np.arange(len(complexities))
        width = 0.25
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, (approach, key, color) in enumerate(zip(approaches, approach_keys, colors)):
            rates = [aggregated['by_complexity'][comp][key]['success_rate'] 
                    for comp in complexities]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, rates, width, label=approach, color=color, 
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Circuit Complexity', fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('RQ1b: Complexity Scaling Analysis\nPerformance Across Complexity Levels',
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in complexities])
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def _plot_by_category(self, aggregated):
        """Plot results by circuit category"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = sorted(aggregated['by_category'].keys())
        approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
        approach_keys = ['prompt_only', 'mermaid_only', 'combined']
        
        x = np.arange(len(categories))
        width = 0.25
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, (approach, key, color) in enumerate(zip(approaches, approach_keys, colors)):
            rates = [aggregated['by_category'][cat][key]['success_rate'] 
                    for cat in categories]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, rates, width, label=approach, color=color,
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Circuit Category', fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('Success Rate by Circuit Category', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in categories])
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def _plot_model_comparison(self, aggregated):
        """Plot comparison between different models"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = sorted(aggregated['by_model'].keys())
        approaches = ['Prompt Only', 'Mermaid Only', 'Combined']
        approach_keys = ['prompt_only', 'mermaid_only', 'combined']
        
        x = np.arange(len(models))
        width = 0.25
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, (approach, key, color) in enumerate(zip(approaches, approach_keys, colors)):
            rates = [aggregated['by_model'][model][key]['success_rate'] 
                    for model in models]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, rates, width, label=approach, color=color,
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def _plot_heatmap(self, aggregated):
        """Plot heatmap of success rates"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap 1: By Complexity
        complexities = sorted(aggregated['by_complexity'].keys())
        approaches = ['Prompt\nOnly', 'Mermaid\nOnly', 'Combined']
        approach_keys = ['prompt_only', 'mermaid_only', 'combined']
        
        data1 = np.array([[aggregated['by_complexity'][comp][key]['success_rate'] 
                          for key in approach_keys] for comp in complexities])
        
        im1 = ax1.imshow(data1.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax1.set_xticks(np.arange(len(complexities)))
        ax1.set_yticks(np.arange(len(approaches)))
        ax1.set_xticklabels([c.title() for c in complexities])
        ax1.set_yticklabels(approaches)
        ax1.set_xlabel('Complexity', fontweight='bold')
        ax1.set_title('Success Rate by Complexity', fontweight='bold')
        
        # Add text annotations
        for i in range(len(complexities)):
            for j in range(len(approaches)):
                text = ax1.text(i, j, f'{data1[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im1, ax=ax1, label='Success Rate (%)')
        
        # Heatmap 2: By Category
        categories = sorted(aggregated['by_category'].keys())
        
        data2 = np.array([[aggregated['by_category'][cat][key]['success_rate'] 
                          for key in approach_keys] for cat in categories])
        
        im2 = ax2.imshow(data2.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax2.set_xticks(np.arange(len(categories)))
        ax2.set_yticks(np.arange(len(approaches)))
        ax2.set_xticklabels([c.title() for c in categories])
        ax2.set_yticklabels(approaches)
        ax2.set_xlabel('Category', fontweight='bold')
        ax2.set_title('Success Rate by Category', fontweight='bold')
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(approaches)):
                text = ax2.text(i, j, f'{data2[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im2, ax=ax2, label='Success Rate (%)')
        
        fig.suptitle('Success Rate Heatmaps', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def _plot_combined_summary(self, aggregated):
        """Plot combined summary visualization"""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Overall Accuracy (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        approaches = ['Prompt\nOnly', 'Mermaid\nOnly', 'Combined']
        approach_keys = ['prompt_only', 'mermaid_only', 'combined']
        rates = [aggregated['overall'][key]['success_rate'] for key in approach_keys]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax1.bar(approaches, rates, color=colors, alpha=0.8, edgecolor='black')
        for bar, rate in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Success Rate (%)', fontweight='bold')
        ax1.set_title('Overall Accuracy', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Complexity Scaling (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        complexities = sorted(aggregated['by_complexity'].keys())
        for i, (key, color, label) in enumerate(zip(approach_keys, colors, 
                                                     ['Prompt Only', 'Mermaid Only', 'Combined'])):
            rates = [aggregated['by_complexity'][comp][key]['success_rate'] 
                    for comp in complexities]
            ax2.plot([c.title() for c in complexities], rates, marker='o', 
                    linewidth=2, color=color, label=label, markersize=8)
        ax2.set_ylabel('Success Rate (%)', fontweight='bold')
        ax2.set_xlabel('Complexity', fontweight='bold')
        ax2.set_title('Complexity Scaling', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(alpha=0.3)
        
        # 3. Category Breakdown (middle row, span both columns)
        ax3 = fig.add_subplot(gs[1, :])
        categories = sorted(aggregated['by_category'].keys())
        x = np.arange(len(categories))
        width = 0.25
        for i, (key, color, label) in enumerate(zip(approach_keys, colors,
                                                     ['Prompt Only', 'Mermaid Only', 'Combined'])):
            rates = [aggregated['by_category'][cat][key]['success_rate'] 
                    for cat in categories]
            offset = (i - 1) * width
            ax3.bar(x + offset, rates, width, label=label, color=color, 
                   alpha=0.8, edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels([c.title() for c in categories])
        ax3.set_ylabel('Success Rate (%)', fontweight='bold')
        ax3.set_xlabel('Category', fontweight='bold')
        ax3.set_title('Performance by Circuit Category', fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Summary Statistics (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')
        summary_text = f"""
SUMMARY STATISTICS

Total Files Analyzed: {aggregated['total_files']}
Total Circuits Tested: {aggregated['total_circuits_tested']}
Models: {', '.join(aggregated['models_tested'])}

Best Overall Approach:
  {max(['Prompt Only', 'Mermaid Only', 'Combined'], 
       key=lambda x: aggregated['overall'][x.lower().replace(' ', '_')]['success_rate'])}
  ({max(rates):.2f}%)

Average Success Rate:
  {np.mean(rates):.2f}%
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 5. Key Findings (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Calculate key findings
        best_approach_key = max(approach_keys, key=lambda x: aggregated['overall'][x]['success_rate'])
        best_approach_name = best_approach_key.replace('_', ' ').title()
        best_rate = aggregated['overall'][best_approach_key]['success_rate']
        
        worst_approach_key = min(approach_keys, key=lambda x: aggregated['overall'][x]['success_rate'])
        worst_rate = aggregated['overall'][worst_approach_key]['success_rate']
        
        improvement = best_rate - worst_rate
        
        findings_text = f"""
KEY FINDINGS

RQ1a: Input Strategy
  ✓ {best_approach_name} performed best
  ✓ {improvement:.1f}% improvement over worst
  
RQ1b: Complexity Impact
  ✓ Performance scales with complexity
  ✓ See complexity scaling plot
  
Recommendation:
  Use {best_approach_name} for
  optimal HDL generation results
        """
        ax5.text(0.1, 0.5, findings_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        fig.suptitle('Comprehensive Results Summary', fontweight='bold', fontsize=16)
        return fig
    
    def save_aggregated_results(self, aggregated, output_file=None):
        """Save aggregated results to JSON"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aggregated_multimodal_results_{timestamp}.json"
        
        output_data = {
            'aggregation_date': datetime.now().isoformat(),
            'aggregated_statistics': aggregated,
            'source_files': [Path(f).name for f in self.json_files]
        }
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved aggregated results: {output_path.name}")
        return output_path
    
    def generate_text_summary(self, aggregated, output_file=None):
        """Generate detailed text summary with lines of code analysis"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"summary_report_{timestamp}.txt"
        
        output_path = self.results_dir / output_file
        
        # Calculate lines of code statistics
        loc_stats = self._calculate_loc_statistics()
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTIMODAL HDL VERIFICATION - DETAILED SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Files Analyzed: {aggregated['total_files']}\n")
            f.write(f"Total Circuits Tested: {aggregated['total_circuits_tested']}\n")
            f.write(f"Models: {', '.join(aggregated['models_tested'])}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # OVERALL CORRECTNESS
            f.write("OVERALL CORRECTNESS\n")
            f.write("-"*80 + "\n\n")
            
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                stats = aggregated['overall'][approach]
                name = approach.replace('_', ' ').title()
                f.write(f"{name}:\n")
                f.write(f"  Correct: {stats['correct']}/{stats['total']} circuits\n")
                f.write(f"  Success Rate: {stats['success_rate']:.2f}%\n")
                f.write(f"  Failed: {stats['total'] - stats['correct']} circuits\n\n")
            
            best_approach = max(['prompt_only', 'mermaid_only', 'combined'],
                              key=lambda x: aggregated['overall'][x]['success_rate'])
            f.write(f"Best Performing Approach: {best_approach.replace('_', ' ').title()}\n")
            f.write(f"  ({aggregated['overall'][best_approach]['success_rate']:.2f}%)\n\n")
            
            f.write("="*80 + "\n\n")
            
            # CORRECTNESS BY CATEGORY
            f.write("CORRECTNESS BY CATEGORY\n")
            f.write("-"*80 + "\n\n")
            
            for category in sorted(aggregated['by_category'].keys()):
                cat_stats = aggregated['by_category'][category]
                f.write(f"{category.upper()}:\n\n")
                
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    name = approach.replace('_', ' ').title()
                    correct = cat_stats[approach]['correct']
                    total = cat_stats[approach]['total']
                    rate = cat_stats[approach]['success_rate']
                    f.write(f"  {name}:\n")
                    f.write(f"    Correct: {correct}/{total}\n")
                    f.write(f"    Success Rate: {rate:.2f}%\n")
                    f.write(f"    Failed: {total - correct}\n\n")
            
            f.write("="*80 + "\n\n")
            
            # CORRECTNESS BY COMPLEXITY
            f.write("CORRECTNESS BY COMPLEXITY\n")
            f.write("-"*80 + "\n\n")
            
            for complexity in sorted(aggregated['by_complexity'].keys()):
                comp_stats = aggregated['by_complexity'][complexity]
                f.write(f"{complexity.upper()}:\n\n")
                
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    name = approach.replace('_', ' ').title()
                    correct = comp_stats[approach]['correct']
                    total = comp_stats[approach]['total']
                    rate = comp_stats[approach]['success_rate']
                    f.write(f"  {name}:\n")
                    f.write(f"    Correct: {correct}/{total}\n")
                    f.write(f"    Success Rate: {rate:.2f}%\n")
                    f.write(f"    Failed: {total - correct}\n\n")
            
            f.write("="*80 + "\n\n")
            
            # LINES OF CODE ANALYSIS
            if loc_stats:
                f.write("LINES OF CODE ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                
                f.write("Average Lines of Code per Approach:\n\n")
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    if approach in loc_stats['by_approach']:
                        stats = loc_stats['by_approach'][approach]
                        name = approach.replace('_', ' ').title()
                        f.write(f"  {name}:\n")
                        f.write(f"    Average: {stats['avg']:.1f} lines\n")
                        f.write(f"    Min: {stats['min']} lines\n")
                        f.write(f"    Max: {stats['max']} lines\n")
                        f.write(f"    Total samples: {stats['count']}\n\n")
                
                if 'by_complexity' in loc_stats:
                    f.write("\nAverage Lines of Code by Complexity:\n\n")
                    for complexity in sorted(loc_stats['by_complexity'].keys()):
                        comp_stats = loc_stats['by_complexity'][complexity]
                        f.write(f"  {complexity.title()}:\n")
                        for approach in ['prompt_only', 'mermaid_only', 'combined']:
                            if approach in comp_stats:
                                avg = comp_stats[approach]['avg']
                                name = approach.replace('_', ' ').title()
                                f.write(f"    {name}: {avg:.1f} lines\n")
                        f.write("\n")
                
                f.write("="*80 + "\n\n")
            
            # MODEL COMPARISON (if multiple models)
            if len(aggregated['models_tested']) > 1:
                f.write("MODEL COMPARISON\n")
                f.write("-"*80 + "\n\n")
                
                for model in sorted(aggregated['by_model'].keys()):
                    model_stats = aggregated['by_model'][model]
                    f.write(f"{model.upper()}:\n")
                    f.write(f"  Test Runs: {model_stats['test_count']}\n")
                    f.write(f"  Total Circuits: {model_stats['total_circuits']}\n\n")
                    
                    for approach in ['prompt_only', 'mermaid_only', 'combined']:
                        name = approach.replace('_', ' ').title()
                        rate = model_stats[approach]['success_rate']
                        correct = model_stats[approach]['correct']
                        total = model_stats[approach]['total']
                        f.write(f"  {name}: {rate:.2f}% ({correct}/{total})\n")
                    f.write("\n")
                
                f.write("="*80 + "\n\n")
            
            # KEY FINDINGS
            f.write("KEY FINDINGS\n")
            f.write("-"*80 + "\n\n")
            
            # Best approach
            best_approach_name = best_approach.replace('_', ' ').title()
            best_rate = aggregated['overall'][best_approach]['success_rate']
            f.write(f"1. Best Overall Approach: {best_approach_name} ({best_rate:.2f}%)\n\n")
            
            # Complexity trend
            complexities = sorted(aggregated['by_complexity'].keys())
            if len(complexities) > 1:
                simple_rate = aggregated['by_complexity'][complexities[0]][best_approach]['success_rate']
                complex_rate = aggregated['by_complexity'][complexities[-1]][best_approach]['success_rate']
                degradation = simple_rate - complex_rate
                f.write(f"2. Complexity Impact: {degradation:.1f}% performance drop from simplest to most complex\n\n")
            
            # Category performance
            if aggregated['by_category']:
                best_category = max(aggregated['by_category'].keys(),
                                  key=lambda x: aggregated['by_category'][x][best_approach]['success_rate'])
                best_cat_rate = aggregated['by_category'][best_category][best_approach]['success_rate']
                f.write(f"3. Best Category: {best_category.title()} ({best_cat_rate:.2f}%)\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"📄 Saved text summary: {output_path.name}")
        return output_path
    
    def _calculate_loc_statistics(self):
        """Calculate lines of code statistics from loaded results"""
        
        loc_data = {
            'by_approach': {
                'prompt_only': {'values': [], 'avg': 0, 'min': 0, 'max': 0, 'count': 0},
                'mermaid_only': {'values': [], 'avg': 0, 'min': 0, 'max': 0, 'count': 0},
                'combined': {'values': [], 'avg': 0, 'min': 0, 'max': 0, 'count': 0}
            },
            'by_complexity': defaultdict(lambda: {
                'prompt_only': {'values': [], 'avg': 0},
                'mermaid_only': {'values': [], 'avg': 0},
                'combined': {'values': [], 'avg': 0}
            })
        }
        
        for result_file in self.all_results:
            results = result_file['data'].get('results', [])
            
            for result in results:
                complexity = result.get('complexity', 'unknown')
                
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    if approach in result:
                        loc = result[approach].get('lines_of_code', 0)
                        if loc > 0:
                            loc_data['by_approach'][approach]['values'].append(loc)
                            loc_data['by_complexity'][complexity][approach]['values'].append(loc)
        
        # Calculate statistics
        for approach in ['prompt_only', 'mermaid_only', 'combined']:
            values = loc_data['by_approach'][approach]['values']
            if values:
                loc_data['by_approach'][approach]['avg'] = sum(values) / len(values)
                loc_data['by_approach'][approach]['min'] = min(values)
                loc_data['by_approach'][approach]['max'] = max(values)
                loc_data['by_approach'][approach]['count'] = len(values)
        
        for complexity in loc_data['by_complexity']:
            for approach in ['prompt_only', 'mermaid_only', 'combined']:
                values = loc_data['by_complexity'][complexity][approach]['values']
                if values:
                    loc_data['by_complexity'][complexity][approach]['avg'] = sum(values) / len(values)
        
        return loc_data if any(loc_data['by_approach'][a]['values'] for a in ['prompt_only', 'mermaid_only', 'combined']) else None


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Aggregate multimodal test results and generate plots'
    )
    parser.add_argument(
        '--dir',
        default='.',
        help='Directory containing result files (default: current directory)'
    )
    parser.add_argument(
        '--pattern',
        default='multimodal_*_results_*.json',
        help='File pattern to match (default: multimodal_*_results_*.json)'
    )
    parser.add_argument(
        '--output',
        help='Output file name (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--plot-dir',
        help='Directory for plots (default: same as --dir)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 MULTIMODAL RESULTS AGGREGATOR WITH PLOTTING")
    print("="*80)
    
    # Create aggregator
    aggregator = MultimodalResultsAggregator(results_dir=args.dir)
    
    # Find and load result files
    files = aggregator.find_result_files(pattern=args.pattern)
    
    if not files:
        print("\n❌ No result files found matching pattern")
        print(f"   Pattern: {args.pattern}")
        print(f"   Directory: {Path(args.dir).absolute()}")
        sys.exit(1)
    
    print()
    aggregator.load_results()
    
    # Aggregate statistics
    print("\n⚙️  Aggregating statistics...")
    aggregated = aggregator.aggregate_statistics()
    
    if aggregated:
        # Print summary
        aggregator.print_summary(aggregated)
        
        # Generate plots
        if not args.no_plots:
            plot_dir = args.plot_dir if args.plot_dir else args.dir
            plots = aggregator.generate_plots(aggregated, output_dir=plot_dir)
            
            if plots:
                print("\n📊 Generated Plots:")
                for plot in plots:
                    print(f"   • {plot.name}")
        
        # Save results
        aggregator.save_aggregated_results(aggregated, output_file=args.output)
        
        # Generate text summary
        aggregator.generate_text_summary(aggregated)
        
        print("\n✅ Aggregation complete!")
    else:
        print("\n❌ Failed to aggregate results")
        sys.exit(1)


if __name__ == "__main__":
    main()