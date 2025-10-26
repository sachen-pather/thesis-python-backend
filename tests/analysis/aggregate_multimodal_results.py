#!/usr/bin/env python3
"""
Aggregate Multimodal Test Results
Parses multiple JSON result files and aggregates overall accuracy metrics.
"""

import json
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys


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
        print("🎯 OVERALL ACCURACY (RQ1: Multimodal Input Strategy)")
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


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Aggregate multimodal test results from multiple JSON files'
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
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 MULTIMODAL RESULTS AGGREGATOR")
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
        
        # Save results
        aggregator.save_aggregated_results(aggregated, output_file=args.output)
        
        print("\n✅ Aggregation complete!")
    else:
        print("\n❌ Failed to aggregate results")
        sys.exit(1)


if __name__ == "__main__":
    main()