# tests/analysis/run_multimodal_experiments.py
"""
Run multimodal testing experiments multiple times for statistical significance
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import os


def run_experiment(run_number, total_runs, backend_url, models):
    """Run a single experiment"""
    print(f"\n{'='*70}")
    print(f"🔬 EXPERIMENT RUN {run_number}/{total_runs}")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {models}")
    print(f"Backend: {backend_url}")
    
    # Set environment variables
    env = os.environ.copy()
    env['BACKEND_URL'] = backend_url
    env['TEST_MODELS'] = models
    env['RUN_NUMBER'] = str(run_number)
    
    # Run the test script
    script_path = Path(__file__).parent / "test_multimodal_mermaid.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ Run {run_number} completed successfully!")
            return True
        else:
            print(f"\n❌ Run {run_number} failed with code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Run {run_number} crashed: {str(e)}")
        return False


def aggregate_results(results_dir, models, num_runs):
    """Aggregate results from multiple runs"""
    print(f"\n{'='*70}")
    print(f"📊 AGGREGATING RESULTS FROM {num_runs} RUNS")
    print(f"{'='*70}\n")
    
    model_list = [m.strip() for m in models.split(',')]
    
    for model in model_list:
        print(f"\n🔬 Model: {model.upper()}")
        print(f"{'-'*70}")
        
        # Find all result files for this model
        pattern = f"multimodal_{model}_results_*.json"
        result_files = sorted(results_dir.glob(pattern))
        
        if not result_files:
            print(f"⚠️  No result files found for {model}")
            continue
        
        print(f"Found {len(result_files)} result files")
        
        # Aggregate statistics
        aggregated = {
            'prompt_only': {'correct': [], 'total': [], 'percentage': []},
            'mermaid_only': {'correct': [], 'total': [], 'percentage': []},
            'combined': {'correct': [], 'total': [], 'percentage': []}
        }
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                stats = data.get('statistics', {})
                
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    if approach in stats:
                        aggregated[approach]['correct'].append(stats[approach]['correct'])
                        aggregated[approach]['total'].append(stats[approach]['total'])
                        aggregated[approach]['percentage'].append(stats[approach]['percentage'])
        
        # Calculate averages and std dev
        print(f"\n{'Approach':<20} {'Avg Success':<12} {'Std Dev':<10} {'Range'}")
        print(f"{'-'*70}")
        
        for approach in ['prompt_only', 'mermaid_only', 'combined']:
            name = approach.replace('_', ' ').title()
            percentages = aggregated[approach]['percentage']
            
            if percentages:
                avg = sum(percentages) / len(percentages)
                
                # Calculate std dev
                if len(percentages) > 1:
                    variance = sum((x - avg) ** 2 for x in percentages) / (len(percentages) - 1)
                    std_dev = variance ** 0.5
                else:
                    std_dev = 0.0
                
                min_pct = min(percentages)
                max_pct = max(percentages)
                
                print(f"{name:<20} {avg:>6.1f}%      {std_dev:>6.2f}%    {min_pct:.1f}%-{max_pct:.1f}%")
        
        # Print individual run results
        print(f"\n📋 Individual Runs:")
        for i, result_file in enumerate(result_files, 1):
            with open(result_file, 'r') as f:
                data = json.load(f)
                stats = data.get('statistics', {})
                
                prompt_pct = stats.get('prompt_only', {}).get('percentage', 0)
                mermaid_pct = stats.get('mermaid_only', {}).get('percentage', 0)
                combined_pct = stats.get('combined', {}).get('percentage', 0)
                
                print(f"  Run {i}: Prompt={prompt_pct:>5.1f}%  Mermaid={mermaid_pct:>5.1f}%  Combined={combined_pct:>5.1f}%")


def compare_models(results_dir, models, num_runs):
    """Compare models across all runs"""
    print(f"\n{'='*70}")
    print(f"🆚 MODEL COMPARISON (Averaged over {num_runs} runs)")
    print(f"{'='*70}\n")
    
    model_list = [m.strip() for m in models.split(',')]
    
    if len(model_list) < 2:
        print("⚠️  Only one model tested - no comparison needed")
        return
    
    model_stats = {}
    
    for model in model_list:
        pattern = f"multimodal_{model}_results_*.json"
        result_files = sorted(results_dir.glob(pattern))
        
        if not result_files:
            continue
        
        # Aggregate
        aggregated = {
            'prompt_only': [],
            'mermaid_only': [],
            'combined': []
        }
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                stats = data.get('statistics', {})
                
                for approach in ['prompt_only', 'mermaid_only', 'combined']:
                    if approach in stats:
                        aggregated[approach].append(stats[approach]['percentage'])
        
        # Calculate averages
        model_stats[model] = {
            'prompt_avg': sum(aggregated['prompt_only']) / len(aggregated['prompt_only']) if aggregated['prompt_only'] else 0,
            'mermaid_avg': sum(aggregated['mermaid_only']) / len(aggregated['mermaid_only']) if aggregated['mermaid_only'] else 0,
            'combined_avg': sum(aggregated['combined']) / len(aggregated['combined']) if aggregated['combined'] else 0
        }
        
        model_stats[model]['overall_avg'] = (
            model_stats[model]['prompt_avg'] + 
            model_stats[model]['mermaid_avg'] + 
            model_stats[model]['combined_avg']
        ) / 3
    
    # Print comparison table
    print(f"{'Model':<15} {'Prompt':<12} {'Mermaid':<12} {'Combined':<12} {'Overall'}")
    print(f"{'-'*70}")
    
    for model in model_list:
        if model in model_stats:
            stats = model_stats[model]
            print(f"{model:<15} {stats['prompt_avg']:>7.1f}%     {stats['mermaid_avg']:>7.1f}%     {stats['combined_avg']:>7.1f}%     {stats['overall_avg']:>7.1f}%")
    
    # Determine winner
    best_model = max(model_stats.items(), key=lambda x: x[1]['overall_avg'])
    print(f"\n🏆 Best Overall Model: {best_model[0].upper()} ({best_model[1]['overall_avg']:.1f}%)")
    
    # Best per approach
    print(f"\n🥇 Best by Approach:")
    best_prompt = max(model_stats.items(), key=lambda x: x[1]['prompt_avg'])
    best_mermaid = max(model_stats.items(), key=lambda x: x[1]['mermaid_avg'])
    best_combined = max(model_stats.items(), key=lambda x: x[1]['combined_avg'])
    
    print(f"   Prompt Only:  {best_prompt[0].upper()} ({best_prompt[1]['prompt_avg']:.1f}%)")
    print(f"   Mermaid Only: {best_mermaid[0].upper()} ({best_mermaid[1]['mermaid_avg']:.1f}%)")
    print(f"   Combined:     {best_combined[0].upper()} ({best_combined[1]['combined_avg']:.1f}%)")


def main():
    """Main experiment runner"""
    print("="*70)
    print("🧪 MULTIMODAL EXPERIMENT RUNNER")
    print("="*70)
    
    # Configuration
    NUM_RUNS = int(os.getenv("NUM_RUNS", "1"))
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
    MODELS = os.getenv("TEST_MODELS", "claude,gpt-4o")
    
    print(f"\nConfiguration:")
    print(f"  Number of runs: {NUM_RUNS}")
    print(f"  Backend URL: {BACKEND_URL}")
    print(f"  Models: {MODELS}")
    
    results_dir = Path(__file__).parent
    
    # Confirm
    response = input(f"\n⏸️  Run {NUM_RUNS} experiments? This will take a while. (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    # Run experiments
    start_time = datetime.now()
    successful_runs = 0
    
    for i in range(1, NUM_RUNS + 1):
        success = run_experiment(i, NUM_RUNS, BACKEND_URL, MODELS)
        if success:
            successful_runs += 1
        
        # Pause between runs (except after last run)
        if i < NUM_RUNS:
            print(f"\n⏸️  Pausing 5 seconds before next run...")
            import time
            time.sleep(5)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ EXPERIMENT SUITE COMPLETE")
    print(f"{'='*70}")
    print(f"Total runs: {NUM_RUNS}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {NUM_RUNS - successful_runs}")
    print(f"Duration: {duration.total_seconds():.1f}s ({duration.total_seconds()/60:.1f}m)")
    
    # Aggregate results
    if successful_runs > 0:
        aggregate_results(results_dir, MODELS, successful_runs)
        compare_models(results_dir, MODELS, successful_runs)
    else:
        print("\n❌ No successful runs to aggregate")


if __name__ == "__main__":
    main()