"""
Analyze Simple vs Complex Circuit Performance
Breaks down results by circuit complexity

SAVE AS: tests/analysis/analyze_simple_vs_complex.py
RUN: python tests/analysis/analyze_simple_vs_complex.py gpt_claude_vae_comparison_20251016_002647.json
"""

import json
import sys
from collections import defaultdict

def calculate_metrics(results, correct_key, predicted_key):
    """Calculate accuracy metrics"""
    if not results:
        return {"count": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    
    correct = sum(1 for r in results if r[correct_key])
    accuracy = correct / len(results) * 100
    
    tp = sum(1 for r in results if not r['expected_normal'] and not r[predicted_key])
    tn = sum(1 for r in results if r['expected_normal'] and r[predicted_key])
    fp = sum(1 for r in results if r['expected_normal'] and not r[predicted_key])
    fn = sum(1 for r in results if not r['expected_normal'] and r[predicted_key])
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "count": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def analyze_by_complexity(json_file):
    """Analyze results split by simple vs complex circuits"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data['detailed_results']
    
    # Define simple circuit names (from comprehensive_vae_test_suite)
    simple_circuits = {
        "2-Input AND", "2-Input OR", "2-Input XOR", "NOT Gate", "2-Input NAND", "2-Input NOR",
        "3-Input AND", "2:1 Mux",
        "Stuck AND (always 0)", "Stuck AND (always 1)", "Inverted AND (NAND)", 
        "Wrong OR (acts like AND)", "Inverted XOR (XNOR)", "Partial Mux (ignores sel)",
        "4-bit Counter", "D Flip-Flop", "T Flip-Flop", "Shift Register",
        "Stuck Counter", "Counter (no reset)", "DFF (stuck output)", "Shift Register (no shift)",
        "Half Adder", "Full Adder",
        "Half Adder (wrong sum)", "Full Adder (no carry)"
    }
    
    # Split results
    simple_results = []
    complex_results = []
    
    for r in results:
        if r['circuit_name'] in simple_circuits:
            simple_results.append(r)
        else:
            complex_results.append(r)
    
    print("="*80)
    print("SIMPLE vs COMPLEX CIRCUIT ANALYSIS")
    print("="*80)
    print(f"\n📊 DATASET BREAKDOWN:")
    print(f"  Simple Circuits (Basic): {len(simple_results)} circuits")
    print(f"  Complex Circuits (Extended): {len(complex_results)} circuits")
    print(f"  Total: {len(results)} circuits")
    
    # Analyze each model on both sets
    models = {}
    
    # Get unique models from results
    for r in results:
        model = r['llm_model']
        if model not in models:
            models[model] = {'simple': [], 'complex': []}
    
    # Categorize results by model and complexity
    for r in results:
        model = r['llm_model']
        if r['circuit_name'] in simple_circuits:
            models[model]['simple'].append(r)
        else:
            models[model]['complex'].append(r)
    
    # Calculate VAE metrics (same for all models)
    vae_simple = [r for r in simple_results if r['vae_available'] and r['vae_correct'] is not None]
    vae_complex = [r for r in complex_results if r['vae_available'] and r['vae_correct'] is not None]
    
    vae_simple_metrics = calculate_metrics(vae_simple, 'vae_correct', 'vae_predicted_normal')
    vae_complex_metrics = calculate_metrics(vae_complex, 'vae_correct', 'vae_predicted_normal')
    
    print("\n" + "="*80)
    print("VAE PERFORMANCE")
    print("="*80)
    print(f"\n{'Complexity':<20} {'Count':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    print(f"{'Simple':<20} {vae_simple_metrics['count']:<10} {vae_simple_metrics['accuracy']:>6.1f}%     "
          f"{vae_simple_metrics['precision']:>6.1f}%     {vae_simple_metrics['recall']:>6.1f}%     "
          f"{vae_simple_metrics['f1_score']:>6.1f}%")
    print(f"{'Complex':<20} {vae_complex_metrics['count']:<10} {vae_complex_metrics['accuracy']:>6.1f}%     "
          f"{vae_complex_metrics['precision']:>6.1f}%     {vae_complex_metrics['recall']:>6.1f}%     "
          f"{vae_complex_metrics['f1_score']:>6.1f}%")
    
    # Analyze each LLM model
    for model_id in sorted(models.keys()):
        model_name = "GPT-4o" if model_id == "gpt-4o" else "Claude Sonnet 3.5"
        
        simple_llm = [r for r in models[model_id]['simple'] 
                      if r['llm_available'] and r['llm_correct'] is not None]
        complex_llm = [r for r in models[model_id]['complex'] 
                       if r['llm_available'] and r['llm_correct'] is not None]
        
        simple_metrics = calculate_metrics(simple_llm, 'llm_correct', 'llm_predicted_normal')
        complex_metrics = calculate_metrics(complex_llm, 'llm_correct', 'llm_predicted_normal')
        
        print(f"\n{'='*80}")
        print(f"{model_name} PERFORMANCE")
        print("="*80)
        print(f"\n{'Complexity':<20} {'Count':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        print(f"{'Simple':<20} {simple_metrics['count']:<10} {simple_metrics['accuracy']:>6.1f}%     "
              f"{simple_metrics['precision']:>6.1f}%     {simple_metrics['recall']:>6.1f}%     "
              f"{simple_metrics['f1_score']:>6.1f}%")
        print(f"{'Complex':<20} {complex_metrics['count']:<10} {complex_metrics['accuracy']:>6.1f}%     "
              f"{complex_metrics['precision']:>6.1f}%     {complex_metrics['recall']:>6.1f}%     "
              f"{complex_metrics['f1_score']:>6.1f}%")
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    print("\n🎯 Accuracy Comparison (Simple vs Complex):")
    print("-" * 80)
    print(f"{'Model':<25} {'Simple':<15} {'Complex':<15} {'Difference':<15}")
    print("-" * 80)
    print(f"{'VAE':<25} {vae_simple_metrics['accuracy']:>6.1f}%        "
          f"{vae_complex_metrics['accuracy']:>6.1f}%        "
          f"{vae_simple_metrics['accuracy'] - vae_complex_metrics['accuracy']:>+6.1f}%")
    
    for model_id in sorted(models.keys()):
        model_name = "GPT-4o" if model_id == "gpt-4o" else "Claude Sonnet 3.5"
        
        simple_llm = [r for r in models[model_id]['simple'] 
                      if r['llm_available'] and r['llm_correct'] is not None]
        complex_llm = [r for r in models[model_id]['complex'] 
                       if r['llm_available'] and r['llm_correct'] is not None]
        
        simple_metrics = calculate_metrics(simple_llm, 'llm_correct', 'llm_predicted_normal')
        complex_metrics = calculate_metrics(complex_llm, 'llm_correct', 'llm_predicted_normal')
        
        print(f"{model_name:<25} {simple_metrics['accuracy']:>6.1f}%        "
              f"{complex_metrics['accuracy']:>6.1f}%        "
              f"{simple_metrics['accuracy'] - complex_metrics['accuracy']:>+6.1f}%")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Determine which models handle complexity better
    vae_drop = vae_simple_metrics['accuracy'] - vae_complex_metrics['accuracy']
    
    insights = []
    for model_id in sorted(models.keys()):
        model_name = "GPT-4o" if model_id == "gpt-4o" else "Claude Sonnet 3.5"
        
        simple_llm = [r for r in models[model_id]['simple'] 
                      if r['llm_available'] and r['llm_correct'] is not None]
        complex_llm = [r for r in models[model_id]['complex'] 
                       if r['llm_available'] and r['llm_correct'] is not None]
        
        simple_metrics = calculate_metrics(simple_llm, 'llm_correct', 'llm_predicted_normal')
        complex_metrics = calculate_metrics(complex_llm, 'llm_correct', 'llm_predicted_normal')
        
        drop = simple_metrics['accuracy'] - complex_metrics['accuracy']
        insights.append((model_name, drop))
    
    print(f"\n1. VAE Performance:")
    print(f"   - Simple circuits: {vae_simple_metrics['accuracy']:.1f}%")
    print(f"   - Complex circuits: {vae_complex_metrics['accuracy']:.1f}%")
    print(f"   - Accuracy drop: {vae_drop:+.1f}%")
    
    for model_name, drop in insights:
        print(f"\n2. {model_name} Performance:")
        simple_llm = [r for r in models['gpt-4o' if 'GPT' in model_name else 'claude']['simple'] 
                      if r['llm_available'] and r['llm_correct'] is not None]
        complex_llm = [r for r in models['gpt-4o' if 'GPT' in model_name else 'claude']['complex'] 
                       if r['llm_available'] and r['llm_correct'] is not None]
        simple_metrics = calculate_metrics(simple_llm, 'llm_correct', 'llm_predicted_normal')
        complex_metrics = calculate_metrics(complex_llm, 'llm_correct', 'llm_predicted_normal')
        
        print(f"   - Simple circuits: {simple_metrics['accuracy']:.1f}%")
        print(f"   - Complex circuits: {complex_metrics['accuracy']:.1f}%")
        print(f"   - Accuracy drop: {drop:+.1f}%")
    
    # Best model for each complexity
    print(f"\n3. Best Model by Complexity:")
    
    # Simple circuits
    best_simple_model = "VAE"
    best_simple_acc = vae_simple_metrics['accuracy']
    
    for model_id in models.keys():
        model_name = "GPT-4o" if model_id == "gpt-4o" else "Claude Sonnet 3.5"
        simple_llm = [r for r in models[model_id]['simple'] 
                      if r['llm_available'] and r['llm_correct'] is not None]
        simple_metrics = calculate_metrics(simple_llm, 'llm_correct', 'llm_predicted_normal')
        
        if simple_metrics['accuracy'] > best_simple_acc:
            best_simple_acc = simple_metrics['accuracy']
            best_simple_model = model_name
    
    # Complex circuits
    best_complex_model = "VAE"
    best_complex_acc = vae_complex_metrics['accuracy']
    
    for model_id in models.keys():
        model_name = "GPT-4o" if model_id == "gpt-4o" else "Claude Sonnet 3.5"
        complex_llm = [r for r in models[model_id]['complex'] 
                       if r['llm_available'] and r['llm_correct'] is not None]
        complex_metrics = calculate_metrics(complex_llm, 'llm_correct', 'llm_predicted_normal')
        
        if complex_metrics['accuracy'] > best_complex_acc:
            best_complex_acc = complex_metrics['accuracy']
            best_complex_model = model_name
    
    print(f"   - Simple circuits: {best_simple_model} ({best_simple_acc:.1f}%)")
    print(f"   - Complex circuits: {best_complex_model} ({best_complex_acc:.1f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_simple_vs_complex.py <json_file>")
        print("\nExample:")
        print("  python tests/analysis/analyze_simple_vs_complex.py gpt_claude_vae_comparison_20251016_002647.json")
        sys.exit(1)
    
    analyze_by_complexity(sys.argv[1])