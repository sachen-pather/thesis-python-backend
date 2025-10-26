"""
Modular Complexity Comparison Framework
Test any combination of Simple, Medium, and Complex circuits

SAVE AS: tests/analysis/modular_complexity_comparison.py
RUN: python tests/analysis/modular_complexity_comparison.py --suites simple medium complex
     python tests/analysis/modular_complexity_comparison.py --suites complex
     python tests/analysis/modular_complexity_comparison.py --suites medium complex

Allows modular testing of different complexity tiers for thesis analysis
"""

import sys
import os
import argparse
import requests
import time
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Import test suites
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_URL = "http://localhost:8000"

# ONLY TEST THESE TWO MODELS
MODELS_TO_TEST = {
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "endpoint_name": "gpt-4o"
    },
    "claude": {
        "name": "Claude Sonnet 3.5",
        "provider": "Anthropic", 
        "endpoint_name": "claude"
    }
}

@dataclass
class VerificationResult:
    circuit_name: str
    category: str
    complexity: str  # NEW: simple, medium, or complex
    expected_normal: bool
    vae_predicted_normal: Optional[bool]
    vae_confidence: Optional[float]
    vae_correct: Optional[bool]
    vae_available: bool
    vae_message: str
    llm_model: str
    llm_predicted_normal: Optional[bool]
    llm_confidence: Optional[float]
    llm_correct: Optional[bool]
    llm_available: bool
    llm_analysis: str
    llm_raw_response: str
    vae_time: float
    llm_time: float
    total_time: float

def log(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "TEST": "🧪"}.get(status, "📝")
    print(f"[{timestamp}] {emoji} {message}")

def parse_llm_verdict(analysis_text: str) -> Optional[bool]:
    """Parse LLM analysis to determine verdict"""
    if not analysis_text:
        return None
    
    analysis_lower = analysis_text.lower()
    
    if 'analysis failed' in analysis_lower or 'unknown model' in analysis_lower:
        return None
    
    import re
    verdict_pattern = r'verdict:\s*(normal|anomalous)'
    verdict_matches = re.findall(verdict_pattern, analysis_lower)
    
    if verdict_matches:
        last_verdict = verdict_matches[-1]
        return last_verdict == 'normal'
    
    header = analysis_lower[:200]
    
    if 'circuit is normal' in header or 'functioning correctly' in header:
        return True
    if 'circuit is anomalous' in header or 'has bugs' in header:
        return False
    
    if any(phrase in analysis_lower for phrase in [
        'stuck at 0', 'stuck at 1', 'hardcoded to', 
        'output does not change', 'signal is stuck'
    ]):
        return False
    
    return None

def estimate_confidence(analysis_text: str, verdict: Optional[bool]) -> float:
    """Estimate confidence from language"""
    if verdict is None:
        return 0.0
    
    analysis_lower = analysis_text.lower()
    if any(word in analysis_lower for word in ['clearly', 'definitely', 'certainly', 'obviously']):
        return 0.9
    elif any(word in analysis_lower for word in ['might', 'could', 'possibly', 'perhaps']):
        return 0.5
    elif any(word in analysis_lower for word in ['appears', 'seems', 'likely', 'probably']):
        return 0.7
    return 0.6

def test_circuit_with_model(name: str, verilog_code: str, expected_normal: bool, 
                            category: str, complexity: str, model: str, 
                            session_id: str) -> Optional[VerificationResult]:
    """Test a single circuit with specific LLM model + VAE"""
    
    model_info = MODELS_TO_TEST[model]
    log(f"Testing: {name} ({complexity}) with {model_info['name']}", "TEST")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/simulation/run-with-verification",
            json={
                "verilog_code": verilog_code,
                "model": model_info['endpoint_name'],
                "session_id": f"{session_id}_{model}"
            },
            timeout=120
        )
        
        if response.status_code != 200:
            log(f"API Error: {response.status_code}", "ERROR")
            return None
        
        data = response.json()
        if not data.get("success"):
            log(f"Simulation Failed", "ERROR")
            return None
        
        total_time = time.time() - start_time
        
        # Extract VAE results
        vae = data.get("verification", {}).get("vae_verification", {})
        vae_available = vae.get("available", False)
        vae_is_anomalous = vae.get("is_anomalous", False) if vae_available else None
        vae_predicted_normal = not vae_is_anomalous if vae_is_anomalous is not None else None
        vae_confidence = vae.get("confidence", 0.0) if vae_available else None
        vae_message = vae.get("message", "")
        vae_time = 1.5
        
        # Extract LLM results
        llm = data.get("verification", {}).get("llm_verification", {})
        llm_available = llm.get("available", False)
        llm_analysis = llm.get("analysis", "")
        
        llm_predicted_normal = None
        llm_confidence = 0.0
        if llm_available and llm_analysis:
            llm_predicted_normal = parse_llm_verdict(llm_analysis)
            if llm_predicted_normal is not None:
                llm_confidence = estimate_confidence(llm_analysis, llm_predicted_normal)
            else:
                log(f"  ⚠️ Could not parse {model_info['name']} verdict - ambiguous response", "WARNING")
        
        llm_time = total_time - vae_time
        
        # Calculate correctness
        vae_correct = (vae_predicted_normal == expected_normal) if vae_predicted_normal is not None else None
        llm_correct = (llm_predicted_normal == expected_normal) if llm_predicted_normal is not None else None
        
        result = VerificationResult(
            circuit_name=name,
            category=category,
            complexity=complexity,  # NEW
            expected_normal=expected_normal,
            vae_predicted_normal=vae_predicted_normal,
            vae_confidence=vae_confidence,
            vae_correct=vae_correct,
            vae_available=vae_available,
            vae_message=vae_message,
            llm_model=model,
            llm_predicted_normal=llm_predicted_normal,
            llm_confidence=llm_confidence,
            llm_correct=llm_correct,
            llm_available=llm_available,
            llm_analysis=llm_analysis[:500] if llm_analysis else "",
            llm_raw_response=llm_analysis[:1000] if llm_analysis else "",
            vae_time=vae_time,
            llm_time=llm_time,
            total_time=total_time
        )
        
        # Log comparison
        if vae_available and llm_available:
            vae_status = "✅" if vae_correct else "❌"
            llm_status = "✅" if llm_correct else ("❓" if llm_correct is None else "❌")
            log(f"  VAE: {vae_status} | {model_info['name']}: {llm_status}", "SUCCESS")
        
        return result
        
    except Exception as e:
        log(f"Exception: {e}", "ERROR")
        return None

def calculate_metrics(results: List[VerificationResult], filter_key: str = None) -> Dict:
    """Calculate accuracy metrics"""
    if filter_key:
        filtered = [r for r in results if getattr(r, filter_key) is not None]
    else:
        filtered = results
    
    if not filtered:
        return {"count": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    
    if filter_key and 'llm' in filter_key:
        correct_key = 'llm_correct'
        predicted_key = 'llm_predicted_normal'
    else:
        correct_key = 'vae_correct'
        predicted_key = 'vae_predicted_normal'
    
    correct = sum(1 for r in filtered if getattr(r, correct_key))
    accuracy = correct / len(filtered) * 100
    
    tp = sum(1 for r in filtered if not r.expected_normal and not getattr(r, predicted_key))
    tn = sum(1 for r in filtered if r.expected_normal and getattr(r, predicted_key))
    fp = sum(1 for r in filtered if r.expected_normal and not getattr(r, predicted_key))
    fn = sum(1 for r in filtered if not r.expected_normal and getattr(r, predicted_key))
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "count": len(filtered),
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }

def load_test_suites(suite_names: List[str]) -> Dict[str, List]:
    """Load requested test suites"""
    circuits = {}
    complexity_map = {}
    
    if 'simple' in suite_names:
        from integration.comprehensive_vae_test_suite import get_test_circuits
        simple_circuits = get_test_circuits()
        for category, tests in simple_circuits.items():
            circuits[category] = tests
            for name, _, _ in tests:
                complexity_map[name] = "simple"
    
    if 'medium' in suite_names:
        from integration.extended_test_suite import get_extended_test_circuits
        medium_circuits = get_extended_test_circuits()
        for category, tests in medium_circuits.items():
            if category in circuits:
                circuits[category].extend(tests)
            else:
                circuits[category] = tests
            for name, _, _ in tests:
                complexity_map[name] = "medium"
    
    if 'complex' in suite_names:
        from integration.complex_test_suite import get_complex_test_circuits
        complex_circuits = get_complex_test_circuits()
        for category, tests in complex_circuits.items():
            if category in circuits:
                circuits[category].extend(tests)
            else:
                circuits[category] = tests
            for name, _, _ in tests:
                complexity_map[name] = "complex"
    
    return circuits, complexity_map

def main():
    parser = argparse.ArgumentParser(description='Modular circuit complexity comparison')
    parser.add_argument('--suites', nargs='+', choices=['simple', 'medium', 'complex'],
                       required=True, help='Which test suites to run')
    args = parser.parse_args()
    
    suite_names = args.suites
    
    log("="*80, "TEST")
    log(f"MODULAR COMPLEXITY COMPARISON: {', '.join(suite_names).upper()}", "TEST")
    log("="*80, "TEST")
    
    # Load test suites
    circuits, complexity_map = load_test_suites(suite_names)
    
    total_circuits = sum(len(tests) for tests in circuits.values())
    total_tests = total_circuits * len(MODELS_TO_TEST)
    
    log(f"\nTest Suites: {', '.join(suite_names)}", "INFO")
    log(f"Total circuits: {total_circuits}", "INFO")
    log(f"Total tests: {total_tests} ({total_circuits} circuits × {len(MODELS_TO_TEST)} models)", "INFO")
    log(f"Estimated time: ~{total_tests * 8 / 60:.1f} minutes", "WARNING")
    log("="*80, "TEST")
    
    # Check backend
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            log("Backend not responding!", "ERROR")
            return
    except:
        log("Backend NOT running! Start with: uvicorn main:app --reload", "ERROR")
        return
    
    session_id = f"modular_test_{'_'.join(suite_names)}_{int(time.time())}"
    all_results = []
    test_num = 0
    
    for category, tests in circuits.items():
        log(f"\n{'='*80}", "TEST")
        log(f"CATEGORY: {category}", "TEST")
        log(f"{'='*80}", "TEST")
        
        for name, code, is_normal in tests:
            log(f"\n[Circuit {test_num//len(MODELS_TO_TEST) + 1}/{total_circuits}] {name}", "INFO")
            complexity = complexity_map.get(name, "unknown")
            
            for model_id in MODELS_TO_TEST.keys():
                test_num += 1
                log(f"  [{test_num}/{total_tests}] Testing with {MODELS_TO_TEST[model_id]['name']}...", "INFO")
                
                result = test_circuit_with_model(name, code, is_normal, category, 
                                                complexity, model_id, session_id)
                
                if result:
                    all_results.append(result)
                
                time.sleep(3)
    
    # ANALYSIS BY COMPLEXITY
    log("\n" + "="*80, "SUCCESS")
    log("RESULTS BY COMPLEXITY", "TEST")
    log("="*80, "SUCCESS")
    
    if not all_results:
        log("No successful tests!", "ERROR")
        return
    
    # Group by complexity
    results_by_complexity = {}
    for result in all_results:
        comp = result.complexity
        if comp not in results_by_complexity:
            results_by_complexity[comp] = []
        results_by_complexity[comp].append(result)
    
    # Print results for each complexity level
    for complexity in ['simple', 'medium', 'complex']:
        if complexity not in results_by_complexity:
            continue
        
        comp_results = results_by_complexity[complexity]
        
        log(f"\n{'='*80}", "INFO")
        log(f"COMPLEXITY LEVEL: {complexity.upper()}", "INFO")
        log(f"{'='*80}", "INFO")
        log(f"Circuits tested: {len(comp_results) // len(MODELS_TO_TEST)}", "INFO")
        
        # VAE metrics for this complexity
        vae_results = [r for r in comp_results if r.vae_available and r.vae_correct is not None]
        if vae_results:
            vae_metrics = calculate_metrics(vae_results, 'vae_correct')
            log(f"\nVAE: {vae_metrics['accuracy']:.1f}% accuracy "
                f"(P:{vae_metrics['precision']:.1f}% R:{vae_metrics['recall']:.1f}% F1:{vae_metrics['f1_score']:.1f}%)", "SUCCESS")
        
        # LLM metrics for this complexity
        for model_id, model_info in MODELS_TO_TEST.items():
            llm_results = [r for r in comp_results if r.llm_model == model_id and 
                          r.llm_available and r.llm_correct is not None]
            if llm_results:
                llm_metrics = calculate_metrics(llm_results, 'llm_correct')
                log(f"{model_info['name']}: {llm_metrics['accuracy']:.1f}% accuracy "
                    f"(P:{llm_metrics['precision']:.1f}% R:{llm_metrics['recall']:.1f}% F1:{llm_metrics['f1_score']:.1f}%)", "SUCCESS")
    
    # Overall comparison
    log(f"\n{'='*80}", "SUCCESS")
    log("OVERALL COMPARISON", "SUCCESS")
    log(f"{'='*80}", "SUCCESS")
    
    log(f"\n📊 ACCURACY BY COMPLEXITY:", "INFO")
    log(f"{'Model':<25} " + "  ".join([f"{c.capitalize():<10}" for c in ['simple', 'medium', 'complex'] if c in results_by_complexity]), "INFO")
    log("-" * 80, "INFO")
    
    # VAE row
    vae_row = "VAE"
    for complexity in ['simple', 'medium', 'complex']:
        if complexity in results_by_complexity:
            vae_results = [r for r in results_by_complexity[complexity] 
                          if r.vae_available and r.vae_correct is not None]
            if vae_results:
                vae_metrics = calculate_metrics(vae_results, 'vae_correct')
                vae_row += f"  {vae_metrics['accuracy']:>6.1f}%    "
        else:
            vae_row += "  " + " "*10
    log(vae_row, "INFO")
    
    # LLM rows
    for model_id, model_info in MODELS_TO_TEST.items():
        model_row = f"{model_info['name']:<25}"
        for complexity in ['simple', 'medium', 'complex']:
            if complexity in results_by_complexity:
                llm_results = [r for r in results_by_complexity[complexity] 
                              if r.llm_model == model_id and r.llm_available and r.llm_correct is not None]
                if llm_results:
                    llm_metrics = calculate_metrics(llm_results, 'llm_correct')
                    model_row += f"  {llm_metrics['accuracy']:>6.1f}%    "
            else:
                model_row += "  " + " "*10
        log(model_row, "INFO")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_suffix = "_".join(suite_names)
    
    json_filename = f"modular_comparison_{suite_suffix}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'summary': {
                'test_suites': suite_names,
                'total_tests': len(all_results),
                'total_circuits': total_circuits,
                'models_tested': list(MODELS_TO_TEST.keys()),
                'test_timestamp': timestamp
            },
            'detailed_results': [asdict(r) for r in all_results]
        }, f, indent=2)
    log(f"\n💾 Results saved to: {json_filename}", "SUCCESS")
    
    df = pd.DataFrame([asdict(r) for r in all_results])
    csv_filename = f"modular_comparison_{suite_suffix}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    log(f"📊 CSV saved to: {csv_filename}", "SUCCESS")
    
    log("\n" + "="*80, "SUCCESS")
    log("TEST COMPLETE!", "SUCCESS")
    log("="*80, "SUCCESS")

if __name__ == "__main__":
    main()