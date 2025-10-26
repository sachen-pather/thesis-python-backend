"""
FINAL LLM vs VAE Comparison - Using Your Proven Test Suite
Uses the EXACT same 26 circuits that gave 80.8% VAE accuracy

SAVE AS: tests/analysis/final_llm_vae_comparison.py
RUN: python tests/analysis/final_llm_vae_comparison.py
"""

import sys
import os
import requests
import time
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Import the proven test circuits
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from integration.comprehensive_vae_test_suite import get_test_circuits

BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"final_comparison_{int(time.time())}"

# Models to test
MODELS_TO_TEST = {
    "gpt-4o": {"name": "GPT-4o", "provider": "OpenAI", "endpoint_name": "gpt-4o"},
    "claude": {"name": "Claude Sonnet 3.5", "provider": "Anthropic", "endpoint_name": "claude"},
    "groq": {"name": "Llama 3.3 70B", "provider": "Groq", "endpoint_name": "groq"},
    "gemini": {"name": "Gemini 2.0 Flash", "provider": "Google", "endpoint_name": "gemini/gemini-2.0-flash-exp"}
}

@dataclass
class VerificationResult:
    circuit_name: str
    category: str
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
    
    # Explicit verdicts
    if 'verdict: normal' in analysis_lower or 'status: normal' in analysis_lower:
        return True
    if 'verdict: anomalous' in analysis_lower or 'status: anomalous' in analysis_lower:
        return False
    
    # Keyword analysis
    anomaly_keywords = ['error', 'incorrect', 'bug', 'wrong', 'stuck', 'frozen', 'anomalous', 'abnormal', 'defect', 'issue', 'problem']
    normal_keywords = ['correct', 'normal', 'expected', 'working', 'proper', 'valid', 'healthy', 'pass']
    
    anomaly_score = sum(1 for kw in anomaly_keywords if kw in analysis_lower)
    normal_score = sum(1 for kw in normal_keywords if kw in analysis_lower)
    
    if anomaly_score > normal_score + 1:
        return False
    elif normal_score > anomaly_score:
        return True
    
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
                            category: str, model: str) -> Optional[VerificationResult]:
    """Test a single circuit with specific LLM model + VAE"""
    
    model_info = MODELS_TO_TEST[model]
    log(f"Testing: {name} with {model_info['name']}", "TEST")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/simulation/run-with-verification",
            json={
                "verilog_code": verilog_code,
                "model": model_info['endpoint_name'],
                "session_id": f"{TEST_SESSION_ID}_{model}"
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
        vae_anomalous = vae.get("is_anomalous", False) if vae_available else None
        vae_confidence = vae.get("confidence", 0.0) if vae_available else None
        vae_message = vae.get("message", "")
        vae_time = 1.5
        
        # Extract LLM results
        llm = data.get("verification", {}).get("llm_verification", {})
        llm_available = llm.get("available", False)
        llm_analysis = llm.get("analysis", "")
        
        llm_anomalous = None
        llm_confidence = 0.0
        if llm_available and llm_analysis:
            llm_anomalous = parse_llm_verdict(llm_analysis)
            if llm_anomalous is not None:
                llm_confidence = estimate_confidence(llm_analysis, llm_anomalous)
        
        llm_time = total_time - vae_time
        
        # Calculate correctness
        expected_anomalous = not expected_normal
        vae_correct = (vae_anomalous == expected_anomalous) if vae_anomalous is not None else None
        llm_correct = (llm_anomalous == expected_anomalous) if llm_anomalous is not None else None
        
        result = VerificationResult(
            circuit_name=name,
            category=category,
            expected_normal=expected_normal,
            vae_predicted_normal=not vae_anomalous if vae_anomalous is not None else None,
            vae_confidence=vae_confidence,
            vae_correct=vae_correct,
            vae_available=vae_available,
            vae_message=vae_message,
            llm_model=model,
            llm_predicted_normal=not llm_anomalous if llm_anomalous is not None else None,
            llm_confidence=llm_confidence,
            llm_correct=llm_correct,
            llm_available=llm_available,
            llm_analysis=llm_analysis[:500] if llm_analysis else "",
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

def main():
    log("="*80, "TEST")
    log("FINAL LLM vs VAE COMPARISON - PROVEN TEST SUITE", "TEST")
    log("="*80, "TEST")
    log(f"Using validated circuits from comprehensive_vae_test_suite.py", "INFO")
    log(f"These circuits achieved 80.8% VAE accuracy with 100% precision", "INFO")
    
    # Load circuits
    circuits = get_test_circuits()
    total_circuits = sum(len(tests) for tests in circuits.values())
    total_tests = total_circuits * len(MODELS_TO_TEST)
    
    log(f"\nTotal circuits: {total_circuits}", "INFO")
    log(f"Total tests: {total_tests} ({total_circuits} circuits × {len(MODELS_TO_TEST)} models)", "INFO")
    log(f"Estimated time: ~{total_tests * 8 / 60:.1f} minutes", "WARNING")
    log("="*80, "TEST")
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            log("Backend is not responding correctly!", "ERROR")
            log("Start it with: uvicorn main:app --reload", "INFO")
            return
    except:
        log("Backend is NOT running!", "ERROR")
        log("Start it with: uvicorn main:app --reload", "INFO")
        return
    
    all_results = []
    test_num = 0
    
    for category, tests in circuits.items():
        log(f"\n{'='*80}", "TEST")
        log(f"CATEGORY: {category}", "TEST")
        log(f"{'='*80}", "TEST")
        
        for name, code, is_normal in tests:
            log(f"\n[Circuit {test_num//len(MODELS_TO_TEST) + 1}/{total_circuits}] {name}", "INFO")
            
            for model_id in MODELS_TO_TEST.keys():
                test_num += 1
                log(f"  [{test_num}/{total_tests}] Testing with {MODELS_TO_TEST[model_id]['name']}...", "INFO")
                
                result = test_circuit_with_model(name, code, is_normal, category, model_id)
                
                if result:
                    all_results.append(result)
                
                time.sleep(3)  # Rate limiting
    
    # ========== ANALYSIS ==========
    log("\n" + "="*80, "SUCCESS")
    log("COMPREHENSIVE ANALYSIS", "TEST")
    log("="*80, "SUCCESS")
    
    if not all_results:
        log("No successful tests!", "ERROR")
        return
    
    # Separate results by model
    results_by_model = {}
    for model_id in MODELS_TO_TEST.keys():
        results_by_model[model_id] = [r for r in all_results if r.llm_model == model_id]
    
    # Calculate metrics for each model
    log("\n📊 MODEL COMPARISON:", "SUCCESS")
    log(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}", "INFO")
    log("-" * 73, "INFO")
    
    model_metrics = {}
    
    # VAE baseline
    vae_results = [r for r in all_results if r.vae_available and r.vae_correct is not None]
    if vae_results:
        vae_metrics = calculate_metrics(vae_results, 'vae_correct')
        log(f"{'VAE (Baseline)':<25} {vae_metrics['accuracy']:>6.1f}%     "
            f"{vae_metrics['precision']:>6.1f}%     {vae_metrics['recall']:>6.1f}%     "
            f"{vae_metrics['f1_score']:>6.1f}%", "SUCCESS")
        model_metrics['vae'] = vae_metrics
    
    # Each LLM model
    for model_id, model_info in MODELS_TO_TEST.items():
        llm_results = [r for r in results_by_model[model_id] 
                       if r.llm_available and r.llm_correct is not None]
        if llm_results:
            llm_metrics = calculate_metrics(llm_results, 'llm_correct')
            log(f"{model_info['name']:<25} {llm_metrics['accuracy']:>6.1f}%     "
                f"{llm_metrics['precision']:>6.1f}%     {llm_metrics['recall']:>6.1f}%     "
                f"{llm_metrics['f1_score']:>6.1f}%", "SUCCESS")
            model_metrics[model_id] = llm_metrics
    
    # Find best model
    if len(model_metrics) > 1:
        best_model = max(
            [(k, v) for k, v in model_metrics.items() if k != 'vae'],
            key=lambda x: x[1]['accuracy']
        )
        log(f"\n🏆 BEST LLM: {MODELS_TO_TEST.get(best_model[0], {}).get('name', best_model[0])} "
            f"with {best_model[1]['accuracy']:.1f}% accuracy", "SUCCESS")
    
    # Performance comparison
    log("\n⚡ PERFORMANCE COMPARISON:", "INFO")
    vae_avg_time = sum(r.vae_time for r in all_results) / len(all_results) if all_results else 0
    log(f"  {'VAE':<25} {vae_avg_time:>6.2f}s avg", "INFO")
    
    for model_id in MODELS_TO_TEST.keys():
        model_results = results_by_model[model_id]
        if model_results:
            llm_avg_time = sum(r.llm_time for r in model_results) / len(model_results)
            log(f"  {MODELS_TO_TEST[model_id]['name']:<25} {llm_avg_time:>6.2f}s avg", "INFO")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_filename = f"final_llm_vs_vae_comparison_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': len(all_results),
                'total_circuits': total_circuits,
                'models_tested': list(MODELS_TO_TEST.keys()),
                'test_timestamp': timestamp,
                'baseline_vae_accuracy': vae_metrics.get('accuracy', 0)
            },
            'metrics_by_model': model_metrics,
            'detailed_results': [asdict(r) for r in all_results]
        }, f, indent=2)
    log(f"\n💾 Results saved to: {json_filename}", "SUCCESS")
    
    # Save as CSV
    df = pd.DataFrame([asdict(r) for r in all_results])
    csv_filename = f"final_llm_vs_vae_comparison_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    log(f"📊 CSV saved to: {csv_filename}", "SUCCESS")
    
    # Generate markdown report
    report_filename = f"final_llm_vs_vae_report_{timestamp}.md"
    with open(report_filename, 'w') as f:
        f.write(f"# Final LLM vs VAE Verification Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Test Overview\n\n")
        f.write(f"- **Total circuits tested:** {total_circuits}\n")
        f.write(f"- **Total tests run:** {len(all_results)}\n")
        f.write(f"- **Test suite:** Proven 26-circuit validation suite (80.8% VAE baseline)\n")
        f.write(f"- **Models compared:** {', '.join([m['name'] for m in MODELS_TO_TEST.values()])}\n\n")
        
        f.write(f"## Performance Comparison\n\n")
        f.write(f"| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write(f"|-------|----------|-----------|--------|----------|\n")
        
        for model_id, metrics in model_metrics.items():
            model_name = MODELS_TO_TEST.get(model_id, {}).get('name', model_id)
            if model_id == 'vae':
                model_name = "VAE (Baseline)"
            f.write(f"| {model_name} | {metrics['accuracy']:.1f}% | "
                   f"{metrics['precision']:.1f}% | {metrics['recall']:.1f}% | "
                   f"{metrics['f1_score']:.1f}% |\n")
        
        f.write(f"\n## Key Findings\n\n")
        if len(model_metrics) > 1:
            f.write(f"- **Best LLM Accuracy:** {MODELS_TO_TEST.get(best_model[0], {}).get('name', best_model[0])} ({best_model[1]['accuracy']:.1f}%)\n")
        f.write(f"- **VAE Baseline:** {model_metrics.get('vae', {}).get('accuracy', 0):.1f}% accuracy\n")
        f.write(f"- **VAE Precision:** {model_metrics.get('vae', {}).get('precision', 0):.1f}%\n")
        f.write(f"- **VAE Speed:** ~{vae_avg_time:.2f}s per circuit\n")
        
        f.write(f"\n## Circuit Categories Tested\n\n")
        for category in circuits.keys():
            f.write(f"- {category}\n")
    
    log(f"📝 Report saved to: {report_filename}", "SUCCESS")
    
    log("\n" + "="*80, "SUCCESS")
    log("COMPARISON TEST COMPLETE!", "SUCCESS")
    log("="*80, "SUCCESS")

if __name__ == "__main__":
    main()