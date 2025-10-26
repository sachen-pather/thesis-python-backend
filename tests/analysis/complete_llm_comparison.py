"""
Complete LLM Verilog Generation & Verification Comparison
Tests BOTH generation quality AND verification accuracy for Claude vs GPT-4o
Uses 2-of-3 consensus voting (VAE + Claude + GPT-4o)

SAVE AS: tests/analysis/complete_llm_comparison.py
RUN: python tests/analysis/complete_llm_comparison.py --suites simple medium complex
"""

import sys
import os
import argparse
import requests
import time
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_URL = "http://localhost:8000"

MODELS = {
    "claude": {"name": "Claude Sonnet 3.5", "provider": "Anthropic"},
    "gpt-4o": {"name": "GPT-4o", "provider": "OpenAI"}
}

@dataclass
class GenerationResult:
    """Complete results for one model's generation and verification"""
    prompt: str
    circuit_name: str
    category: str
    complexity: str
    generator_model: str
    
    # Generation metrics
    generated_verilog: str
    generation_success: bool
    lines_of_code: int
    has_testbench: bool
    compilation_success: bool
    
    # Simulation
    simulation_success: bool
    waveform_csv: Optional[str]
    
    # Triple verification (True = anomalous, False = normal)
    vae_verdict: Optional[bool]
    vae_confidence: float
    claude_verdict: Optional[bool]
    claude_confidence: float
    claude_analysis: str
    gpt4o_verdict: Optional[bool]
    gpt4o_confidence: float
    gpt4o_analysis: str
    
    # 2-of-3 consensus voting result
    consensus_anomalous: Optional[bool]
    consensus_confidence: float
    
    # Timing
    generation_time: float
    simulation_time: float
    verification_time: float

def log(message, status="INFO"):
    """Logging with timestamps and emoji"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji_map = {
        "INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", 
        "WARNING": "⚠️", "TEST": "🧪", "GENERATE": "🤖",
        "SIMULATE": "⚡", "VERIFY": "🔍", "METRIC": "📊"
    }
    emoji = emoji_map.get(status, "📝")
    print(f"[{timestamp}] {emoji} {message}")

def calculate_consensus(vae_anom: Optional[bool], claude_anom: Optional[bool], 
                       gpt4o_anom: Optional[bool]) -> Tuple[Optional[bool], float]:
    """
    2-of-3 voting: If at least 2 verifiers say anomalous, it's anomalous
    Returns: (consensus_verdict, confidence_score)
    
    Note: LLM verdicts are inverted - they return True for normal, False for anomalous
          VAE returns True for anomalous, False for normal
    """
    # Convert LLM verdicts to match VAE convention (True = anomalous)
    verdicts = []
    
    if vae_anom is not None:
        verdicts.append(vae_anom)
    
    if claude_anom is not None:
        # Invert: Claude returns True=normal, False=anomalous
        verdicts.append(not claude_anom)
    
    if gpt4o_anom is not None:
        # Invert: GPT-4o returns True=normal, False=anomalous
        verdicts.append(not gpt4o_anom)
    
    if len(verdicts) < 2:
        return None, 0.0
    
    # Count votes
    anomalous_votes = sum(verdicts)
    total_votes = len(verdicts)
    
    # 2-of-3 or 2-of-2 voting
    is_anomalous = anomalous_votes >= (total_votes / 2)
    confidence = anomalous_votes / total_votes
    
    return is_anomalous, confidence

def get_circuit_prompts(suites: List[str]) -> List[Tuple[str, str, str, str]]:
    """Get circuit prompts: 4 simple + 10 medium + 8 complex"""
    prompts = []
    
    circuit_to_prompt = {
        # ==================== SIMPLE (4 circuits) ====================
        "2-Input AND": (
            "Design a 2-input AND gate in Verilog with inputs a, b and output out. Include a testbench that tests all 4 input combinations.",
            "Combinational - Normal", "simple"
        ),
        "2-Input XOR": (
            "Design a 2-input XOR gate in Verilog. Use inputs a, b and output out. Include testbench with all combinations.",
            "Combinational - Normal", "simple"
        ),
        "D Flip-Flop": (
            "Create a D flip-flop with clock, reset, d input, and q output. Reset should be asynchronous active high. Include testbench.",
            "Sequential - Normal", "simple"
        ),
        "Half Adder": (
            "Create a half adder with inputs a, b and outputs sum, carry. Include testbench for all combinations.",
            "Arithmetic - Normal", "simple"
        ),
        
        # ==================== MEDIUM (10 circuits) ====================
        "4-bit Incrementer": (
            "Design a 4-bit incrementer that adds 1 to input a[3:0] and outputs result out[3:0] plus overflow bit. Include testbench testing values 0, 7, 14, 15.",
            "Combinational - Normal", "medium"
        ),
        "4-bit Comparator": (
            "Create a 4-bit magnitude comparator with inputs a[3:0], b[3:0] and outputs eq (equal), gt (greater than), lt (less than). Include comprehensive testbench.",
            "Combinational - Normal", "medium"
        ),
        "4-to-2 Priority Encoder": (
            "Design a 4-to-2 priority encoder with input in[3:0], output out[1:0], and valid bit. Highest bit has priority. Include testbench with all cases.",
            "Combinational - Normal", "medium"
        ),
        "2-to-4 Decoder": (
            "Create a 2-to-4 decoder with input in[1:0], enable signal, and output out[3:0]. Only one output bit is high based on input when enabled. Include testbench.",
            "Combinational - Normal", "medium"
        ),
        "4:1 Multiplexer": (
            "Design a 4:1 multiplexer with input in[3:0], select sel[1:0], and output out. Include testbench testing all selections.",
            "Combinational - Normal", "medium"
        ),
        "8-bit Parity Generator": (
            "Create an 8-bit even parity generator with input data[7:0] and output parity. Include testbench with various test vectors.",
            "Combinational - Normal", "medium"
        ),
        "4-bit UpDown Counter": (
            "Create a 4-bit up/down counter with clock, reset, up (direction), enable, and count[3:0] output. Counts up when up=1, down when up=0. Include testbench.",
            "Sequential - Normal", "medium"
        ),
        "4-bit Ring Counter": (
            "Design a 4-bit ring counter with clock, reset, and q[3:0] output. Initialize to 0001 on reset, rotate left each clock. Include testbench showing full cycle.",
            "Sequential - Normal", "medium"
        ),
        "4-bit LFSR": (
            "Create a 4-bit Linear Feedback Shift Register with clock, reset, and q[3:0] output. Use XOR feedback from bits 3 and 2. Initialize to 0001. Include testbench.",
            "Sequential - Normal", "medium"
        ),
        "Full Adder": (
            "Design a full adder with inputs a, b, cin and outputs sum, cout. Include testbench testing all 8 combinations.",
            "Arithmetic - Normal", "medium"
        ),
        
        # ==================== COMPLEX (8 circuits) ====================
        "Traffic Light Controller": (
            "Design a traffic light controller FSM with 4 states: NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW. "
            "Inputs: clk, rst, emergency. Outputs: ns_light[1:0], ew_light[1:0] where 00=RED, 01=YELLOW, 10=GREEN. "
            "Timing: GREEN lasts 8 clock cycles, YELLOW lasts 2 cycles. Emergency input makes both lights RED immediately. "
            "Use a counter for state timing. Include comprehensive testbench showing full cycle and emergency.",
            "State Machines - Normal", "complex"
        ),
        "Sequence Detector (1011)": (
            "Create a sequence detector FSM that detects the pattern 1011 in serial input din. "
            "Inputs: clk, rst, din. Output: detected (pulses high for one cycle when pattern is found). "
            "Use overlapping detection (new pattern can start before previous ends). Include testbench with multiple pattern occurrences.",
            "State Machines - Normal", "complex"
        ),
        "Simple UART Transmitter": (
            "Design a UART transmitter with states IDLE, START, DATA, STOP. "
            "Inputs: clk, rst, start (trigger), data[7:0]. Outputs: tx (serial output), busy. "
            "Protocol: Send START bit (0), then 8 data bits LSB first, then STOP bit (1). "
            "Busy should be high during transmission. Include testbench sending 0xA5.",
            "State Machines - Normal", "complex"
        ),
        "Vending Machine FSM": (
            "Design a vending machine FSM that accepts 5-cent and 10-cent coins. Item costs 15 cents. "
            "Inputs: clk, rst, nickel (5c), dime (10c). Outputs: dispense, change[1:0]. "
            "States track accumulated amount (0c, 5c, 10c, 15c). "
            "Output change if overpayment. Include testbench with various payment scenarios.",
            "State Machines - Normal", "complex"
        ),
        "8-bit Register File": (
            "Create an 8-bit register file with 4 registers. "
            "Inputs: clk, we (write enable), rd_addr1[1:0], rd_addr2[1:0], wr_addr[1:0], wr_data[7:0]. "
            "Outputs: rd_data1[7:0], rd_data2[7:0] (dual read ports). "
            "Include testbench with simultaneous read/write operations.",
            "CPU Components - Normal", "complex"
        ),
        "Simple ALU with Flags": (
            "Design an 8-bit ALU with inputs a[7:0], b[7:0], op[2:0] and outputs result[7:0], zero, carry, negative. "
            "Operations: 000=ADD, 001=SUB, 010=AND, 011=OR, 100=XOR, 101=NOT a, 110=SHL, 111=SHR. "
            "Flags: zero (result==0), carry (from add/sub), negative (result[7]). "
            "Include testbench testing all operations.",
            "CPU Components - Normal", "complex"
        ),
        "4-bit Johnson Counter": (
            "Design a 4-bit Johnson counter (twisted ring counter) with clock, reset, and q[3:0] output. "
            "Shifts in complement of MSB. Sequence should be: 0000, 1000, 1100, 1110, 1111, 0111, 0011, 0001, then repeat. "
            "Include testbench showing 2 complete cycles.",
            "Sequential - Normal", "complex"
        ),
        "Edge Detector": (
            "Create a positive edge detector that outputs a single-cycle pulse when input signal transitions from 0 to 1. "
            "Inputs: clk, rst, signal. Output: pulse. "
            "Use a register to store previous value and detect rising edge. Include testbench with multiple edges.",
            "Sequential - Normal", "complex"
        ),
    }
    
    for name, (prompt, category, complexity) in circuit_to_prompt.items():
        if complexity in suites:
            prompts.append((name, prompt, category, complexity))
    
    return prompts

def generate_verilog_with_model(prompt: str, model: str, session_id: str) -> Tuple[bool, str, Dict]:
    """Generate Verilog code using specified model"""
    try:
        start_time = time.time()
        
        # Step 1: Generate Mermaid
        mermaid_response = requests.post(
            f"{BASE_URL}/api/design/generate-mermaid",
            json={"prompt": prompt, "model": model, "session_id": session_id},
            timeout=60
        )
        
        if mermaid_response.status_code != 200:
            return False, "", {"error": "Mermaid generation failed", "generation_time": time.time() - start_time}
        
        mermaid_data = mermaid_response.json()
        if not mermaid_data.get("success"):
            return False, "", {"error": mermaid_data.get("error", "Unknown"), "generation_time": time.time() - start_time}
        
        mermaid_code = mermaid_data.get("mermaid_code", "")
        
        # Step 2: Generate Verilog
        verilog_response = requests.post(
            f"{BASE_URL}/api/design/generate-verilog",
            json={
                "mermaid_code": mermaid_code,
                "description": prompt,
                "model": model,
                "use_rag": False,
                "session_id": session_id
            },
            timeout=60
        )
        
        if verilog_response.status_code != 200:
            return False, "", {"error": "Verilog generation failed", "generation_time": time.time() - start_time}
        
        verilog_data = verilog_response.json()
        if not verilog_data.get("success"):
            return False, "", {"error": verilog_data.get("error", "Unknown"), "generation_time": time.time() - start_time}
        
        verilog_code = verilog_data.get("verilog_code", "")
        stats = verilog_data.get("stats", {})
        stats["generation_time"] = time.time() - start_time
        
        return True, verilog_code, stats
        
    except Exception as e:
        return False, "", {"error": str(e), "generation_time": time.time() - start_time}

def simulate_and_verify(verilog_code: str, verifier_model: str, session_id: str) -> Dict:
    """Simulate code and get verification from specified verifier"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/simulation/run-with-verification",
            json={
                "verilog_code": verilog_code,
                "model": verifier_model,
                "session_id": session_id
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return {"success": False, "error": f"API error {response.status_code}"}
        
        return response.json()
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def parse_llm_verdict(analysis_text: str) -> Tuple[Optional[bool], float]:
    """Parse LLM analysis to get verdict (True=normal, False=anomalous) and confidence"""
    if not analysis_text:
        return None, 0.0
    
    analysis_lower = analysis_text.lower()
    
    # Look for explicit verdict
    if 'verdict: normal' in analysis_lower or 'verdict:normal' in analysis_lower:
        is_normal = True
    elif 'verdict: anomalous' in analysis_lower or 'verdict:anomalous' in analysis_lower:
        is_normal = False
    else:
        # Infer from content
        if any(phrase in analysis_lower for phrase in [
            'circuit is normal', 'functioning correctly', 'behaves as expected',
            'working properly', 'correct behavior', 'operates correctly'
        ]):
            is_normal = True
        elif any(phrase in analysis_lower for phrase in [
            'circuit is anomalous', 'has bugs', 'incorrect', 'stuck at',
            'not working', 'faulty', 'broken', 'does not match'
        ]):
            is_normal = False
        else:
            return None, 0.0
    
    # Estimate confidence
    if any(word in analysis_lower for word in ['clearly', 'definitely', 'certainly', 'obviously']):
        confidence = 0.9
    elif any(word in analysis_lower for word in ['likely', 'probably', 'appears']):
        confidence = 0.7
    elif any(word in analysis_lower for word in ['might', 'could', 'possibly']):
        confidence = 0.5
    else:
        confidence = 0.6
    
    return is_normal, confidence

def test_circuit_complete(name: str, prompt: str, category: str, 
                         complexity: str, session_id: str) -> Dict[str, GenerationResult]:
    """Complete test for one circuit with both models"""
    log(f"\n{'='*80}", "TEST")
    log(f"CIRCUIT: {name} ({complexity})", "TEST")
    log(f"{'='*80}", "TEST")
    
    results = {}
    
    for model_id in MODELS.keys():
        log(f"\n🤖 Generating with {MODELS[model_id]['name']}...", "GENERATE")
        
        # Step 1: Generate Verilog
        gen_success, verilog_code, gen_stats = generate_verilog_with_model(
            prompt, model_id, f"{session_id}_{model_id}_{name.replace(' ', '_')}"
        )
        
        if not gen_success:
            log(f"❌ Generation failed: {gen_stats.get('error', 'Unknown')}", "ERROR")
            results[model_id] = GenerationResult(
                prompt=prompt, circuit_name=name, category=category,
                complexity=complexity, generator_model=model_id,
                generated_verilog="", generation_success=False,
                lines_of_code=0, has_testbench=False, compilation_success=False,
                simulation_success=False, waveform_csv=None,
                vae_verdict=None, vae_confidence=0.0,
                claude_verdict=None, claude_confidence=0.0, claude_analysis="",
                gpt4o_verdict=None, gpt4o_confidence=0.0, gpt4o_analysis="",
                consensus_anomalous=None, consensus_confidence=0.0,
                generation_time=gen_stats.get('generation_time', 0),
                simulation_time=0, verification_time=0
            )
            continue
        
        log(f"✅ Generated {gen_stats.get('lines', 0)} lines", "SUCCESS")
        
        # Step 2: Simulate with self-verification
        log(f"⚡ Simulating with {MODELS[model_id]['name']} verification...", "SIMULATE")
        sim_start = time.time()
        
        sim_results = simulate_and_verify(
            verilog_code, model_id, 
            f"{session_id}_{model_id}_{name.replace(' ', '_')}_self"
        )
        
        sim_time = time.time() - sim_start
        
        if not sim_results.get("success"):
            log(f"❌ Simulation failed: {sim_results.get('error', 'Unknown')}", "ERROR")
            results[model_id] = GenerationResult(
                prompt=prompt, circuit_name=name, category=category,
                complexity=complexity, generator_model=model_id,
                generated_verilog=verilog_code, generation_success=True,
                lines_of_code=gen_stats.get('lines', 0),
                has_testbench=gen_stats.get('has_testbench', False),
                compilation_success=False, simulation_success=False,
                waveform_csv=None,
                vae_verdict=None, vae_confidence=0.0,
                claude_verdict=None, claude_confidence=0.0, claude_analysis="",
                gpt4o_verdict=None, gpt4o_confidence=0.0, gpt4o_analysis="",
                consensus_anomalous=None, consensus_confidence=0.0,
                generation_time=gen_stats.get('generation_time', 0),
                simulation_time=sim_time, verification_time=0
            )
            continue
        
        log(f"✅ Simulation successful", "SUCCESS")
        
        # Extract VAE results
        vae_verif = sim_results.get("verification", {}).get("vae_verification", {})
        vae_available = vae_verif.get("available", False)
        vae_anomalous = vae_verif.get("is_anomalous", False) if vae_available else None
        vae_confidence = vae_verif.get("confidence", 0.0) if vae_available else 0.0
        
        if vae_available:
            status = "🔴 ANOMALOUS" if vae_anomalous else "✅ NORMAL"
            log(f"  VAE: {status} (conf: {vae_confidence:.3f})", "VERIFY")
        
        # Extract self-verification
        llm_verif = sim_results.get("verification", {}).get("llm_verification", {})
        self_analysis = llm_verif.get("analysis", "")
        self_is_normal, self_confidence = parse_llm_verdict(self_analysis)
        
        if self_is_normal is not None:
            status = "✅ NORMAL" if self_is_normal else "🔴 ANOMALOUS"
            log(f"  {MODELS[model_id]['name']} (self): {status} (conf: {self_confidence:.2f})", "VERIFY")
        
        # Step 3: Cross-verification
        other_model = "gpt-4o" if model_id == "claude" else "claude"
        log(f"🔍 Cross-verification with {MODELS[other_model]['name']}...", "VERIFY")
        
        verify_start = time.time()
        cross_results = simulate_and_verify(
            verilog_code, other_model,
            f"{session_id}_{model_id}_{name.replace(' ', '_')}_cross_{other_model}"
        )
        verify_time = time.time() - verify_start
        
        # Extract cross-verification
        cross_llm = cross_results.get("verification", {}).get("llm_verification", {})
        cross_analysis = cross_llm.get("analysis", "")
        cross_is_normal, cross_confidence = parse_llm_verdict(cross_analysis)
        
        if cross_is_normal is not None:
            status = "✅ NORMAL" if cross_is_normal else "🔴 ANOMALOUS"
            log(f"  {MODELS[other_model]['name']} (cross): {status} (conf: {cross_confidence:.2f})", "VERIFY")
        
        # Organize results by verifier
        if model_id == "claude":
            claude_verdict = self_is_normal
            claude_conf = self_confidence
            claude_analysis = self_analysis
            gpt4o_verdict = cross_is_normal
            gpt4o_conf = cross_confidence
            gpt4o_analysis = cross_analysis
        else:
            gpt4o_verdict = self_is_normal
            gpt4o_conf = self_confidence
            gpt4o_analysis = self_analysis
            claude_verdict = cross_is_normal
            claude_conf = cross_confidence
            claude_analysis = cross_analysis
        
        # Calculate 2-of-3 consensus
        consensus, consensus_conf = calculate_consensus(vae_anomalous, claude_verdict, gpt4o_verdict)
        
        if consensus is not None:
            status = "🔴 ANOMALOUS" if consensus else "✅ NORMAL"
            log(f"  🗳️  2-of-3 Consensus: {status} (conf: {consensus_conf:.2f})", "VERIFY")
        
        results[model_id] = GenerationResult(
            prompt=prompt, circuit_name=name, category=category,
            complexity=complexity, generator_model=model_id,
            generated_verilog=verilog_code, generation_success=True,
            lines_of_code=gen_stats.get('lines', 0),
            has_testbench=gen_stats.get('has_testbench', False),
            compilation_success=True, simulation_success=True,
            waveform_csv=sim_results.get("waveform_csv", ""),
            vae_verdict=vae_anomalous, vae_confidence=vae_confidence,
            claude_verdict=claude_verdict, claude_confidence=claude_conf,
            claude_analysis=claude_analysis[:500],
            gpt4o_verdict=gpt4o_verdict, gpt4o_confidence=gpt4o_conf,
            gpt4o_analysis=gpt4o_analysis[:500],
            consensus_anomalous=consensus, consensus_confidence=consensus_conf,
            generation_time=gen_stats.get('generation_time', 0),
            simulation_time=sim_time, verification_time=verify_time
        )
        
        time.sleep(2)
    
    return results

def calculate_generation_metrics(all_results: List[GenerationResult]) -> Dict:
    """Calculate generation quality metrics"""
    metrics = {}
    
    for model in MODELS.keys():
        model_results = [r for r in all_results if r.generator_model == model]
        
        if not model_results:
            continue
        
        total = len(model_results)
        successful_gen = sum(1 for r in model_results if r.generation_success)
        compiled = sum(1 for r in model_results if r.compilation_success)
        simulated = sum(1 for r in model_results if r.simulation_success)
        has_tb = sum(1 for r in model_results if r.has_testbench)
        
        # Functional correctness = consensus says normal (not anomalous)
        functional_correct = sum(1 for r in model_results 
                                if r.consensus_anomalous is False)
        
        avg_loc = sum(r.lines_of_code for r in model_results) / total if total > 0 else 0
        
        metrics[model] = {
            "total_circuits": total,
            "generation_success_rate": successful_gen / total * 100,
            "compilation_rate": compiled / total * 100,
            "simulation_rate": simulated / total * 100,
            "testbench_inclusion": has_tb / total * 100,
            "functional_correctness": functional_correct / total * 100,
            "avg_lines_of_code": avg_loc
        }
    
    return metrics

def generate_report(all_results: List[GenerationResult], 
                   gen_metrics: Dict, suites: List[str], session_id: str):
    """Generate comprehensive comparison report"""
    
    log("\n" + "="*80, "METRIC")
    log("COMPLETE LLM COMPARISON REPORT", "METRIC")
    log("="*80, "METRIC")
    
    log(f"\nTest Suites: {', '.join(suites)}", "INFO")
    log(f"Total Circuits Tested: {len(all_results) // 2}", "INFO")
    log(f"Session ID: {session_id}", "INFO")
    
    # ========== GENERATION QUALITY ==========
    log("\n" + "="*80, "METRIC")
    log("📊 GENERATION QUALITY COMPARISON", "METRIC")
    log("="*80, "METRIC")
    
    log("\nOverall Metrics:", "INFO")
    log(f"{'Metric':<30} {'Claude':<15} {'GPT-4o':<15}", "INFO")
    log("-" * 60, "INFO")
    
    metrics_to_show = [
        ("Generation Success", "generation_success_rate"),
        ("Compilation Rate", "compilation_rate"),
        ("Simulation Rate", "simulation_rate"),
        ("Testbench Inclusion", "testbench_inclusion"),
        ("Functional Correctness", "functional_correctness"),
        ("Avg Lines of Code", "avg_lines_of_code")
    ]
    
    for label, key in metrics_to_show:
        claude_val = gen_metrics.get("claude", {}).get(key, 0)
        gpt4o_val = gen_metrics.get("gpt-4o", {}).get(key, 0)
        
        if "Code" in label:
            log(f"{label:<30} {claude_val:<15.0f} {gpt4o_val:<15.0f}", "INFO")
        else:
            log(f"{label:<30} {claude_val:<15.1f}% {gpt4o_val:<15.1f}%", "INFO")
    
    # By complexity
    log("\n📊 By Complexity:", "INFO")
    log(f"{'Complexity':<15} {'Claude Correctness':<25} {'GPT-4o Correctness':<25}", "INFO")
    log("-" * 65, "INFO")
    
    for complexity in ['simple', 'medium', 'complex']:
        if complexity not in suites:
            continue
        
        claude_results = [r for r in all_results if r.generator_model == "claude" and r.complexity == complexity]
        gpt4o_results = [r for r in all_results if r.generator_model == "gpt-4o" and r.complexity == complexity]
        
        if claude_results:
            claude_correct = sum(1 for r in claude_results if r.consensus_anomalous is False)
            claude_pct = claude_correct / len(claude_results) * 100
        else:
            claude_pct = 0
        
        if gpt4o_results:
            gpt4o_correct = sum(1 for r in gpt4o_results if r.consensus_anomalous is False)
            gpt4o_pct = gpt4o_correct / len(gpt4o_results) * 100
        else:
            gpt4o_pct = 0
        
        log(f"{complexity.capitalize():<15} {claude_pct:<25.1f}% {gpt4o_pct:<25.1f}%", "INFO")
    
    # Winner
    claude_score = gen_metrics.get("claude", {}).get("functional_correctness", 0)
    gpt4o_score = gen_metrics.get("gpt-4o", {}).get("functional_correctness", 0)
    
    if claude_score > gpt4o_score + 5:
        winner = "Claude Sonnet 3.5"
    elif gpt4o_score > claude_score + 5:
        winner = "GPT-4o"
    else:
        winner = "Tie (within 5%)"
    
    log(f"\n🏆 GENERATION WINNER: {winner}", "SUCCESS")
    
    # ========== KEY FINDINGS ==========
    log("\n" + "="*80, "METRIC")
    log("💡 KEY FINDINGS", "METRIC")
    log("="*80, "METRIC")
    
    findings = []
    
    # Generation finding
    diff = abs(claude_score - gpt4o_score)
    if diff < 5:
        findings.append("Generation quality is comparable between models")
    else:
        better = "Claude" if claude_score > gpt4o_score else "GPT-4o"
        findings.append(f"{better} produces significantly better code ({max(claude_score, gpt4o_score):.1f}% vs {min(claude_score, gpt4o_score):.1f}%)")
    
    # Complexity findings
    for complexity in ['simple', 'medium', 'complex']:
        if complexity not in suites:
            continue
        
        claude_results = [r for r in all_results if r.generator_model == "claude" and r.complexity == complexity]
        gpt4o_results = [r for r in all_results if r.generator_model == "gpt-4o" and r.complexity == complexity]
        
        if claude_results and gpt4o_results:
            claude_correct = sum(1 for r in claude_results if r.consensus_anomalous is False) / len(claude_results) * 100
            gpt4o_correct = sum(1 for r in gpt4o_results if r.consensus_anomalous is False) / len(gpt4o_results) * 100
            
            if abs(claude_correct - gpt4o_correct) > 10:
                better = "Claude" if claude_correct > gpt4o_correct else "GPT-4o"
                findings.append(f"{better} excels at {complexity} circuits ({max(claude_correct, gpt4o_correct):.1f}% vs {min(claude_correct, gpt4o_correct):.1f}%)")
    
    # Testbench findings
    claude_tb = gen_metrics.get("claude", {}).get("testbench_inclusion", 0)
    gpt4o_tb = gen_metrics.get("gpt-4o", {}).get("testbench_inclusion", 0)
    if min(claude_tb, gpt4o_tb) >= 95:
        findings.append("Both models consistently include testbenches")
    
    # Code length
    claude_loc = gen_metrics.get("claude", {}).get("avg_lines_of_code", 0)
    gpt4o_loc = gen_metrics.get("gpt-4o", {}).get("avg_lines_of_code", 0)
    if abs(claude_loc - gpt4o_loc) > 10:
        more_verbose = "Claude" if claude_loc > gpt4o_loc else "GPT-4o"
        findings.append(f"{more_verbose} generates more verbose code ({max(claude_loc, gpt4o_loc):.0f} vs {min(claude_loc, gpt4o_loc):.0f} avg lines)")
    
    for i, finding in enumerate(findings, 1):
        log(f"{i}. {finding}", "INFO")
    
    # ========== SAVE RESULTS ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_suffix = "_".join(suites)
    
    # Save JSON
    json_filename = f"complete_llm_comparison_{suite_suffix}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'session_id': session_id,
            'test_suites': suites,
            'generation_metrics': gen_metrics,
            'key_findings': findings,
            'detailed_results': [asdict(r) for r in all_results]
        }, f, indent=2)
    
    log(f"\n💾 Detailed results saved: {json_filename}", "SUCCESS")
    
    # Save CSV
    df = pd.DataFrame([asdict(r) for r in all_results])
    csv_filename = f"complete_llm_comparison_{suite_suffix}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    log(f"📊 CSV saved: {csv_filename}", "SUCCESS")
    
    log("\n" + "="*80, "SUCCESS")
    log("✅ COMPARISON COMPLETE!", "SUCCESS")
    log("="*80, "SUCCESS")

def main():
    parser = argparse.ArgumentParser(
        description='Complete LLM Verilog generation comparison with 2-of-3 consensus voting'
    )
    parser.add_argument(
        '--suites', 
        nargs='+', 
        choices=['simple', 'medium', 'complex'],
        required=True,
        help='Which test suites to run (4 simple + 10 medium + 8 complex)'
    )
    args = parser.parse_args()
    
    suites = args.suites
    
    log("="*80, "TEST")
    log(f"COMPLETE LLM COMPARISON: {', '.join(suites).upper()}", "TEST")
    log("Testing: 4 simple + 10 medium + 8 complex circuits", "TEST")
    log("="*80, "TEST")
    
    # Check backend
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            log("❌ Backend not responding!", "ERROR")
            return
        log("✅ Backend is running", "SUCCESS")
    except:
        log("❌ Backend NOT running! Start with: uvicorn main:app --reload", "ERROR")
        return
    
    # Get circuit prompts
    prompts = get_circuit_prompts(suites)
    log(f"\nTotal circuits to test: {len(prompts)}", "INFO")
    log(f"Total tests: {len(prompts) * 2} (each circuit × 2 models)", "INFO")
    log(f"Estimated time: ~{len(prompts) * 2 * 10 / 60:.1f} minutes", "WARNING")
    
    # Run tests
    session_id = f"complete_comparison_{'_'.join(suites)}_{int(time.time())}"
    all_results = []
    
    for i, (name, prompt, category, complexity) in enumerate(prompts, 1):
        log(f"\n[{i}/{len(prompts)}] Testing: {name}", "TEST")
        
        circuit_results = test_circuit_complete(
            name, prompt, category, complexity, session_id
        )
        
        for model_id, result in circuit_results.items():
            all_results.append(result)
        
        time.sleep(1)
    
    # Calculate metrics
    log("\n📊 Calculating metrics...", "METRIC")
    gen_metrics = calculate_generation_metrics(all_results)
    
    # Generate report
    generate_report(all_results, gen_metrics, suites, session_id)

if __name__ == "__main__":
    main()