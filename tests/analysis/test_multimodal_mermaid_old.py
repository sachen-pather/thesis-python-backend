"""
Multi-Modal Input Test Suite - Mermaid Diagram Impact Analysis
Tests Verilog generation with and without Mermaid diagrams in prompts

SAVE AS: tests/analysis/test_multimodal_mermaid.py
RUN FROM ROOT: python tests/analysis/test_multimodal_mermaid.py
"""

import sys
import os
import time
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Path setup
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = ROOT_DIR / "thesis_generation_results" / "multimodal_mermaid"

sys.path.insert(0, str(ROOT_DIR))

BASE_URL = "http://localhost:8000"

MODELS = {
    "claude": {"name": "Claude Sonnet 3.5", "provider": "Anthropic"},
    "gpt-4o": {"name": "GPT-4o", "provider": "OpenAI"}
}

@dataclass
class MultiModalResult:
    circuit_name: str
    category: str
    complexity: str
    original_prompt: str
    model: str
    
    # Approach 1: Prompt Only (baseline - direct prompt → verilog)
    prompt_only_verilog: str
    prompt_only_generation_success: bool
    prompt_only_lines_of_code: int
    prompt_only_has_testbench: bool
    prompt_only_compilation_success: bool
    prompt_only_simulation_success: bool
    prompt_only_vae_verdict: Optional[bool]
    prompt_only_vae_confidence: float
    prompt_only_consensus_anomalous: Optional[bool]
    prompt_only_generation_time: float
    prompt_only_simulation_time: float
    
    # Approach 2: Mermaid Only (prompt → mermaid → verilog, NO prompt in step 2)
    mermaid_only_mermaid: str
    mermaid_only_verilog: str
    mermaid_only_generation_success: bool
    mermaid_only_lines_of_code: int
    mermaid_only_has_testbench: bool
    mermaid_only_compilation_success: bool
    mermaid_only_simulation_success: bool
    mermaid_only_vae_verdict: Optional[bool]
    mermaid_only_vae_confidence: float
    mermaid_only_consensus_anomalous: Optional[bool]
    mermaid_only_generation_time: float
    mermaid_only_simulation_time: float
    
    # Approach 3: Combined (prompt → mermaid → [prompt + mermaid] → verilog)
    combined_prompt: str
    combined_verilog: str
    combined_generation_success: bool
    combined_lines_of_code: int
    combined_has_testbench: bool
    combined_compilation_success: bool
    combined_simulation_success: bool
    combined_vae_verdict: Optional[bool]
    combined_vae_confidence: float
    combined_consensus_anomalous: Optional[bool]
    combined_generation_time: float
    combined_simulation_time: float
    
    # Comparison metrics
    best_approach: str  # "prompt_only", "mermaid_only", "combined", or "tie"
    correctness_ranking: str  # e.g., "combined > mermaid_only > prompt_only"
    simulation_ranking: str

def log(message: str, status="INFO"):
    """Enhanced logging with emojis"""
    emoji_map = {
        "INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", 
        "WARNING": "⚠️", "TEST": "🧪", "GENERATE": "🤖",
        "SIMULATE": "⚡", "VERIFY": "🔍", "METRIC": "📊",
        "COMPARE": "⚖️", "MULTIMODAL": "🎨"
    }
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = emoji_map.get(status, "📝")
    print(f"[{timestamp}] {emoji} {message}")

def get_test_circuits() -> List[Tuple[str, str, str, str]]:
    """Get curated test circuits across different categories"""
    return [
        # Simple circuits (5)
        ("2-to-1 MUX", "Design a 2-to-1 multiplexer with inputs a, b, select sel, and output out. Include testbench testing all combinations.", "Combinational", "simple"),
        ("D Flip-Flop", "Create a D flip-flop with clock, reset, input d, and output q. Include testbench with timing verification.", "Sequential", "simple"),
        ("4-bit Counter", "Design a 4-bit synchronous counter with clock, reset, enable, and count[3:0]. Include testbench.", "Sequential", "simple"),
        ("Full Adder", "Create a full adder with inputs a, b, cin and outputs sum, cout. Include testbench with all 8 cases.", "Combinational", "simple"),
        ("2-bit Comparator", "Design a 2-bit comparator with inputs a[1:0], b[1:0] and outputs eq, gt, lt. Include testbench.", "Combinational", "simple"),
        
        # Medium circuits (5)
        ("4-bit ALU", "Design a 4-bit ALU supporting ADD, SUB, AND, OR operations. Inputs: a[3:0], b[3:0], op[1:0]. Outputs: result[3:0], zero. Include testbench.", "Arithmetic", "medium"),
        ("8:1 Multiplexer", "Create an 8:1 multiplexer with input data[7:0], select sel[2:0], and output out. Include testbench.", "Combinational", "medium"),
        ("Sequence Detector", "Design an FSM detecting pattern 1011 in serial input with overlapping detection. Include testbench with sequence: 10110111011.", "State Machine", "medium"),
        ("4-bit Shift Register", "Create a 4-bit PISO shift register with clock, reset, load, parallel input[3:0], and serial output. Include testbench.", "Sequential", "medium"),
        ("Priority Encoder", "Design a 4-to-2 priority encoder with input in[3:0], output out[1:0], and valid bit. Highest bit has priority. Include testbench.", "Combinational", "medium"),
        
        # Complex circuits (5)
        ("Traffic Light Controller", "Design a traffic light FSM with 4 states: NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW. Inputs: clk, rst, emergency. Outputs: ns_light[1:0], ew_light[1:0]. Timing: GREEN=8 cycles, YELLOW=2 cycles. Emergency makes both RED. Include testbench.", "State Machine", "complex"),
        ("UART Transmitter", "Design UART transmitter with states: IDLE, START, DATA, STOP. Protocol: START(0), 8 data bits LSB-first, STOP(1). Baud rate divider. Include testbench.", "State Machine", "complex"),
        ("8-bit Register File", "Create 8-register × 8-bit register file with dual read ports and single write port. Include testbench with simultaneous operations.", "CPU Component", "complex"),
        ("PWM Generator", "Design 8-bit resolution PWM generator with configurable duty cycle. Include testbench with duty cycles: 0%, 25%, 50%, 75%, 100%.", "Advanced Sequential", "complex"),
        ("SPI Master", "Design SPI master supporting mode 0 (CPOL=0, CPHA=0). Include SCLK generation, MOSI/MISO handling. Include testbench.", "Protocol", "complex"),
    ]

def generate_prompt_only(prompt: str, model: str, session_id: str) -> Tuple[bool, str, Dict]:
    """Approach 1: Direct prompt → verilog (no mermaid step)"""
    try:
        start = time.time()
        
        # Generate Verilog directly from prompt without mermaid
        # We'll use the verilog endpoint but with an empty/minimal mermaid
        verilog_resp = requests.post(f"{BASE_URL}/api/design/generate-verilog",
            json={
                "mermaid_code": "",  # Empty mermaid
                "description": prompt,
                "model": model, 
                "use_rag": False, 
                "session_id": session_id
            }, timeout=90)
        
        if verilog_resp.status_code != 200:
            return False, "", {"error": "Verilog generation failed", "generation_time": time.time() - start}
        
        verilog_data = verilog_resp.json()
        if not verilog_data.get("success"):
            return False, "", {"error": verilog_data.get("error"), "generation_time": time.time() - start}
        
        stats = verilog_data.get("stats", {})
        stats["generation_time"] = time.time() - start
        return True, verilog_data.get("verilog_code", ""), stats
        
    except Exception as e:
        return False, "", {"error": str(e), "generation_time": time.time() - start}

def generate_standard_flow(prompt: str, model: str, session_id: str) -> Tuple[bool, str, str, Dict]:
    """Approach 2: Mermaid Only - prompt → mermaid → verilog (NO prompt in step 2)"""
    try:
        start = time.time()
        
        # Step 1: Generate Mermaid
        mermaid_resp = requests.post(f"{BASE_URL}/api/design/generate-mermaid",
            json={"prompt": prompt, "model": model, "session_id": session_id}, timeout=90)
        
        if mermaid_resp.status_code != 200:
            return False, "", "", {"error": "Mermaid generation failed", "generation_time": time.time() - start}
        
        mermaid_data = mermaid_resp.json()
        if not mermaid_data.get("success"):
            return False, "", "", {"error": mermaid_data.get("error"), "generation_time": time.time() - start}
        
        mermaid_code = mermaid_data.get("mermaid_code", "")
        
        # Step 2: Generate Verilog from Mermaid ONLY (no original prompt)
        verilog_resp = requests.post(f"{BASE_URL}/api/design/generate-verilog",
            json={
                "mermaid_code": mermaid_code, 
                "description": "",  # Empty description - mermaid only!
                "model": model, 
                "use_rag": False, 
                "session_id": session_id
            }, timeout=90)
        
        if verilog_resp.status_code != 200:
            return False, mermaid_code, "", {"error": "Verilog generation failed", "generation_time": time.time() - start}
        
        verilog_data = verilog_resp.json()
        if not verilog_data.get("success"):
            return False, mermaid_code, "", {"error": verilog_data.get("error"), "generation_time": time.time() - start}
        
        stats = verilog_data.get("stats", {})
        stats["generation_time"] = time.time() - start
        return True, mermaid_code, verilog_data.get("verilog_code", ""), stats
        
    except Exception as e:
        return False, "", "", {"error": str(e), "generation_time": time.time() - start}

def generate_multimodal_flow(base_prompt: str, mermaid: str, model: str, session_id: str) -> Tuple[bool, str, Dict]:
    """Approach 3: Combined - prompt → mermaid → [prompt + mermaid] → verilog"""
    try:
        start = time.time()
        
        # Create enhanced prompt with BOTH original prompt AND mermaid diagram
        enhanced_prompt = f"""{base_prompt}

Here is the Mermaid diagram specification for this circuit:

```mermaid
{mermaid}
```

Generate Verilog code that implements this exact architecture shown in the Mermaid diagram."""
        
        # Generate Verilog from both prompt and mermaid
        verilog_resp = requests.post(f"{BASE_URL}/api/design/generate-verilog",
            json={
                "mermaid_code": mermaid, 
                "description": enhanced_prompt,
                "model": model, 
                "use_rag": False, 
                "session_id": session_id
            }, timeout=90)
        
        if verilog_resp.status_code != 200:
            return False, "", {"error": "Verilog generation failed", "generation_time": time.time() - start}
        
        verilog_data = verilog_resp.json()
        if not verilog_data.get("success"):
            return False, "", {"error": verilog_data.get("error"), "generation_time": time.time() - start}
        
        stats = verilog_data.get("stats", {})
        stats["generation_time"] = time.time() - start
        stats["enhanced_prompt"] = enhanced_prompt
        return True, verilog_data.get("verilog_code", ""), stats
        
    except Exception as e:
        return False, "", {"error": str(e), "generation_time": time.time() - start}

def simulate_and_verify(verilog: str, model: str, session_id: str) -> Dict:
    """Run simulation and verification"""
    try:
        response = requests.post(f"{BASE_URL}/api/simulation/run-with-verification",
            json={"verilog_code": verilog, "model": model, "session_id": session_id}, timeout=180)
        return response.json() if response.status_code == 200 else {"success": False}
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculate_consensus(vae: Optional[bool], llm_verdict: Optional[bool]) -> Tuple[Optional[bool], float]:
    """Simple consensus from VAE and LLM"""
    if vae is None and llm_verdict is None:
        return None, 0.0
    
    if vae is not None and llm_verdict is not None:
        # Both agree
        if vae == (not llm_verdict):  # llm_verdict True means normal
            return vae, 1.0
        else:
            # Disagree - trust VAE more
            return vae, 0.6
    
    # Only one available
    if vae is not None:
        return vae, 0.7
    else:
        return not llm_verdict if llm_verdict is not None else None, 0.7

def parse_llm_verdict(text: str) -> Optional[bool]:
    """Parse LLM verdict - returns True if normal, False if anomalous, None if unclear"""
    if not text:
        return None
    
    text_lower = text.lower()
    
    if 'verdict: normal' in text_lower:
        return True
    elif 'verdict: anomalous' in text_lower:
        return False
    
    # Fallback pattern matching
    if any(p in text_lower for p in ['circuit is normal', 'functioning correctly', 'behaves as expected']):
        return True
    elif any(p in text_lower for p in ['circuit is anomalous', 'has bugs', 'incorrect behavior']):
        return False
    
    return None

def test_circuit_multimodal(name: str, prompt: str, category: str, complexity: str, 
                           model: str, session_id: str) -> MultiModalResult:
    """Test one circuit with ALL THREE approaches"""
    log(f"\nTESTING: {name} with {MODELS[model]['name']}", "TEST")
    
    # ========== APPROACH 1: PROMPT ONLY ==========
    log(f"  📝 Approach 1: Prompt Only (direct prompt → verilog)...", "GENERATE")
    po_success, po_verilog, po_gen_stats = generate_prompt_only(
        prompt, model, f"{session_id}_prompt_only")
    
    po_result = {
        'verilog': po_verilog,
        'generation_success': po_success,
        'lines_of_code': po_gen_stats.get('lines', 0),
        'has_testbench': po_gen_stats.get('has_testbench', False),
        'compilation_success': False,
        'simulation_success': False,
        'vae_verdict': None,
        'vae_confidence': 0.0,
        'consensus_anomalous': None,
        'generation_time': po_gen_stats.get('generation_time', 0),
        'simulation_time': 0.0
    }
    
    if po_success:
        log(f"    ✅ Generated {po_gen_stats.get('lines', 0)} lines", "SUCCESS")
        
        sim_start = time.time()
        po_sim = simulate_and_verify(po_verilog, model, f"{session_id}_po_sim")
        po_result['simulation_time'] = time.time() - sim_start
        
        if po_sim.get("success"):
            po_result['compilation_success'] = True
            po_result['simulation_success'] = True
            
            vae_verif = po_sim.get("verification", {}).get("vae_verification", {})
            if vae_verif.get("available"):
                po_result['vae_verdict'] = vae_verif.get("is_anomalous", False)
                po_result['vae_confidence'] = vae_verif.get("confidence", 0.0)
            
            llm_verif = po_sim.get("verification", {}).get("llm_verification", {})
            llm_verdict = parse_llm_verdict(llm_verif.get("analysis", ""))
            
            consensus, conf = calculate_consensus(po_result['vae_verdict'], llm_verdict)
            po_result['consensus_anomalous'] = consensus
            
            log(f"    ✅ Sim success | VAE: {po_result['vae_verdict']} | Consensus: {consensus}", "SUCCESS")
        else:
            log(f"    ❌ Simulation failed", "ERROR")
    else:
        log(f"    ❌ Generation failed", "ERROR")
    
    time.sleep(1)
    
    # ========== APPROACH 2: MERMAID ONLY ==========
    log(f"  🎨 Approach 2: Mermaid Only (prompt → mermaid → verilog)...", "GENERATE")
    mo_success, mo_mermaid, mo_verilog, mo_gen_stats = generate_standard_flow(
        prompt, model, f"{session_id}_mermaid_only")
    
    mo_result = {
        'mermaid': mo_mermaid,
        'verilog': mo_verilog,
        'generation_success': mo_success,
        'lines_of_code': mo_gen_stats.get('lines', 0),
        'has_testbench': mo_gen_stats.get('has_testbench', False),
        'compilation_success': False,
        'simulation_success': False,
        'vae_verdict': None,
        'vae_confidence': 0.0,
        'consensus_anomalous': None,
        'generation_time': mo_gen_stats.get('generation_time', 0),
        'simulation_time': 0.0
    }
    
    if mo_success:
        log(f"    ✅ Generated {mo_gen_stats.get('lines', 0)} lines", "SUCCESS")
        
        sim_start = time.time()
        mo_sim = simulate_and_verify(mo_verilog, model, f"{session_id}_mo_sim")
        mo_result['simulation_time'] = time.time() - sim_start
        
        if mo_sim.get("success"):
            mo_result['compilation_success'] = True
            mo_result['simulation_success'] = True
            
            vae_verif = mo_sim.get("verification", {}).get("vae_verification", {})
            if vae_verif.get("available"):
                mo_result['vae_verdict'] = vae_verif.get("is_anomalous", False)
                mo_result['vae_confidence'] = vae_verif.get("confidence", 0.0)
            
            llm_verif = mo_sim.get("verification", {}).get("llm_verification", {})
            llm_verdict = parse_llm_verdict(llm_verif.get("analysis", ""))
            
            consensus, conf = calculate_consensus(mo_result['vae_verdict'], llm_verdict)
            mo_result['consensus_anomalous'] = consensus
            
            log(f"    ✅ Sim success | VAE: {mo_result['vae_verdict']} | Consensus: {consensus}", "SUCCESS")
        else:
            log(f"    ❌ Simulation failed", "ERROR")
    else:
        log(f"    ❌ Generation failed", "ERROR")
    
    time.sleep(1)
    
    # ========== APPROACH 3: COMBINED ==========
    log(f"  🔗 Approach 3: Combined (prompt + mermaid → verilog)...", "MULTIMODAL")
    
    comb_result = {
        'verilog': "",
        'generation_success': False,
        'lines_of_code': 0,
        'has_testbench': False,
        'compilation_success': False,
        'simulation_success': False,
        'vae_verdict': None,
        'vae_confidence': 0.0,
        'consensus_anomalous': None,
        'generation_time': 0.0,
        'simulation_time': 0.0,
        'enhanced_prompt': ""
    }
    
    if mo_success and mo_mermaid:  # Only run if we have a mermaid diagram
        comb_success, comb_verilog, comb_gen_stats = generate_multimodal_flow(
            prompt, mo_mermaid, model, f"{session_id}_combined")
        
        comb_result.update({
            'verilog': comb_verilog,
            'generation_success': comb_success,
            'lines_of_code': comb_gen_stats.get('lines', 0),
            'has_testbench': comb_gen_stats.get('has_testbench', False),
            'generation_time': comb_gen_stats.get('generation_time', 0),
            'enhanced_prompt': comb_gen_stats.get('enhanced_prompt', '')
        })
        
        if comb_success:
            log(f"    ✅ Generated {comb_gen_stats.get('lines', 0)} lines", "SUCCESS")
            
            sim_start = time.time()
            comb_sim = simulate_and_verify(comb_verilog, model, f"{session_id}_comb_sim")
            comb_result['simulation_time'] = time.time() - sim_start
            
            if comb_sim.get("success"):
                comb_result['compilation_success'] = True
                comb_result['simulation_success'] = True
                
                vae_verif = comb_sim.get("verification", {}).get("vae_verification", {})
                if vae_verif.get("available"):
                    comb_result['vae_verdict'] = vae_verif.get("is_anomalous", False)
                    comb_result['vae_confidence'] = vae_verif.get("confidence", 0.0)
                
                llm_verif = comb_sim.get("verification", {}).get("llm_verification", {})
                llm_verdict = parse_llm_verdict(llm_verif.get("analysis", ""))
                
                consensus, conf = calculate_consensus(comb_result['vae_verdict'], llm_verdict)
                comb_result['consensus_anomalous'] = consensus
                
                log(f"    ✅ Sim success | VAE: {comb_result['vae_verdict']} | Consensus: {consensus}", "SUCCESS")
            else:
                log(f"    ❌ Simulation failed", "ERROR")
        else:
            log(f"    ❌ Generation failed", "ERROR")
    else:
        log(f"    ⚠️  Skipped (no mermaid from approach 2)", "WARNING")
    
    # ========== COMPARISON ==========
    approaches = {
        'prompt_only': po_result['consensus_anomalous'],
        'mermaid_only': mo_result['consensus_anomalous'],
        'combined': comb_result['consensus_anomalous']
    }
    
    # Determine best approach
    correct_approaches = []
    for name, verdict in approaches.items():
        if verdict is False:  # False means normal/correct
            correct_approaches.append(name)
    
    if len(correct_approaches) == 1:
        best = correct_approaches[0]
    elif len(correct_approaches) > 1:
        best = "tie"
    else:
        # None correct - check which failed least badly
        sim_success = {
            'prompt_only': po_result['simulation_success'],
            'mermaid_only': mo_result['simulation_success'],
            'combined': comb_result['simulation_success']
        }
        best = max(sim_success, key=sim_success.get) if any(sim_success.values()) else "none"
    
    # Create rankings
    def rank_approaches(metric_dict):
        sorted_items = sorted(metric_dict.items(), key=lambda x: (x[1] is None, not x[1] if x[1] is not None else True))
        return " > ".join([k for k, v in sorted_items if v is not None])
    
    correctness_rank = rank_approaches({
        'prompt_only': po_result['consensus_anomalous'] is False,
        'mermaid_only': mo_result['consensus_anomalous'] is False,
        'combined': comb_result['consensus_anomalous'] is False
    })
    
    simulation_rank = rank_approaches({
        'prompt_only': po_result['simulation_success'],
        'mermaid_only': mo_result['simulation_success'],
        'combined': comb_result['simulation_success']
    })
    
    log(f"  ⚖️  Best Approach: {best.upper()}", "COMPARE")
    
    return MultiModalResult(
        circuit_name=name,
        category=category,
        complexity=complexity,
        original_prompt=prompt,
        model=model,
        
        # Prompt Only
        prompt_only_verilog=po_result['verilog'],
        prompt_only_generation_success=po_result['generation_success'],
        prompt_only_lines_of_code=po_result['lines_of_code'],
        prompt_only_has_testbench=po_result['has_testbench'],
        prompt_only_compilation_success=po_result['compilation_success'],
        prompt_only_simulation_success=po_result['simulation_success'],
        prompt_only_vae_verdict=po_result['vae_verdict'],
        prompt_only_vae_confidence=po_result['vae_confidence'],
        prompt_only_consensus_anomalous=po_result['consensus_anomalous'],
        prompt_only_generation_time=po_result['generation_time'],
        prompt_only_simulation_time=po_result['simulation_time'],
        
        # Mermaid Only
        mermaid_only_mermaid=mo_result['mermaid'],
        mermaid_only_verilog=mo_result['verilog'],
        mermaid_only_generation_success=mo_result['generation_success'],
        mermaid_only_lines_of_code=mo_result['lines_of_code'],
        mermaid_only_has_testbench=mo_result['has_testbench'],
        mermaid_only_compilation_success=mo_result['compilation_success'],
        mermaid_only_simulation_success=mo_result['simulation_success'],
        mermaid_only_vae_verdict=mo_result['vae_verdict'],
        mermaid_only_vae_confidence=mo_result['vae_confidence'],
        mermaid_only_consensus_anomalous=mo_result['consensus_anomalous'],
        mermaid_only_generation_time=mo_result['generation_time'],
        mermaid_only_simulation_time=mo_result['simulation_time'],
        
        # Combined
        combined_prompt=comb_result['enhanced_prompt'],
        combined_verilog=comb_result['verilog'],
        combined_generation_success=comb_result['generation_success'],
        combined_lines_of_code=comb_result['lines_of_code'],
        combined_has_testbench=comb_result['has_testbench'],
        combined_compilation_success=comb_result['compilation_success'],
        combined_simulation_success=comb_result['simulation_success'],
        combined_vae_verdict=comb_result['vae_verdict'],
        combined_vae_confidence=comb_result['vae_confidence'],
        combined_consensus_anomalous=comb_result['consensus_anomalous'],
        combined_generation_time=comb_result['generation_time'],
        combined_simulation_time=comb_result['simulation_time'],
        
        # Comparison
        best_approach=best,
        correctness_ranking=correctness_rank,
        simulation_ranking=simulation_rank
    )

def generate_analysis_report(all_results: List[MultiModalResult], session_id: str):
    """Generate comprehensive three-way analysis report"""
    log("\n" + "="*80, "METRIC")
    log("THREE-WAY MULTI-MODAL INPUT ANALYSIS", "METRIC")
    log("="*80, "METRIC")
    
    total_tests = len(all_results)
    log(f"\nTotal Tests: {total_tests}", "INFO")
    
    # Overall metrics by model
    for model_id in ["claude", "gpt-4o"]:
        model_results = [r for r in all_results if r.model == model_id]
        if not model_results:
            continue
        
        log(f"\n{'='*80}", "INFO")
        log(f"{MODELS[model_id]['name']} Results ({len(model_results)} circuits)", "INFO")
        log(f"{'='*80}", "INFO")
        
        # Success rates for all three approaches
        po_gen = sum(1 for r in model_results if r.prompt_only_generation_success)
        mo_gen = sum(1 for r in model_results if r.mermaid_only_generation_success)
        comb_gen = sum(1 for r in model_results if r.combined_generation_success)
        
        po_sim = sum(1 for r in model_results if r.prompt_only_simulation_success)
        mo_sim = sum(1 for r in model_results if r.mermaid_only_simulation_success)
        comb_sim = sum(1 for r in model_results if r.combined_simulation_success)
        
        log(f"\n📊 Generation Success:", "INFO")
        log(f"  Prompt Only:    {po_gen}/{len(model_results)} ({po_gen/len(model_results)*100:.1f}%)", "INFO")
        log(f"  Mermaid Only:   {mo_gen}/{len(model_results)} ({mo_gen/len(model_results)*100:.1f}%)", "INFO")
        log(f"  Combined:       {comb_gen}/{len(model_results)} ({comb_gen/len(model_results)*100:.1f}%)", "INFO")
        
        log(f"\n⚡ Simulation Success:", "INFO")
        log(f"  Prompt Only:    {po_sim}/{len(model_results)} ({po_sim/len(model_results)*100:.1f}%)", "INFO")
        log(f"  Mermaid Only:   {mo_sim}/{len(model_results)} ({mo_sim/len(model_results)*100:.1f}%)", "INFO")
        log(f"  Combined:       {comb_sim}/{len(model_results)} ({comb_sim/len(model_results)*100:.1f}%)", "INFO")
        
        # Functional correctness
        po_correct = sum(1 for r in model_results if r.prompt_only_consensus_anomalous is False)
        mo_correct = sum(1 for r in model_results if r.mermaid_only_consensus_anomalous is False)
        comb_correct = sum(1 for r in model_results if r.combined_consensus_anomalous is False)
        
        log(f"\n✅ Functional Correctness:", "INFO")
        log(f"  Prompt Only:    {po_correct}/{len(model_results)} ({po_correct/len(model_results)*100:.1f}%)", "INFO")
        log(f"  Mermaid Only:   {mo_correct}/{len(model_results)} ({mo_correct/len(model_results)*100:.1f}%)", "INFO")
        log(f"  Combined:       {comb_correct}/{len(model_results)} ({comb_correct/len(model_results)*100:.1f}%)", "INFO")
        
        # Winner distribution
        winners = defaultdict(int)
        for r in model_results:
            winners[r.best_approach] += 1
        
        log(f"\n🏆 Best Approach Winners:", "INFO")
        log(f"  Prompt Only:    {winners['prompt_only']} circuits", "INFO")
        log(f"  Mermaid Only:   {winners['mermaid_only']} circuits", "INFO")
        log(f"  Combined:       {winners['combined']} circuits", "INFO")
        log(f"  Tie:            {winners['tie']} circuits", "INFO")
        log(f"  None:           {winners['none']} circuits", "INFO")
        
        # Code metrics
        avg_po_loc = sum(r.prompt_only_lines_of_code for r in model_results if r.prompt_only_generation_success) / max(po_gen, 1)
        avg_mo_loc = sum(r.mermaid_only_lines_of_code for r in model_results if r.mermaid_only_generation_success) / max(mo_gen, 1)
        avg_comb_loc = sum(r.combined_lines_of_code for r in model_results if r.combined_generation_success) / max(comb_gen, 1)
        
        log(f"\n📏 Average Lines of Code:", "INFO")
        log(f"  Prompt Only:    {avg_po_loc:.1f} lines", "INFO")
        log(f"  Mermaid Only:   {avg_mo_loc:.1f} lines", "INFO")
        log(f"  Combined:       {avg_comb_loc:.1f} lines", "INFO")
        
        # Timing metrics
        avg_po_time = sum(r.prompt_only_generation_time for r in model_results if r.prompt_only_generation_success) / max(po_gen, 1)
        avg_mo_time = sum(r.mermaid_only_generation_time for r in model_results if r.mermaid_only_generation_success) / max(mo_gen, 1)
        avg_comb_time = sum(r.combined_generation_time for r in model_results if r.combined_generation_success) / max(comb_gen, 1)
        
        log(f"\n⏱️  Average Generation Time:", "INFO")
        log(f"  Prompt Only:    {avg_po_time:.2f}s", "INFO")
        log(f"  Mermaid Only:   {avg_mo_time:.2f}s", "INFO")
        log(f"  Combined:       {avg_comb_time:.2f}s", "INFO")
    
    # By complexity
    log(f"\n{'='*80}", "INFO")
    log(f"Results by Complexity", "INFO")
    log(f"{'='*80}", "INFO")
    
    for complexity in ["simple", "medium", "complex"]:
        comp_results = [r for r in all_results if r.complexity == complexity]
        if not comp_results:
            continue
        
        po_correct = sum(1 for r in comp_results if r.prompt_only_consensus_anomalous is False)
        mo_correct = sum(1 for r in comp_results if r.mermaid_only_consensus_anomalous is False)
        comb_correct = sum(1 for r in comp_results if r.combined_consensus_anomalous is False)
        
        log(f"\n{complexity.capitalize()} Circuits ({len(comp_results)} tests):", "INFO")
        log(f"  Prompt Only Correctness:    {po_correct}/{len(comp_results)} ({po_correct/len(comp_results)*100:.1f}%)", "INFO")
        log(f"  Mermaid Only Correctness:   {mo_correct}/{len(comp_results)} ({mo_correct/len(comp_results)*100:.1f}%)", "INFO")
        log(f"  Combined Correctness:       {comb_correct}/{len(comp_results)} ({comb_correct/len(comp_results)*100:.1f}%)", "INFO")
        
        winners = defaultdict(int)
        for r in comp_results:
            winners[r.best_approach] += 1
        
        log(f"  Winners: Prompt={winners['prompt_only']}, Mermaid={winners['mermaid_only']}, Combined={winners['combined']}, Tie={winners['tie']}", "INFO")
    
    # Key findings
    log(f"\n{'='*80}", "METRIC")
    log(f"KEY FINDINGS", "METRIC")
    log(f"{'='*80}", "METRIC")
    
    total_po_wins = sum(1 for r in all_results if r.best_approach == 'prompt_only')
    total_mo_wins = sum(1 for r in all_results if r.best_approach == 'mermaid_only')
    total_comb_wins = sum(1 for r in all_results if r.best_approach == 'combined')
    
    log(f"\n🏆 Approach Performance:", "INFO")
    log(f"  Prompt Only Wins:    {total_po_wins} circuits", "INFO")
    log(f"  Mermaid Only Wins:   {total_mo_wins} circuits", "INFO")
    log(f"  Combined Wins:       {total_comb_wins} circuits", "INFO")
    
    # Determine overall winner
    winner_map = {
        total_po_wins: "Prompt Only",
        total_mo_wins: "Mermaid Only", 
        total_comb_wins: "Combined"
    }
    max_wins = max(total_po_wins, total_mo_wins, total_comb_wins)
    overall_winner = winner_map[max_wins]
    
    log(f"\n🎯 OVERALL WINNER: {overall_winner.upper()}", "SUCCESS")
    
    # Statistical significance
    total_correct_po = sum(1 for r in all_results if r.prompt_only_consensus_anomalous is False)
    total_correct_mo = sum(1 for r in all_results if r.mermaid_only_consensus_anomalous is False)
    total_correct_comb = sum(1 for r in all_results if r.combined_consensus_anomalous is False)
    
    log(f"\n📊 Overall Correctness Rates:", "METRIC")
    log(f"  Prompt Only:    {total_correct_po}/{total_tests} ({total_correct_po/total_tests*100:.1f}%)", "METRIC")
    log(f"  Mermaid Only:   {total_correct_mo}/{total_tests} ({total_correct_mo/total_tests*100:.1f}%)", "METRIC")
    log(f"  Combined:       {total_correct_comb}/{total_tests} ({total_correct_comb/total_tests*100:.1f}%)", "METRIC")
    
    # Thesis recommendations
    log(f"\n{'='*80}", "METRIC")
    log(f"THESIS RECOMMENDATIONS", "METRIC")
    log(f"{'='*80}", "METRIC")
    
    if total_comb_wins > total_mo_wins and total_comb_wins > total_po_wins:
        log(f"\n✅ Multi-modal input (prompt + diagram) provides BEST results", "SUCCESS")
        log(f"   Architectural context enhances LLM code generation", "SUCCESS")
    elif total_mo_wins > total_po_wins and total_mo_wins > total_comb_wins:
        log(f"\n🎨 Diagram-only approach performs BEST", "SUCCESS")
        log(f"   Mermaid provides sufficient architectural specification", "SUCCESS")
    elif total_po_wins > total_mo_wins and total_po_wins > total_comb_wins:
        log(f"\n📝 Direct prompt approach performs BEST", "WARNING")
        log(f"   Additional architectural context may not improve generation", "WARNING")
    else:
        log(f"\n⚖️  No clear winner - context dependency varies by circuit complexity", "INFO")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = OUTPUT_DIR / f"three_way_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'session_id': session_id,
            'test_type': 'three_way_multimodal_comparison',
            'total_circuits': total_tests,
            'timestamp': timestamp,
            'summary': {
                'prompt_only_wins': total_po_wins,
                'mermaid_only_wins': total_mo_wins,
                'combined_wins': total_comb_wins,
                'overall_winner': overall_winner,
                'correctness_rates': {
                    'prompt_only': f"{total_correct_po/total_tests*100:.1f}%",
                    'mermaid_only': f"{total_correct_mo/total_tests*100:.1f}%",
                    'combined': f"{total_correct_comb/total_tests*100:.1f}%"
                }
            },
            'results': [asdict(r) for r in all_results]
        }, f, indent=2)
    
    # CSV - Full Data
    df = pd.DataFrame([asdict(r) for r in all_results])
    csv_path = OUTPUT_DIR / f"three_way_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Comparison summary CSV - Clean version
    def verdict_str(val):
        if val is False:
            return 'Yes'  # False anomalous = correct
        elif val is True:
            return 'No'   # True anomalous = incorrect
        else:
            return 'N/A'
    
    comparison_data = []
    for r in all_results:
        comparison_data.append({
            'Circuit': r.circuit_name,
            'Model': r.model,
            'Complexity': r.complexity,
            'PromptOnly_Correct': verdict_str(r.prompt_only_consensus_anomalous),
            'MermaidOnly_Correct': verdict_str(r.mermaid_only_consensus_anomalous),
            'Combined_Correct': verdict_str(r.combined_consensus_anomalous),
            'PromptOnly_LOC': r.prompt_only_lines_of_code,
            'MermaidOnly_LOC': r.mermaid_only_lines_of_code,
            'Combined_LOC': r.combined_lines_of_code,
            'PromptOnly_Time': f"{r.prompt_only_generation_time:.2f}s",
            'MermaidOnly_Time': f"{r.mermaid_only_generation_time:.2f}s",
            'Combined_Time': f"{r.combined_generation_time:.2f}s",
            'Best_Approach': r.best_approach,
            'Correctness_Ranking': r.correctness_ranking,
            'Simulation_Ranking': r.simulation_ranking
        })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_csv_path = OUTPUT_DIR / f"three_way_comparison_{timestamp}.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    
    log(f"\n{'='*80}", "SUCCESS")
    log(f"💾 Results saved to:", "SUCCESS")
    log(f"   📄 Full Results: {json_path.relative_to(ROOT_DIR)}", "SUCCESS")
    log(f"   📊 Data CSV: {csv_path.relative_to(ROOT_DIR)}", "SUCCESS")
    log(f"   📈 Comparison: {comp_csv_path.relative_to(ROOT_DIR)}", "SUCCESS")
    log(f"{'='*80}", "SUCCESS")

def main():
    """Main execution"""
    log("="*80, "TEST")
    log("THREE-WAY MULTI-MODAL INPUT TEST SUITE", "TEST")
    log("Testing: Prompt Only vs Mermaid Only vs Combined", "TEST")
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
    
    # Get test circuits
    circuits = get_test_circuits()
    log(f"\nTest Configuration:", "INFO")
    log(f"  Circuits: {len(circuits)}", "INFO")
    log(f"  Models: {len(MODELS)}", "INFO")
    log(f"  Approaches: 3 (Prompt Only, Mermaid Only, Combined)", "INFO")
    log(f"  Total Tests: {len(circuits) * len(MODELS)} circuits × 3 approaches = {len(circuits) * len(MODELS) * 3} generations", "INFO")
    log(f"  Estimated Time: ~{len(circuits) * len(MODELS) * 12 / 60:.1f} minutes", "WARNING")
    
    log(f"\nApproach Descriptions:", "INFO")
    log(f"  1️⃣  Prompt Only:    Direct prompt → Verilog (baseline)", "INFO")
    log(f"  2️⃣  Mermaid Only:   Prompt → Mermaid → Verilog (no prompt in step 2)", "INFO")
    log(f"  3️⃣  Combined:       Prompt → Mermaid → (Prompt + Mermaid) → Verilog", "INFO")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    session_id = f"three_way_{int(time.time())}"
    all_results = []
    
    for i, (name, prompt, category, complexity) in enumerate(circuits, 1):
        log(f"\n{'='*80}", "TEST")
        log(f"Circuit {i}/{len(circuits)}: {name}", "TEST")
        log(f"{'='*80}", "TEST")
        
        for model_id in MODELS.keys():
            result = test_circuit_multimodal(
                name, prompt, category, complexity, model_id,
                f"{session_id}_{i}_{model_id}"
            )
            all_results.append(result)
            
            time.sleep(2)  # Rate limiting between models
    
    # Generate report
    generate_analysis_report(all_results, session_id)
    
    log("\n" + "="*80, "SUCCESS")
    log("✅ THREE-WAY MULTI-MODAL TEST SUITE COMPLETE!", "SUCCESS")
    log("="*80, "SUCCESS")
    
if __name__ == "__main__":
    main()