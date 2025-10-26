"""
VAE vs LLM Verification Comparison Test
Compares VAE-based verification against LLM-based waveform analysis

Run from project root:
    python tests/analysis/compare_vae_vs_llm.py
"""

import sys
import os
import requests
import time
from datetime import datetime
import json
import pandas as pd

BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"comparison_test_{int(time.time())}"

def log(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "TEST": "🧪"}.get(status, "📝")
    print(f"[{timestamp}] {emoji} {message}")

def test_circuit_both_methods(name, verilog_code, expected_normal, category):
    """Test a circuit with BOTH VAE and LLM, return comparison"""
    log(f"\nTesting: {name}")
    
    try:
        # Run simulation with VAE
        response = requests.post(
            f"{BASE_URL}/api/simulation/run-with-verification",
            json={"verilog_code": verilog_code, "session_id": TEST_SESSION_ID},
            timeout=60
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if not data.get("success"):
            return None
        
        # Extract VAE results
        vae = data.get("verification", {}).get("vae_verification", {})
        vae_available = vae.get("available", False)
        vae_anomalous = vae.get("is_anomalous", False) if vae_available else None
        vae_confidence = vae.get("confidence", 0.0) if vae_available else None
        vae_message = vae.get("message", "") if vae_available else "N/A"
        
        # Extract LLM results  
        llm = data.get("verification", {}).get("llm_verification", {})
        llm_available = llm.get("available", False)
        llm_analysis = llm.get("analysis", "") if llm_available else "N/A"
        
        # Parse LLM analysis for verdict (simple heuristic)
        llm_anomalous = None
        if llm_available and llm_analysis:
            llm_lower = llm_analysis.lower()
            # Look for anomaly indicators
            if any(word in llm_lower for word in ['error', 'incorrect', 'bug', 'wrong', 'issue', 'problem', 'anomal']):
                llm_anomalous = True
            elif any(word in llm_lower for word in ['correct', 'normal', 'expected', 'working', 'proper']):
                llm_anomalous = False
        
        # Determine correctness
        expected_anomalous = not expected_normal
        vae_correct = (vae_anomalous == expected_anomalous) if vae_anomalous is not None else None
        llm_correct = (llm_anomalous == expected_anomalous) if llm_anomalous is not None else None
        
        result = {
            'name': name,
            'category': category,
            'expected_normal': expected_normal,
            
            # VAE results
            'vae_available': vae_available,
            'vae_predicted_normal': not vae_anomalous if vae_anomalous is not None else None,
            'vae_confidence': vae_confidence,
            'vae_correct': vae_correct,
            'vae_message': vae_message,
            
            # LLM results
            'llm_available': llm_available,
            'llm_predicted_normal': not llm_anomalous if llm_anomalous is not None else None,
            'llm_correct': llm_correct,
            'llm_analysis': llm_analysis[:200] + "..." if len(llm_analysis) > 200 else llm_analysis,
        }
        
        # Log comparison
        if vae_available and llm_available:
            vae_status = "✅" if vae_correct else "❌"
            llm_status = "✅" if llm_correct else "❌"
            log(f"  VAE: {vae_status} (conf: {vae_confidence:.3f})")
            log(f"  LLM: {llm_status}")
            
            if vae_correct and llm_correct:
                log(f"  Both Correct! ✅✅", "SUCCESS")
            elif vae_correct and not llm_correct:
                log(f"  VAE Better ✅ vs ❌", "WARNING")
            elif not vae_correct and llm_correct:
                log(f"  LLM Better ❌ vs ✅", "WARNING")
            else:
                log(f"  Both Wrong ❌❌", "ERROR")
        
        return result
        
    except Exception as e:
        log(f"Exception: {e}", "ERROR")
        return None

def get_comparison_test_circuits():
    """Get test circuits for comparison"""
    return {
        "Combinational - Normal": [
            ("2-Input AND", '''`timescale 1ns/1ps
module and_gate(input wire a, b, output wire out);
assign out = a & b;
endmodule
module testbench;
reg a, b; wire out;
and_gate dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', True),
            
            ("2-Input OR", '''`timescale 1ns/1ps
module or_gate(input wire a, b, output wire out);
assign out = a | b;
endmodule
module testbench;
reg a, b; wire out;
or_gate dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', True),
        ],
        
        "Combinational - Buggy": [
            ("Stuck AND (always 0)", '''`timescale 1ns/1ps
module bad_and(input wire a, b, output wire out);
assign out = 1'b0;
endmodule
module testbench;
reg a, b; wire out;
bad_and dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', False),
            
            ("Inverted AND (NAND)", '''`timescale 1ns/1ps
module bad_and3(input wire a, b, output wire out);
assign out = ~(a & b);
endmodule
module testbench;
reg a, b; wire out;
bad_and3 dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', False),
        ],
        
        "Sequential - Normal": [
            ("4-bit Counter", '''`timescale 1ns/1ps
module counter(input wire clk, rst, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else count <= count + 1'b1;
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count;
counter dut(.clk(clk), .rst(rst), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#100; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d", $time, rst, count);
endmodule''', True),
        ],
        
        "Sequential - Buggy": [
            ("Stuck Counter", '''`timescale 1ns/1ps
module bad_counter(input wire clk, rst, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else count <= count;
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count;
bad_counter dut(.clk(clk), .rst(rst), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#100; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d", $time, rst, count);
endmodule''', False),
        ],
    }

def main():
    log("\n" + "="*80)
    log("VAE vs LLM VERIFICATION COMPARISON", "TEST")
    log("="*80)
    log("Testing circuits with BOTH methods for direct comparison")
    
    circuits = get_comparison_test_circuits()
    total = sum(len(tests) for tests in circuits.values())
    log(f"Total test cases: {total}")
    log(f"Estimated time: ~{total * 5 / 60:.1f} minutes")
    log("="*80)
    
    all_results = []
    test_num = 0
    
    for category, tests in circuits.items():
        log(f"\n{'='*80}")
        log(f"CATEGORY: {category}", "TEST")
        log(f"{'='*80}")
        
        for name, code, is_normal in tests:
            test_num += 1
            log(f"\n[{test_num:2d}/{total:2d}] {name}")
            
            result = test_circuit_both_methods(name, code, is_normal, category)
            
            if result:
                all_results.append(result)
            
            time.sleep(5)  # Longer delay for LLM calls
    
    # Analysis
    log("\n" + "="*80)
    log("COMPARISON ANALYSIS", "TEST")
    log("="*80)
    
    if not all_results:
        log("No successful tests!", "ERROR")
        return
    
    # Calculate metrics for both methods
    vae_results = [r for r in all_results if r['vae_available'] and r['vae_correct'] is not None]
    llm_results = [r for r in all_results if r['llm_available'] and r['llm_correct'] is not None]
    
    vae_accuracy = sum(1 for r in vae_results if r['vae_correct']) / len(vae_results) * 100 if vae_results else 0
    llm_accuracy = sum(1 for r in llm_results if r['llm_correct']) / len(llm_results) * 100 if llm_results else 0
    
    log(f"\n📊 OVERALL COMPARISON:")
    log(f"  VAE Accuracy:  {vae_accuracy:.1f}% ({sum(1 for r in vae_results if r['vae_correct'])}/{len(vae_results)})")
    log(f"  LLM Accuracy:  {llm_accuracy:.1f}% ({sum(1 for r in llm_results if r['llm_correct'])}/{len(llm_results)})")
    
    # Agreement analysis
    both_available = [r for r in all_results if r['vae_available'] and r['llm_available'] 
                      and r['vae_correct'] is not None and r['llm_correct'] is not None]
    
    if both_available:
        both_correct = sum(1 for r in both_available if r['vae_correct'] and r['llm_correct'])
        vae_only_correct = sum(1 for r in both_available if r['vae_correct'] and not r['llm_correct'])
        llm_only_correct = sum(1 for r in both_available if not r['vae_correct'] and r['llm_correct'])
        both_wrong = sum(1 for r in both_available if not r['vae_correct'] and not r['llm_correct'])
        
        log(f"\n📈 AGREEMENT ANALYSIS:")
        log(f"  Both Correct:    {both_correct}/{len(both_available)} ({both_correct/len(both_available)*100:.1f}%)")
        log(f"  VAE Only Right:  {vae_only_correct}/{len(both_available)} ({vae_only_correct/len(both_available)*100:.1f}%)")
        log(f"  LLM Only Right:  {llm_only_correct}/{len(both_available)} ({llm_only_correct/len(both_available)*100:.1f}%)")
        log(f"  Both Wrong:      {both_wrong}/{len(both_available)} ({both_wrong/len(both_available)*100:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vae_vs_llm_comparison_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': len(all_results),
                'vae_accuracy': vae_accuracy,
                'llm_accuracy': llm_accuracy,
                'vae_count': len(vae_results),
                'llm_count': len(llm_results),
            },
            'agreement': {
                'both_correct': both_correct if both_available else 0,
                'vae_only': vae_only_correct if both_available else 0,
                'llm_only': llm_only_correct if both_available else 0,
                'both_wrong': both_wrong if both_available else 0,
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    log(f"\n💾 Comparison results saved to: {filename}")
    
    # Create comparison table
    df = pd.DataFrame(all_results)
    df.to_csv(f"vae_vs_llm_table_{timestamp}.csv", index=False)
    log(f"📊 Comparison table saved to: vae_vs_llm_table_{timestamp}.csv")
    
    log("\n" + "="*80)
    if vae_accuracy > llm_accuracy:
        log("🏆 VAE OUTPERFORMS LLM-BASED VERIFICATION!", "SUCCESS")
    elif llm_accuracy > vae_accuracy:
        log("🏆 LLM OUTPERFORMS VAE-BASED VERIFICATION!", "SUCCESS")
    else:
        log("🤝 VAE AND LLM PERFORM SIMILARLY!", "SUCCESS")
    log("="*80)

if __name__ == "__main__":
    main()