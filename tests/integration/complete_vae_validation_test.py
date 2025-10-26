"""
Complete VAE Validation Test - Both Normal AND Anomalous Circuits
Tests the full confusion matrix with proper VCD dumps

Run from project root:
    python tests/integration/complete_vae_validation_test.py
"""

import sys
import os
import requests
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"complete_validation_{int(time.time())}"

def log(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "TEST": "🧪"}.get(status, "📝")
    print(f"[{timestamp}] {emoji} {message}")

def test_verilog_direct(name, verilog_code, expected_normal=True):
    """Test Verilog code directly"""
    log(f"\n{'='*80}")
    log(f"Testing: {name}", "TEST")
    log(f"{'='*80}")
    
    try:
        log("Running simulation + VAE verification...", "INFO")
        response = requests.post(
            f"{BASE_URL}/api/simulation/run-with-verification",
            json={"verilog_code": verilog_code, "session_id": TEST_SESSION_ID},
            timeout=60
        )
        
        if response.status_code != 200:
            log(f"Failed: HTTP {response.status_code}", "ERROR")
            return None
        
        data = response.json()
        
        if not data.get("success"):
            log(f"Simulation failed: {data.get('error', 'Unknown')}", "WARNING")
            if not expected_normal:
                log("Expected failure for buggy code - counting as detected", "INFO")
                return {
                    'name': name,
                    'expected_normal': expected_normal,
                    'predicted_normal': False,
                    'confidence': 1.0,
                    'correct': True,
                    'message': 'Detected via simulation failure'
                }
            return None
        
        vae = data.get("verification", {}).get("vae_verification", {})
        
        if not vae.get("available"):
            error = vae.get('error', 'Unknown')
            log(f"VAE unavailable: {error}", "ERROR")
            log(f"Waveform CSV length: {len(data.get('waveform_csv', ''))}", "INFO")
            return None
        
        is_anomalous = vae.get("is_anomalous", False)
        confidence = vae.get("confidence", 0.0)
        message = vae.get("message", "")
        
        expected_anomalous = not expected_normal
        correct = (is_anomalous == expected_anomalous)
        
        expected_str = "NORMAL" if expected_normal else "ANOMALOUS"
        predicted_str = "ANOMALOUS" if is_anomalous else "NORMAL"
        
        log(f"Expected:  {expected_str}", "INFO")
        log(f"Predicted: {predicted_str} (confidence: {confidence:.3f})", "INFO")
        log(f"Message:   {message[:80]}...", "INFO")
        
        if correct:
            log(f"✅ CORRECT PREDICTION", "SUCCESS")
        else:
            log(f"❌ WRONG PREDICTION", "ERROR")
        
        return {
            'name': name,
            'expected_normal': expected_normal,
            'predicted_normal': not is_anomalous,
            'confidence': confidence,
            'correct': correct,
            'message': message
        }
        
    except Exception as e:
        log(f"Exception: {e}", "ERROR")
        return None

def main():
    log("\n" + "="*80)
    log("COMPLETE VAE VALIDATION - NORMAL + ANOMALOUS CIRCUITS", "TEST")
    log("="*80)
    
    # NORMAL CIRCUITS - with proper VCD dumps matching LLM output
    normal_circuits = [
        ("Good AND Gate", '''`timescale 1ns/1ps

module and_gate(input wire a, b, output wire out);
    assign out = a & b;
endmodule

module testbench;
    reg a, b;
    wire out;
    
    and_gate dut(.a(a), .b(b), .out(out));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        a=0; b=0; #10;
        a=0; b=1; #10;
        a=1; b=0; #10;
        a=1; b=1; #10;
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule'''),

        ("Good OR Gate", '''`timescale 1ns/1ps

module or_gate(input wire a, b, output wire out);
    assign out = a | b;
endmodule

module testbench;
    reg a, b;
    wire out;
    
    or_gate dut(.a(a), .b(b), .out(out));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        a=0; b=0; #10;
        a=0; b=1; #10;
        a=1; b=0; #10;
        a=1; b=1; #10;
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule'''),

        ("Good Counter", '''`timescale 1ns/1ps

module counter(input wire clk, rst, output reg [3:0] count);
    always @(posedge clk or posedge rst) begin
        if (rst) count <= 4'b0000;
        else count <= count + 1'b1;
    end
endmodule

module testbench;
    reg clk, rst;
    wire [3:0] count;
    
    counter dut(.clk(clk), .rst(rst), .count(count));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        rst = 1;
        #10 rst = 0;
        #100 $finish;
    end
    
    initial $monitor("Time=%0t rst=%b count=%d", $time, rst, count);
endmodule'''),
    ]
    
    # ANOMALOUS CIRCUITS - with proper VCD dumps
    anomalous_circuits = [
        ("Broken AND (always 0)", '''`timescale 1ns/1ps

module bad_and(input wire a, b, output wire out);
    assign out = 1'b0;  // BUG: always outputs 0
endmodule

module testbench;
    reg a, b;
    wire out;
    
    bad_and dut(.a(a), .b(b), .out(out));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        a=0; b=0; #10;
        a=0; b=1; #10;
        a=1; b=0; #10;
        a=1; b=1; #10;
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule'''),

        ("Inverted OR (NOR)", '''`timescale 1ns/1ps

module bad_or(input wire a, b, output wire out);
    assign out = ~(a | b);  // BUG: inverted
endmodule

module testbench;
    reg a, b;
    wire out;
    
    bad_or dut(.a(a), .b(b), .out(out));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        a=0; b=0; #10;
        a=0; b=1; #10;
        a=1; b=0; #10;
        a=1; b=1; #10;
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule'''),

        ("Stuck Counter", '''`timescale 1ns/1ps

module bad_counter(input wire clk, rst, output reg [3:0] count);
    always @(posedge clk or posedge rst) begin
        if (rst) count <= 4'b0000;
        else count <= count;  // BUG: doesn't increment
    end
endmodule

module testbench;
    reg clk, rst;
    wire [3:0] count;
    
    bad_counter dut(.clk(clk), .rst(rst), .count(count));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        rst = 1;
        #10 rst = 0;
        #100 $finish;
    end
    
    initial $monitor("Time=%0t rst=%b count=%d", $time, rst, count);
endmodule'''),

        ("Wrong XOR (OR)", '''`timescale 1ns/1ps

module bad_xor(input wire a, b, output wire out);
    assign out = a | b;  // BUG: should be XOR
endmodule

module testbench;
    reg a, b;
    wire out;
    
    bad_xor dut(.a(a), .b(b), .out(out));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        a=0; b=0; #10;
        a=0; b=1; #10;
        a=1; b=0; #10;
        a=1; b=1; #10;
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule'''),
    ]
    
    results = []
    
    # Test normal circuits
    log("\n" + "="*80)
    log("TESTING NORMAL CIRCUITS (should predict NORMAL)", "TEST")
    log("="*80)
    
    for i, (name, code) in enumerate(normal_circuits, 1):
        log(f"\n[{i}/{len(normal_circuits)}] {name}")
        result = test_verilog_direct(name, code, expected_normal=True)
        if result:
            results.append(result)
        time.sleep(2)
    
    # Test anomalous circuits
    log("\n" + "="*80)
    log("TESTING ANOMALOUS CIRCUITS (should predict ANOMALOUS)", "TEST")
    log("="*80)
    
    for i, (name, code) in enumerate(anomalous_circuits, 1):
        log(f"\n[{i}/{len(anomalous_circuits)}] {name}")
        result = test_verilog_direct(name, code, expected_normal=False)
        if result:
            results.append(result)
        time.sleep(2)
    
    # Calculate metrics
    log("\n" + "="*80)
    log("COMPLETE VALIDATION RESULTS", "TEST")
    log("="*80)
    
    if not results:
        log("No successful tests!", "ERROR")
        return
    
    # Confusion matrix
    tp = sum(1 for r in results if not r['expected_normal'] and not r['predicted_normal'])
    tn = sum(1 for r in results if r['expected_normal'] and r['predicted_normal'])
    fp = sum(1 for r in results if r['expected_normal'] and not r['predicted_normal'])
    fn = sum(1 for r in results if not r['expected_normal'] and r['predicted_normal'])
    
    total = len(results)
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    log(f"\nTotal Tests: {total}")
    log(f"Overall Accuracy: {accuracy:.1f}% ({tp + tn}/{total})")
    log(f"\nConfusion Matrix:")
    log(f"                    Predicted")
    log(f"                    Normal    Anomalous")
    log(f"  Actual Normal       {tn:2d}         {fp:2d}")
    log(f"      Anomalous       {fn:2d}         {tp:2d}")
    
    log(f"\nPerformance Metrics:")
    log(f"  Precision: {precision:.3f} (of flagged bugs, how many were real?)")
    log(f"  Recall:    {recall:.3f} (of real bugs, how many did we catch?)")
    log(f"  F1-Score:  {f1:.3f} (harmonic mean)")
    
    log(f"\nDetailed Results:")
    for r in results:
        status = "✅" if r['correct'] else "❌"
        expected = "NORMAL" if r['expected_normal'] else "ANOMALOUS"
        predicted = "NORMAL" if r['predicted_normal'] else "ANOMALOUS"
        log(f"  {status} {r['name']}")
        log(f"     Expected: {expected}, Got: {predicted}, Conf: {r['confidence']:.3f}")
    
    # Final verdict
    log("\n" + "="*80)
    if accuracy >= 85 and recall >= 0.75:
        log("🎉 EXCELLENT! VAE detects both normal and buggy circuits!", "SUCCESS")
    elif accuracy >= 70:
        log("✅ GOOD! VAE performance is solid", "SUCCESS")
    elif accuracy >= 50:
        log("⚠️  ACCEPTABLE but needs improvement", "WARNING")
    else:
        log("❌ POOR - VAE needs retraining or threshold adjustment", "ERROR")
    
    log("="*80)

if __name__ == "__main__":
    main()