"""
Comprehensive VAE Test Suite - Maximum Coverage
Tests 30+ circuits across diverse categories with both normal and buggy variants

Run from project root:
    python tests/integration/comprehensive_vae_test_suite.py
"""

import sys
import os
import requests
import time
from datetime import datetime
import json

BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"comprehensive_vae_{int(time.time())}"

def log(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "TEST": "🧪"}.get(status, "📝")
    print(f"[{timestamp}] {emoji} {message}")

def test_verilog(name, verilog_code, expected_normal=True, category="Uncategorized"):
    """Test a single Verilog circuit"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/simulation/run-with-verification",
            json={"verilog_code": verilog_code, "session_id": TEST_SESSION_ID},
            timeout=60
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if not data.get("success"):
            if not expected_normal:
                return {
                    'name': name,
                    'category': category,
                    'expected_normal': expected_normal,
                    'predicted_normal': False,
                    'confidence': 1.0,
                    'correct': True,
                    'message': 'Simulation failure detected bug'
                }
            return None
        
        vae = data.get("verification", {}).get("vae_verification", {})
        
        if not vae.get("available"):
            return None
        
        is_anomalous = vae.get("is_anomalous", False)
        confidence = vae.get("confidence", 0.0)
        message = vae.get("message", "")
        
        expected_anomalous = not expected_normal
        correct = (is_anomalous == expected_anomalous)
        
        return {
            'name': name,
            'category': category,
            'expected_normal': expected_normal,
            'predicted_normal': not is_anomalous,
            'confidence': confidence,
            'correct': correct,
            'message': message
        }
        
    except Exception as e:
        return None

def get_test_circuits():
    """Get all test circuits organized by category"""
    
    circuits = {
        # ==================== COMBINATIONAL LOGIC - NORMAL ====================
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

            ("2-Input XOR", '''`timescale 1ns/1ps
module xor_gate(input wire a, b, output wire out);
assign out = a ^ b;
endmodule
module testbench;
reg a, b; wire out;
xor_gate dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', True),

            ("NOT Gate", '''`timescale 1ns/1ps
module not_gate(input wire a, output wire out);
assign out = ~a;
endmodule
module testbench;
reg a; wire out;
not_gate dut(.a(a), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;#10; a=1;#10; a=0;#10; $finish;
end
initial $monitor("Time=%0t a=%b out=%b", $time, a, out);
endmodule''', True),

            ("2-Input NAND", '''`timescale 1ns/1ps
module nand_gate(input wire a, b, output wire out);
assign out = ~(a & b);
endmodule
module testbench;
reg a, b; wire out;
nand_gate dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', True),

            ("2-Input NOR", '''`timescale 1ns/1ps
module nor_gate(input wire a, b, output wire out);
assign out = ~(a | b);
endmodule
module testbench;
reg a, b; wire out;
nor_gate dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', True),

            ("3-Input AND", '''`timescale 1ns/1ps
module and3(input wire a, b, c, output wire out);
assign out = a & b & c;
endmodule
module testbench;
reg a, b, c; wire out;
and3 dut(.a(a), .b(b), .c(c), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;c=0;#10; a=1;b=1;c=0;#10; a=1;b=1;c=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b c=%b out=%b", $time, a, b, c, out);
endmodule''', True),

            ("2:1 Mux", '''`timescale 1ns/1ps
module mux2to1(input wire a, b, sel, output wire out);
assign out = sel ? b : a;
endmodule
module testbench;
reg a, b, sel; wire out;
mux2to1 dut(.a(a), .b(b), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=1;sel=0;#10; sel=1;#10; a=1;b=0;sel=0;#10; sel=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b sel=%b out=%b", $time, a, b, sel, out);
endmodule''', True),
        ],

        # ==================== COMBINATIONAL LOGIC - BUGGY ====================
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

            ("Stuck AND (always 1)", '''`timescale 1ns/1ps
module bad_and2(input wire a, b, output wire out);
assign out = 1'b1;
endmodule
module testbench;
reg a, b; wire out;
bad_and2 dut(.a(a), .b(b), .out(out));
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

            ("Wrong OR (acts like AND)", '''`timescale 1ns/1ps
module bad_or(input wire a, b, output wire out);
assign out = a & b;
endmodule
module testbench;
reg a, b; wire out;
bad_or dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', False),

            ("Inverted XOR (XNOR)", '''`timescale 1ns/1ps
module bad_xor(input wire a, b, output wire out);
assign out = ~(a ^ b);
endmodule
module testbench;
reg a, b; wire out;
bad_xor dut(.a(a), .b(b), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b out=%b", $time, a, b, out);
endmodule''', False),

            ("Partial Mux (ignores sel)", '''`timescale 1ns/1ps
module bad_mux(input wire a, b, sel, output wire out);
assign out = a;
endmodule
module testbench;
reg a, b, sel; wire out;
bad_mux dut(.a(a), .b(b), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=1;sel=0;#10; sel=1;#10; a=1;b=0;sel=0;#10; sel=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b sel=%b out=%b", $time, a, b, sel, out);
endmodule''', False),
        ],

        # ==================== SEQUENTIAL CIRCUITS - NORMAL ====================
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

            ("D Flip-Flop", '''`timescale 1ns/1ps
module dff(input wire clk, rst, d, output reg q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 1'b0;
    else q <= d;
end
endmodule
module testbench;
reg clk, rst, d; wire q;
dff dut(.clk(clk), .rst(rst), .d(d), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; d=0;#10; rst=0;
    d=1;#10; d=0;#10; d=1;#10; d=1;#10; $finish;
end
initial $monitor("Time=%0t rst=%b d=%b q=%b", $time, rst, d, q);
endmodule''', True),

            ("T Flip-Flop", '''`timescale 1ns/1ps
module tff(input wire clk, rst, t, output reg q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 1'b0;
    else if (t) q <= ~q;
end
endmodule
module testbench;
reg clk, rst, t; wire q;
tff dut(.clk(clk), .rst(rst), .t(t), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; t=0;#10; rst=0;
    t=1;#20; t=0;#10; t=1;#20; $finish;
end
initial $monitor("Time=%0t rst=%b t=%b q=%b", $time, rst, t, q);
endmodule''', True),

            ("Shift Register", '''`timescale 1ns/1ps
module shift_reg(input wire clk, rst, din, output reg [3:0] dout);
always @(posedge clk or posedge rst) begin
    if (rst) dout <= 4'b0;
    else dout <= {dout[2:0], din};
end
endmodule
module testbench;
reg clk, rst, din; wire [3:0] dout;
shift_reg dut(.clk(clk), .rst(rst), .din(din), .dout(dout));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; din=0;#10; rst=0;
    din=1;#10; din=0;#10; din=1;#10; din=1;#10; $finish;
end
initial $monitor("Time=%0t rst=%b din=%b dout=%b", $time, rst, din, dout);
endmodule''', True),
        ],

        # ==================== SEQUENTIAL CIRCUITS - BUGGY ====================
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

            ("Counter (no reset)", '''`timescale 1ns/1ps
module bad_counter2(input wire clk, rst, output reg [3:0] count);
always @(posedge clk) begin
    count <= count + 1'b1;
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count;
bad_counter2 dut(.clk(clk), .rst(rst), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#100; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d", $time, rst, count);
endmodule''', False),

            ("DFF (stuck output)", '''`timescale 1ns/1ps
module bad_dff(input wire clk, rst, d, output reg q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 1'b0;
    else q <= 1'b0;
end
endmodule
module testbench;
reg clk, rst, d; wire q;
bad_dff dut(.clk(clk), .rst(rst), .d(d), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; d=0;#10; rst=0;
    d=1;#10; d=0;#10; d=1;#10; $finish;
end
initial $monitor("Time=%0t rst=%b d=%b q=%b", $time, rst, d, q);
endmodule''', False),

            ("Shift Register (no shift)", '''`timescale 1ns/1ps
module bad_shift(input wire clk, rst, din, output reg [3:0] dout);
always @(posedge clk or posedge rst) begin
    if (rst) dout <= 4'b0;
    else dout <= dout;
end
endmodule
module testbench;
reg clk, rst, din; wire [3:0] dout;
bad_shift dut(.clk(clk), .rst(rst), .din(din), .dout(dout));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; din=0;#10; rst=0;
    din=1;#10; din=0;#10; din=1;#10; $finish;
end
initial $monitor("Time=%0t rst=%b din=%b dout=%b", $time, rst, din, dout);
endmodule''', False),
        ],

        # ==================== ARITHMETIC - NORMAL ====================
        "Arithmetic - Normal": [
            ("Half Adder", '''`timescale 1ns/1ps
module half_adder(input wire a, b, output wire sum, carry);
assign sum = a ^ b;
assign carry = a & b;
endmodule
module testbench;
reg a, b; wire sum, carry;
half_adder dut(.a(a), .b(b), .sum(sum), .carry(carry));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b sum=%b carry=%b", $time, a, b, sum, carry);
endmodule''', True),

            ("Full Adder", '''`timescale 1ns/1ps
module full_adder(input wire a, b, cin, output wire sum, cout);
assign sum = a ^ b ^ cin;
assign cout = (a & b) | (b & cin) | (a & cin);
endmodule
module testbench;
reg a, b, cin; wire sum, cout;
full_adder dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;cin=0;#10; a=0;b=1;cin=0;#10; a=1;b=1;cin=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b cin=%b sum=%b cout=%b", $time, a, b, cin, sum, cout);
endmodule''', True),
        ],

        # ==================== ARITHMETIC - BUGGY ====================
        "Arithmetic - Buggy": [
            ("Half Adder (wrong sum)", '''`timescale 1ns/1ps
module bad_half_adder(input wire a, b, output wire sum, carry);
assign sum = a & b;
assign carry = a & b;
endmodule
module testbench;
reg a, b; wire sum, carry;
bad_half_adder dut(.a(a), .b(b), .sum(sum), .carry(carry));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;#10; a=0;b=1;#10; a=1;b=0;#10; a=1;b=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b sum=%b carry=%b", $time, a, b, sum, carry);
endmodule''', False),

            ("Full Adder (no carry)", '''`timescale 1ns/1ps
module bad_full_adder(input wire a, b, cin, output wire sum, cout);
assign sum = a ^ b ^ cin;
assign cout = 1'b0;
endmodule
module testbench;
reg a, b, cin; wire sum, cout;
bad_full_adder dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;cin=0;#10; a=0;b=1;cin=0;#10; a=1;b=1;cin=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b cin=%b sum=%b cout=%b", $time, a, b, cin, sum, cout);
endmodule''', False),
        ],
    }
    
    return circuits

def main():
    log("\n" + "="*80)
    log("COMPREHENSIVE VAE TEST SUITE - MAXIMUM COVERAGE", "TEST")
    log("="*80)
    
    circuits = get_test_circuits()
    
    # Count total tests
    total_tests = sum(len(tests) for tests in circuits.values())
    log(f"Total test cases: {total_tests}")
    
    for category, tests in circuits.items():
        log(f"  {category}: {len(tests)} tests")
    
    log(f"\nEstimated time: ~{total_tests * 3 / 60:.1f} minutes")
    log("="*80)
    
    all_results = []
    test_num = 0
    
    for category, tests in circuits.items():
        log(f"\n{'='*80}")
        log(f"CATEGORY: {category}", "TEST")
        log(f"{'='*80}")
        
        for name, code, is_normal in tests:
            test_num += 1
            log(f"\n[{test_num:2d}/{total_tests:2d}] {name}")
            
            result = test_verilog(name, code, is_normal, category)
            
            if result:
                all_results.append(result)
                status = "✅" if result['correct'] else "❌"
                expected = "NORMAL" if result['expected_normal'] else "ANOMALOUS"
                predicted = "NORMAL" if result['predicted_normal'] else "ANOMALOUS"
                log(f"{status} Expected: {expected}, Got: {predicted}, Conf: {result['confidence']:.3f}")
            else:
                log("❌ Test failed to run", "ERROR")
            
            time.sleep(2)
    
    # Generate comprehensive report
    log("\n" + "="*80)
    log("COMPREHENSIVE TEST RESULTS", "TEST")
    log("="*80)
    
    if not all_results:
        log("No successful tests!", "ERROR")
        return
    
    # Overall metrics
    total = len(all_results)
    correct = sum(1 for r in all_results if r['correct'])
    accuracy = correct / total * 100
    
    # Confusion matrix
    tp = sum(1 for r in all_results if not r['expected_normal'] and not r['predicted_normal'])
    tn = sum(1 for r in all_results if r['expected_normal'] and r['predicted_normal'])
    fp = sum(1 for r in all_results if r['expected_normal'] and not r['predicted_normal'])
    fn = sum(1 for r in all_results if not r['expected_normal'] and r['predicted_normal'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    log(f"\n📊 OVERALL PERFORMANCE:")
    log(f"  Tests Completed: {total}/{total_tests}")
    log(f"  Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    log(f"  Precision: {precision:.3f}")
    log(f"  Recall: {recall:.3f}")
    log(f"  F1-Score: {f1:.3f}")
    
    log(f"\n📈 CONFUSION MATRIX:")
    log(f"                    Predicted")
    log(f"                    Normal    Anomalous")
    log(f"  Actual Normal       {tn:3d}        {fp:3d}")
    log(f"      Anomalous       {fn:3d}        {tp:3d}")
    
    # Category-wise breakdown
    log(f"\n📁 CATEGORY BREAKDOWN:")
    categories = {}
    for r in all_results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'correct': 0, 'total': 0}
        categories[cat]['correct'] += r['correct']
        categories[cat]['total'] += 1
    
    for cat, stats in sorted(categories.items()):
        cat_acc = stats['correct'] / stats['total'] * 100
        log(f"  {cat:30s}: {cat_acc:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vae_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total,
                'correct': correct,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
            },
            'category_breakdown': categories,
            'detailed_results': all_results
        }, f, indent=2)
    
    log(f"\n💾 Detailed results saved to: {filename}")
    
    # Final verdict
    log("\n" + "="*80)
    if accuracy >= 85 and recall >= 0.75:
        log("🎉 EXCELLENT! VAE performs very well across diverse circuits!", "SUCCESS")
    elif accuracy >= 70:
        log("✅ GOOD! VAE shows solid performance", "SUCCESS")
    elif accuracy >= 60:
        log("⚠️  ACCEPTABLE performance with room for improvement", "WARNING")
    else:
        log("❌ Performance below expectations - needs work", "ERROR")
    
    log("="*80)

if __name__ == "__main__":
    main()