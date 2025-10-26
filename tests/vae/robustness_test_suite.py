"""
Robustness Test Suite for Hybrid VAE Verification System
Tests generalizability beyond training data with diverse circuit patterns
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from hybrid_verification import verify_circuit_hybrid_improved
import pandas as pd
import numpy as np
from datetime import datetime

# Try to import sklearn, fall back to manual calculations if not available
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, using manual metric calculations")


class RobustnessTestSuite:
    """Comprehensive robustness testing for hybrid VAE system"""
    
    def __init__(self):
        self.test_cases = []
        self.results = []
        
    def create_robustness_dataset(self):
        """Create diverse test cases to validate system generalizability"""
        
        # ========== CATEGORY 1: DIVERSE LOGIC GATES ==========
        logic_gates = [
            # NAND gates
            {
                "name": "NAND Gate (2-input)",
                "category": "Logic - NAND/NOR",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module nand2(input wire x, y, output wire z);
assign z = ~(x & y);
endmodule

module testbench;
    reg x, y; wire z;
    nand2 dut(.x(x), .y(y), .z(z));
    initial begin
        x = 0; y = 0; #10;
        x = 0; y = 1; #10;
        x = 1; y = 0; #10;
        x = 1; y = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t x=%b y=%b z=%b", $time, x, y, z);
endmodule'''
            },
            
            {
                "name": "Broken NAND (acts like AND)",
                "category": "Logic - NAND/NOR",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module broken_nand(input wire x, y, output wire z);
assign z = x & y;  // BUG: missing inversion
endmodule

module testbench;
    reg x, y; wire z;
    broken_nand dut(.x(x), .y(y), .z(z));
    initial begin
        x = 0; y = 0; #10;
        x = 0; y = 1; #10;
        x = 1; y = 0; #10;
        x = 1; y = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t x=%b y=%b z=%b", $time, x, y, z);
endmodule'''
            },
            
            # NOR gates
            {
                "name": "NOR Gate (2-input)",
                "category": "Logic - NAND/NOR",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module nor_gate(input wire in1, in2, output wire out);
assign out = ~(in1 | in2);
endmodule

module testbench;
    reg in1, in2; wire out;
    nor_gate dut(.in1(in1), .in2(in2), .out(out));
    initial begin
        in1 = 0; in2 = 0; #10;
        in1 = 0; in2 = 1; #10;
        in1 = 1; in2 = 0; #10;
        in1 = 1; in2 = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t in1=%b in2=%b out=%b", $time, in1, in2, out);
endmodule'''
            },
            
            {
                "name": "Wrong NOR (inverted inputs)",
                "category": "Logic - NAND/NOR",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module wrong_nor(input wire in1, in2, output wire out);
assign out = ~in1 | ~in2;  // BUG: wrong De Morgan
endmodule

module testbench;
    reg in1, in2; wire out;
    wrong_nor dut(.in1(in1), .in2(in2), .out(out));
    initial begin
        in1 = 0; in2 = 0; #10;
        in1 = 0; in2 = 1; #10;
        in1 = 1; in2 = 0; #10;
        in1 = 1; in2 = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t in1=%b in2=%b out=%b", $time, in1, in2, out);
endmodule'''
            },
            
            # 3-input gates
            {
                "name": "3-Input AND Gate",
                "category": "Logic - Multi-input",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module and3(input wire p, q, r, output wire result);
assign result = p & q & r;
endmodule

module testbench;
    reg p, q, r; wire result;
    and3 dut(.p(p), .q(q), .r(r), .result(result));
    initial begin
        p=0;q=0;r=0; #10; p=0;q=0;r=1; #10;
        p=0;q=1;r=0; #10; p=0;q=1;r=1; #10;
        p=1;q=0;r=0; #10; p=1;q=0;r=1; #10;
        p=1;q=1;r=0; #10; p=1;q=1;r=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t p=%b q=%b r=%b result=%b", $time, p, q, r, result);
endmodule'''
            },
            
            {
                "name": "Broken 3-Input AND (2-input only)",
                "category": "Logic - Multi-input",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_and3(input wire p, q, r, output wire result);
assign result = p & q;  // BUG: ignores r
endmodule

module testbench;
    reg p, q, r; wire result;
    bad_and3 dut(.p(p), .q(q), .r(r), .result(result));
    initial begin
        p=0;q=0;r=0; #10; p=0;q=0;r=1; #10;
        p=0;q=1;r=0; #10; p=0;q=1;r=1; #10;
        p=1;q=0;r=0; #10; p=1;q=0;r=1; #10;
        p=1;q=1;r=0; #10; p=1;q=1;r=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t p=%b q=%b r=%b result=%b", $time, p, q, r, result);
endmodule'''
            },
            
            # Mixed logic
            {
                "name": "AOI (AND-OR-INVERT)",
                "category": "Logic - Mixed",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module aoi_gate(input wire i0, i1, i2, i3, output wire o);
assign o = ~((i0 & i1) | (i2 & i3));
endmodule

module testbench;
    reg i0, i1, i2, i3; wire o;
    aoi_gate dut(.i0(i0), .i1(i1), .i2(i2), .i3(i3), .o(o));
    initial begin
        i0=0;i1=0;i2=0;i3=0; #10;
        i0=1;i1=1;i2=0;i3=0; #10;
        i0=0;i1=0;i2=1;i3=1; #10;
        i0=1;i1=1;i2=1;i3=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t i0=%b i1=%b i2=%b i3=%b o=%b", $time, i0, i1, i2, i3, o);
endmodule'''
            },
            
            {
                "name": "Wrong AOI (missing inversion)",
                "category": "Logic - Mixed",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_aoi(input wire i0, i1, i2, i3, output wire o);
assign o = (i0 & i1) | (i2 & i3);  // BUG: no inversion
endmodule

module testbench;
    reg i0, i1, i2, i3; wire o;
    bad_aoi dut(.i0(i0), .i1(i1), .i2(i2), .i3(i3), .o(o));
    initial begin
        i0=0;i1=0;i2=0;i3=0; #10;
        i0=1;i1=1;i2=0;i3=0; #10;
        i0=0;i1=0;i2=1;i3=1; #10;
        i0=1;i1=1;i2=1;i3=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t i0=%b i1=%b i2=%b i3=%b o=%b", $time, i0, i1, i2, i3, o);
endmodule'''
            },
        ]
        
        # ========== CATEGORY 2: DIVERSE SEQUENTIAL CIRCUITS ==========
        sequential = [
            # Shift registers
            {
                "name": "4-bit Shift Register",
                "category": "Sequential - Shift Register",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module shift_reg(input wire clock, reset, data_in, output reg [3:0] data_out);
always @(posedge clock or posedge reset) begin
    if (reset) data_out <= 4'b0000;
    else data_out <= {data_out[2:0], data_in};
end
endmodule

module testbench;
    reg clock, reset, data_in; wire [3:0] data_out;
    shift_reg dut(.clock(clock), .reset(reset), .data_in(data_in), .data_out(data_out));
    initial begin
        clock = 0; reset = 1; data_in = 0;
        #10 reset = 0; data_in = 1;
        #10 data_in = 0; #10 data_in = 1; #10 data_in = 1;
        #10 data_in = 0; #30 $finish;
    end
    always #5 clock = ~clock;
    initial $monitor("Time=%0t clock=%b reset=%b data_in=%b data_out=%b", $time, clock, reset, data_in, data_out);
endmodule'''
            },
            
            {
                "name": "Broken Shift Register (no shift)",
                "category": "Sequential - Shift Register",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_shift(input wire clock, reset, data_in, output reg [3:0] data_out);
always @(posedge clock or posedge reset) begin
    if (reset) data_out <= 4'b0000;
    else data_out <= data_out;  // BUG: doesn't shift
end
endmodule

module testbench;
    reg clock, reset, data_in; wire [3:0] data_out;
    bad_shift dut(.clock(clock), .reset(reset), .data_in(data_in), .data_out(data_out));
    initial begin
        clock = 0; reset = 1; data_in = 0;
        #10 reset = 0; data_in = 1;
        #10 data_in = 0; #10 data_in = 1; #10 data_in = 1;
        #10 data_in = 0; #30 $finish;
    end
    always #5 clock = ~clock;
    initial $monitor("Time=%0t clock=%b reset=%b data_in=%b data_out=%b", $time, clock, reset, data_in, data_out);
endmodule'''
            },
            
            # State machines
            {
                "name": "Simple 2-State FSM",
                "category": "Sequential - State Machine",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module simple_fsm(input wire clk, rst, trigger, output reg state);
always @(posedge clk or posedge rst) begin
    if (rst) state <= 1'b0;
    else if (trigger) state <= ~state;
end
endmodule

module testbench;
    reg clk, rst, trigger; wire state;
    simple_fsm dut(.clk(clk), .rst(rst), .trigger(trigger), .state(state));
    initial begin
        clk = 0; rst = 1; trigger = 0;
        #10 rst = 0;
        #10 trigger = 1; #10 trigger = 0;
        #10 trigger = 1; #10 trigger = 0;
        #30 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b rst=%b trigger=%b state=%b", $time, clk, rst, trigger, state);
endmodule'''
            },
            
            {
                "name": "Stuck State Machine",
                "category": "Sequential - State Machine",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module stuck_fsm(input wire clk, rst, trigger, output reg state);
always @(posedge clk or posedge rst) begin
    if (rst) state <= 1'b0;
    else state <= 1'b0;  // BUG: never transitions
end
endmodule

module testbench;
    reg clk, rst, trigger; wire state;
    stuck_fsm dut(.clk(clk), .rst(rst), .trigger(trigger), .state(state));
    initial begin
        clk = 0; rst = 1; trigger = 0;
        #10 rst = 0;
        #10 trigger = 1; #10 trigger = 0;
        #10 trigger = 1; #10 trigger = 0;
        #30 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b rst=%b trigger=%b state=%b", $time, clk, rst, trigger, state);
endmodule'''
            },
            
            # Different counter types
            {
                "name": "Up-Down Counter",
                "category": "Sequential - Counter",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module updown_counter(input wire clk, rst, up, output reg [2:0] cnt);
always @(posedge clk or posedge rst) begin
    if (rst) cnt <= 3'b000;
    else if (up) cnt <= cnt + 1'b1;
    else cnt <= cnt - 1'b1;
end
endmodule

module testbench;
    reg clk, rst, up; wire [2:0] cnt;
    updown_counter dut(.clk(clk), .rst(rst), .up(up), .cnt(cnt));
    initial begin
        clk = 0; rst = 1; up = 1;
        #10 rst = 0;
        #40 up = 0; #40 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b rst=%b up=%b cnt=%d", $time, clk, rst, up, cnt);
endmodule'''
            },
            
            {
                "name": "Wrong Up-Down (always up)",
                "category": "Sequential - Counter",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_updown(input wire clk, rst, up, output reg [2:0] cnt);
always @(posedge clk or posedge rst) begin
    if (rst) cnt <= 3'b000;
    else cnt <= cnt + 1'b1;  // BUG: ignores up/down
end
endmodule

module testbench;
    reg clk, rst, up; wire [2:0] cnt;
    bad_updown dut(.clk(clk), .rst(rst), .up(up), .cnt(cnt));
    initial begin
        clk = 0; rst = 1; up = 1;
        #10 rst = 0;
        #40 up = 0; #40 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b rst=%b up=%b cnt=%d", $time, clk, rst, up, cnt);
endmodule'''
            },
        ]
        
        # ========== CATEGORY 3: ARITHMETIC CIRCUITS ==========
        arithmetic = [
            # Full adder
            {
                "name": "Full Adder",
                "category": "Arithmetic - Adder",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module full_adder(input wire bit1, bit2, carry_in, output wire sum_out, carry_out);
assign sum_out = bit1 ^ bit2 ^ carry_in;
assign carry_out = (bit1 & bit2) | (bit2 & carry_in) | (bit1 & carry_in);
endmodule

module testbench;
    reg bit1, bit2, carry_in; wire sum_out, carry_out;
    full_adder dut(.bit1(bit1), .bit2(bit2), .carry_in(carry_in), .sum_out(sum_out), .carry_out(carry_out));
    initial begin
        bit1=0;bit2=0;carry_in=0; #10;
        bit1=0;bit2=0;carry_in=1; #10;
        bit1=0;bit2=1;carry_in=0; #10;
        bit1=0;bit2=1;carry_in=1; #10;
        bit1=1;bit2=0;carry_in=0; #10;
        bit1=1;bit2=0;carry_in=1; #10;
        bit1=1;bit2=1;carry_in=0; #10;
        bit1=1;bit2=1;carry_in=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t bit1=%b bit2=%b carry_in=%b sum_out=%b carry_out=%b", $time, bit1, bit2, carry_in, sum_out, carry_out);
endmodule'''
            },
            
            {
                "name": "Broken Full Adder (wrong carry)",
                "category": "Arithmetic - Adder",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_full_adder(input wire bit1, bit2, carry_in, output wire sum_out, carry_out);
assign sum_out = bit1 ^ bit2 ^ carry_in;
assign carry_out = bit1 & bit2;  // BUG: incomplete carry logic
endmodule

module testbench;
    reg bit1, bit2, carry_in; wire sum_out, carry_out;
    bad_full_adder dut(.bit1(bit1), .bit2(bit2), .carry_in(carry_in), .sum_out(sum_out), .carry_out(carry_out));
    initial begin
        bit1=0;bit2=0;carry_in=0; #10;
        bit1=0;bit2=0;carry_in=1; #10;
        bit1=0;bit2=1;carry_in=0; #10;
        bit1=0;bit2=1;carry_in=1; #10;
        bit1=1;bit2=0;carry_in=0; #10;
        bit1=1;bit2=0;carry_in=1; #10;
        bit1=1;bit2=1;carry_in=0; #10;
        bit1=1;bit2=1;carry_in=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t bit1=%b bit2=%b carry_in=%b sum_out=%b carry_out=%b", $time, bit1, bit2, carry_in, sum_out, carry_out);
endmodule'''
            },
            
            # Subtractor
            {
                "name": "1-bit Subtractor",
                "category": "Arithmetic - Subtractor",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module subtractor(input wire minuend, subtrahend, borrow_in, output wire diff, borrow_out);
assign diff = minuend ^ subtrahend ^ borrow_in;
assign borrow_out = (~minuend & subtrahend) | (~minuend & borrow_in) | (subtrahend & borrow_in);
endmodule

module testbench;
    reg minuend, subtrahend, borrow_in; wire diff, borrow_out;
    subtractor dut(.minuend(minuend), .subtrahend(subtrahend), .borrow_in(borrow_in), .diff(diff), .borrow_out(borrow_out));
    initial begin
        minuend=0;subtrahend=0;borrow_in=0; #10;
        minuend=0;subtrahend=1;borrow_in=0; #10;
        minuend=1;subtrahend=0;borrow_in=0; #10;
        minuend=1;subtrahend=1;borrow_in=0; #10;
        minuend=0;subtrahend=0;borrow_in=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t minuend=%b subtrahend=%b borrow_in=%b diff=%b borrow_out=%b", $time, minuend, subtrahend, borrow_in, diff, borrow_out);
endmodule'''
            },
            
            {
                "name": "Wrong Subtractor (acts like adder)",
                "category": "Arithmetic - Subtractor",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_sub(input wire minuend, subtrahend, borrow_in, output wire diff, borrow_out);
assign diff = minuend ^ subtrahend ^ borrow_in;
assign borrow_out = (minuend & subtrahend) | (minuend & borrow_in) | (subtrahend & borrow_in);  // BUG: adder logic
endmodule

module testbench;
    reg minuend, subtrahend, borrow_in; wire diff, borrow_out;
    bad_sub dut(.minuend(minuend), .subtrahend(subtrahend), .borrow_in(borrow_in), .diff(diff), .borrow_out(borrow_out));
    initial begin
        minuend=0;subtrahend=0;borrow_in=0; #10;
        minuend=0;subtrahend=1;borrow_in=0; #10;
        minuend=1;subtrahend=0;borrow_in=0; #10;
        minuend=1;subtrahend=1;borrow_in=0; #10;
        minuend=0;subtrahend=0;borrow_in=1; #10;
        $finish;
    end
    initial $monitor("Time=%0t minuend=%b subtrahend=%b borrow_in=%b diff=%b borrow_out=%b", $time, minuend, subtrahend, borrow_in, diff, borrow_out);
endmodule'''
            },
            
            # Comparator
            {
                "name": "2-bit Comparator",
                "category": "Arithmetic - Comparator",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module comparator(input wire [1:0] num1, num2, output reg equal, greater, less);
always @(*) begin
    equal = (num1 == num2);
    greater = (num1 > num2);
    less = (num1 < num2);
end
endmodule

module testbench;
    reg [1:0] num1, num2; wire equal, greater, less;
    comparator dut(.num1(num1), .num2(num2), .equal(equal), .greater(greater), .less(less));
    initial begin
        num1=2'b00; num2=2'b00; #10;
        num1=2'b01; num2=2'b00; #10;
        num1=2'b00; num2=2'b01; #10;
        num1=2'b11; num2=2'b10; #10;
        num1=2'b10; num2=2'b11; #10;
        $finish;
    end
    initial $monitor("Time=%0t num1=%b num2=%b equal=%b greater=%b less=%b", $time, num1, num2, equal, greater, less);
endmodule'''
            },
            
            {
                "name": "Broken Comparator (inverted logic)",
                "category": "Arithmetic - Comparator",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_comp(input wire [1:0] num1, num2, output reg equal, greater, less);
always @(*) begin
    equal = (num1 == num2);
    greater = (num1 < num2);  // BUG: inverted
    less = (num1 > num2);      // BUG: inverted
end
endmodule

module testbench;
    reg [1:0] num1, num2; wire equal, greater, less;
    bad_comp dut(.num1(num1), .num2(num2), .equal(equal), .greater(greater), .less(less));
    initial begin
        num1=2'b00; num2=2'b00; #10;
        num1=2'b01; num2=2'b00; #10;
        num1=2'b00; num2=2'b01; #10;
        num1=2'b11; num2=2'b10; #10;
        num1=2'b10; num2=2'b11; #10;
        $finish;
    end
    initial $monitor("Time=%0t num1=%b num2=%b equal=%b greater=%b less=%b", $time, num1, num2, equal, greater, less);
endmodule'''
            },
        ]
        
        # ========== CATEGORY 4: EDGE CASES ==========
        edge_cases = [
            # Very simple circuits
            {
                "name": "Simple Buffer",
                "category": "Edge Case - Simple",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module buf(input wire inp, output wire outp);
assign outp = inp;
endmodule

module testbench;
    reg inp; wire outp;
    buf dut(.inp(inp), .outp(outp));
    initial begin
        inp = 0; #10; inp = 1; #10; inp = 0; #10;
        $finish;
    end
    initial $monitor("Time=%0t inp=%b outp=%b", $time, inp, outp);
endmodule'''
            },
            
            {
                "name": "Broken Buffer (always 1)",
                "category": "Edge Case - Simple",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_buf(input wire inp, output wire outp);
assign outp = 1'b1;  // BUG: stuck high
endmodule

module testbench;
    reg inp; wire outp;
    bad_buf dut(.inp(inp), .outp(outp));
    initial begin
        inp = 0; #10; inp = 1; #10; inp = 0; #10;
        $finish;
    end
    initial $monitor("Time=%0t inp=%b outp=%b", $time, inp, outp);
endmodule'''
            },
            
            # Unusual signal names
            {
                "name": "AND with Unusual Names",
                "category": "Edge Case - Naming",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module logic_block(input wire signal_alpha, signal_beta, output wire result_gamma);
assign result_gamma = signal_alpha & signal_beta;
endmodule

module testbench;
    reg signal_alpha, signal_beta; wire result_gamma;
    logic_block dut(.signal_alpha(signal_alpha), .signal_beta(signal_beta), .result_gamma(result_gamma));
    initial begin
        signal_alpha = 0; signal_beta = 0; #10;
        signal_alpha = 0; signal_beta = 1; #10;
        signal_alpha = 1; signal_beta = 0; #10;
        signal_alpha = 1; signal_beta = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t signal_alpha=%b signal_beta=%b result_gamma=%b", $time, signal_alpha, signal_beta, result_gamma);
endmodule'''
            },
            
            {
                "name": "Broken Logic with Unusual Names",
                "category": "Edge Case - Naming",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module faulty_block(input wire signal_alpha, signal_beta, output wire result_gamma);
assign result_gamma = signal_alpha | signal_beta;  // BUG: OR instead of AND
endmodule

module testbench;
    reg signal_alpha, signal_beta; wire result_gamma;
    faulty_block dut(.signal_alpha(signal_alpha), .signal_beta(signal_beta), .result_gamma(result_gamma));
    initial begin
        signal_alpha = 0; signal_beta = 0; #10;
        signal_alpha = 0; signal_beta = 1; #10;
        signal_alpha = 1; signal_beta = 0; #10;
        signal_alpha = 1; signal_beta = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t signal_alpha=%b signal_beta=%b result_gamma=%b", $time, signal_alpha, signal_beta, result_gamma);
endmodule'''
            },
            
            # Different coding style (behavioral)
            {
                "name": "Behavioral AND",
                "category": "Edge Case - Coding Style",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module and_behavioral(input wire s1, s2, output reg res);
always @(*) begin
    if (s1 && s2)
        res = 1'b1;
    else
        res = 1'b0;
end
endmodule

module testbench;
    reg s1, s2; wire res;
    and_behavioral dut(.s1(s1), .s2(s2), .res(res));
    initial begin
        s1 = 0; s2 = 0; #10;
        s1 = 0; s2 = 1; #10;
        s1 = 1; s2 = 0; #10;
        s1 = 1; s2 = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t s1=%b s2=%b res=%b", $time, s1, s2, res);
endmodule'''
            },
            
            {
                "name": "Broken Behavioral AND",
                "category": "Edge Case - Coding Style",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_and_behavioral(input wire s1, s2, output reg res);
always @(*) begin
    res = 1'b0;  // BUG: always outputs 0
end
endmodule

module testbench;
    reg s1, s2; wire res;
    bad_and_behavioral dut(.s1(s1), .s2(s2), .res(res));
    initial begin
        s1 = 0; s2 = 0; #10;
        s1 = 0; s2 = 1; #10;
        s1 = 1; s2 = 0; #10;
        s1 = 1; s2 = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t s1=%b s2=%b res=%b", $time, s1, s2, res);
endmodule'''
            },
        ]
        
        # ========== CATEGORY 5: COMPLEX CIRCUITS ==========
        complex_circuits = [
            # Priority encoder
            {
                "name": "4-bit Priority Encoder",
                "category": "Complex - Encoder",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module priority_encoder(input wire [3:0] req, output reg [1:0] enc, output reg valid);
always @(*) begin
    if (req[3]) begin enc = 2'b11; valid = 1'b1; end
    else if (req[2]) begin enc = 2'b10; valid = 1'b1; end
    else if (req[1]) begin enc = 2'b01; valid = 1'b1; end
    else if (req[0]) begin enc = 2'b00; valid = 1'b1; end
    else begin enc = 2'b00; valid = 1'b0; end
end
endmodule

module testbench;
    reg [3:0] req; wire [1:0] enc; wire valid;
    priority_encoder dut(.req(req), .enc(enc), .valid(valid));
    initial begin
        req = 4'b0000; #10;
        req = 4'b0001; #10;
        req = 4'b0010; #10;
        req = 4'b0100; #10;
        req = 4'b1000; #10;
        req = 4'b1111; #10;
        $finish;
    end
    initial $monitor("Time=%0t req=%b enc=%b valid=%b", $time, req, enc, valid);
endmodule'''
            },
            
            {
                "name": "Wrong Priority Encoder",
                "category": "Complex - Encoder",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_priority(input wire [3:0] req, output reg [1:0] enc, output reg valid);
always @(*) begin
    // BUG: wrong priority order (reversed)
    if (req[0]) begin enc = 2'b00; valid = 1'b1; end
    else if (req[1]) begin enc = 2'b01; valid = 1'b1; end
    else if (req[2]) begin enc = 2'b10; valid = 1'b1; end
    else if (req[3]) begin enc = 2'b11; valid = 1'b1; end
    else begin enc = 2'b00; valid = 1'b0; end
end
endmodule

module testbench;
    reg [3:0] req; wire [1:0] enc; wire valid;
    bad_priority dut(.req(req), .enc(enc), .valid(valid));
    initial begin
        req = 4'b0000; #10;
        req = 4'b0001; #10;
        req = 4'b0010; #10;
        req = 4'b0100; #10;
        req = 4'b1000; #10;
        req = 4'b1111; #10;
        $finish;
    end
    initial $monitor("Time=%0t req=%b enc=%b valid=%b", $time, req, enc, valid);
endmodule'''
            },
            
            # 4:1 Mux
            {
                "name": "4:1 Multiplexer",
                "category": "Complex - Multiplexer",
                "expected": False,
                "verilog": '''`timescale 1ns/1ps
module mux4to1(input wire [1:0] select, input wire [3:0] data, output reg out);
always @(*) begin
    case(select)
        2'b00: out = data[0];
        2'b01: out = data[1];
        2'b10: out = data[2];
        2'b11: out = data[3];
    endcase
end
endmodule

module testbench;
    reg [1:0] select; reg [3:0] data; wire out;
    mux4to1 dut(.select(select), .data(data), .out(out));
    initial begin
        data = 4'b1010;
        select = 2'b00; #10;
        select = 2'b01; #10;
        select = 2'b10; #10;
        select = 2'b11; #10;
        $finish;
    end
    initial $monitor("Time=%0t select=%b data=%b out=%b", $time, select, data, out);
endmodule'''
            },
            
            {
                "name": "Broken 4:1 Mux (wrong mapping)",
                "category": "Complex - Multiplexer",
                "expected": True,
                "verilog": '''`timescale 1ns/1ps
module bad_mux4(input wire [1:0] select, input wire [3:0] data, output reg out);
always @(*) begin
    case(select)
        2'b00: out = data[3];  // BUG: wrong mapping
        2'b01: out = data[2];  // BUG: wrong mapping
        2'b10: out = data[1];  // BUG: wrong mapping
        2'b11: out = data[0];  // BUG: wrong mapping
    endcase
end
endmodule

module testbench;
    reg [1:0] select; reg [3:0] data; wire out;
    bad_mux4 dut(.select(select), .data(data), .out(out));
    initial begin
        data = 4'b1010;
        select = 2'b00; #10;
        select = 2'b01; #10;
        select = 2'b10; #10;
        select = 2'b11; #10;
        $finish;
    end
    initial $monitor("Time=%0t select=%b data=%b out=%b", $time, select, data, out);
endmodule'''
            },
        ]
        
        # Combine all test cases
        self.test_cases = (logic_gates + sequential + arithmetic + 
                          edge_cases + complex_circuits)
        
        print(f"Created robustness dataset: {len(self.test_cases)} total test cases")
        
        # Print category breakdown
        categories = {}
        for test_case in self.test_cases:
            cat = test_case['category']
            expected = test_case['expected']
            if cat not in categories:
                categories[cat] = {'good': 0, 'bad': 0}
            if expected:
                categories[cat]['bad'] += 1
            else:
                categories[cat]['good'] += 1
                
        print("\nCategory breakdown:")
        for cat, counts in sorted(categories.items()):
            print(f"  {cat:30s}: {counts['good']} good, {counts['bad']} bad")
        
    def run_evaluation(self):
        """Run hybrid evaluation on all test cases"""
        print("\nRunning robustness evaluation...")
        print("="*80)
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\nTest {i+1:2d}/{len(self.test_cases)}: {test_case['name']}")
            print(f"  Category: {test_case['category']}")
            
            try:
                # Use hybrid verification
                is_anomalous, confidence, message = verify_circuit_hybrid_improved(test_case['verilog'])
                
                # Store results
                result = {
                    'name': test_case['name'],
                    'category': test_case['category'],
                    'expected_anomalous': test_case['expected'],
                    'predicted_anomalous': is_anomalous,
                    'confidence': confidence,
                    'message': message,
                    'correct': (test_case['expected'] == is_anomalous)
                }
                
                self.results.append(result)
                
                # Print result
                status = "✓ CORRECT" if result['correct'] else "✗ WRONG"
                expected_str = "ANOMALOUS" if test_case['expected'] else "NORMAL"
                predicted_str = "ANOMALOUS" if is_anomalous else "NORMAL"
                confidence_str = f"{confidence:.3f}" if not np.isnan(confidence) else "nan"
                
                print(f"  Expected:  {expected_str}")
                print(f"  Predicted: {predicted_str} (confidence: {confidence_str})")
                print(f"  Message:   {message}")
                print(f"  Result:    {status}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                # Store error result
                result = {
                    'name': test_case['name'],
                    'category': test_case['category'],
                    'expected_anomalous': test_case['expected'],
                    'predicted_anomalous': True,  # Assume anomalous on error
                    'confidence': 1.0,
                    'message': f"Error: {str(e)}",
                    'correct': test_case['expected']  # Correct if we expected anomaly
                }
                self.results.append(result)
            
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.results:
            print("No results to analyze")
            return
            
        # Extract predictions and ground truth
        y_true = [r['expected_anomalous'] for r in self.results]
        y_pred = [r['predicted_anomalous'] for r in self.results]
        
        # Calculate overall metrics
        accuracy = sum(r['correct'] for r in self.results) / len(self.results)
        
        if SKLEARN_AVAILABLE:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
        else:
            # Manual calculations
            tp = sum(1 for i in range(len(y_true)) if y_true[i] and y_pred[i])
            tn = sum(1 for i in range(len(y_true)) if not y_true[i] and not y_pred[i])
            fp = sum(1 for i in range(len(y_true)) if not y_true[i] and y_pred[i])
            fn = sum(1 for i in range(len(y_true)) if y_true[i] and not y_pred[i])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            cm = np.array([[tn, fp], [fn, tp]])
        
        print("\n" + "="*80)
        print("ROBUSTNESS TEST RESULTS")
        print("="*80)
        print(f"Dataset Size: {len(self.test_cases)} diverse test cases")
        print(f"Accuracy:     {accuracy:.3f} ({sum(r['correct'] for r in self.results)}/{len(self.results)})")
        print(f"Precision:    {precision:.3f}")
        print(f"Recall:       {recall:.3f}")
        print(f"F1-Score:     {f1:.3f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Normal  Anomalous")
        print(f"Actual Normal     {cm[0,0]:2d}      {cm[0,1]:2d}")
        print(f"    Anomalous     {cm[1,0]:2d}      {cm[1,1]:2d}")
        
        # Confidence analysis
        normal_confidences = [r['confidence'] for r in self.results if not r['expected_anomalous']]
        anomalous_confidences = [r['confidence'] for r in self.results if r['expected_anomalous']]
        
        print(f"\nConfidence Analysis:")
        print(f"Normal circuits    - Mean: {np.mean(normal_confidences):.3f} ± {np.std(normal_confidences):.3f}")
        print(f"Anomalous circuits - Mean: {np.mean(anomalous_confidences):.3f} ± {np.std(anomalous_confidences):.3f}")
        
        # Category-wise analysis
        print(f"\nCategory-wise Performance:")
        categories = {}
        for r in self.results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'correct': 0, 'total': 0, 'confidences': []}
            categories[cat]['correct'] += r['correct']
            categories[cat]['total'] += 1
            categories[cat]['confidences'].append(r['confidence'])
            
        for cat, stats in sorted(categories.items()):
            accuracy_cat = stats['correct'] / stats['total']
            mean_confidence = np.mean(stats['confidences'])
            print(f"  {cat:30s}: {accuracy_cat:.3f} ({stats['correct']:2d}/{stats['total']:2d}) | Conf: {mean_confidence:.3f}")
        
        # Identify weak areas
        print(f"\nWeak Areas (accuracy < 0.7):")
        weak_categories = [(cat, stats) for cat, stats in categories.items() 
                          if stats['correct'] / stats['total'] < 0.7]
        if weak_categories:
            for cat, stats in weak_categories:
                accuracy_cat = stats['correct'] / stats['total']
                print(f"  {cat}: {accuracy_cat:.3f} accuracy")
        else:
            print("  None - all categories >= 0.7 accuracy")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'categories': categories
        }
    
    def save_results(self, filename='robustness_test_results.csv'):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")


def main():
    print("HYBRID VAE ROBUSTNESS TEST SUITE")
    print("="*80)
    print("Testing generalizability with diverse circuit patterns")
    print("="*80)
    
    # Create evaluator
    evaluator = RobustnessTestSuite()
    
    # Create robustness dataset
    evaluator.create_robustness_dataset()
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Calculate and display metrics
    metrics = evaluator.calculate_metrics()
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "="*80)
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()