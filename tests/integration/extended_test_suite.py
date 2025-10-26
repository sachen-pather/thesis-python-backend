"""
Extended Test Suite - Medium Complexity Circuits
Tests 48 circuits with realistic medium-complexity designs

SAVE AS: tests/integration/extended_test_suite.py
RUN: python tests/integration/extended_test_suite.py (for standalone testing)
     OR import get_extended_test_circuits() in comparison scripts

This suite focuses on:
- 4-8 bit operations
- 2-4 inputs/outputs
- Realistic logic patterns (comparators, encoders, decoders, ALU slices)
- Mix of combinational and sequential
- Multi-output circuits

NOTE ON VALIDATION:
Some circuits use 'always @(*)' with 'output reg' for combinational logic.
This is valid Verilog - these are NOT sequential circuits and don't need clocks.
The validator may incorrectly flag these as sequential, but they compile and work correctly.
"""

def get_extended_test_circuits():
    """Get medium complexity test circuits organized by category"""
    
    circuits = {
        # ==================== COMBINATIONAL LOGIC - NORMAL (10 circuits) ====================
        "Combinational - Normal": [
            ("4-bit Incrementer", '''`timescale 1ns/1ps
module incrementer(input wire [3:0] a, output wire [3:0] out, output wire overflow);
assign {overflow, out} = a + 4'b0001;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire overflow;
incrementer dut(.a(a), .out(out), .overflow(overflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd0;#10; a=4'd7;#10; a=4'd14;#10; a=4'd15;#10; a=4'd3;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d overflow=%b", $time, a, out, overflow);
endmodule''', True),

            ("4-bit Comparator", '''`timescale 1ns/1ps
module comparator(input wire [3:0] a, b, output wire eq, gt, lt);
assign eq = (a == b);
assign gt = (a > b);
assign lt = (a < b);
endmodule
module testbench;
reg [3:0] a, b; wire eq, gt, lt;
comparator dut(.a(a), .b(b), .eq(eq), .gt(gt), .lt(lt));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd5;#10; a=4'd8; b=4'd3;#10; a=4'd2; b=4'd7;#10; 
    a=4'd15;b=4'd0;#10; a=4'd0; b=4'd15;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d eq=%b gt=%b lt=%b", $time, a, b, eq, gt, lt);
endmodule''', True),

            ("4-to-2 Priority Encoder", '''`timescale 1ns/1ps
module priority_encoder(input wire [3:0] in, output reg [1:0] out, output reg valid);
always @(*) begin
    casez(in)
        4'b1???: begin out = 2'd3; valid = 1'b1; end
        4'b01??: begin out = 2'd2; valid = 1'b1; end
        4'b001?: begin out = 2'd1; valid = 1'b1; end
        4'b0001: begin out = 2'd0; valid = 1'b1; end
        default: begin out = 2'd0; valid = 1'b0; end
    endcase
end
endmodule
module testbench;
reg [3:0] in; wire [1:0] out; wire valid;
priority_encoder dut(.in(in), .out(out), .valid(valid));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b0000;#10; in=4'b0001;#10; in=4'b0010;#10; in=4'b0100;#10;
    in=4'b1000;#10; in=4'b1111;#10; in=4'b0101;#10; $finish;
end
initial $monitor("Time=%0t in=%b out=%d valid=%b", $time, in, out, valid);
endmodule''', True),

            ("2-to-4 Decoder", '''`timescale 1ns/1ps
module decoder(input wire [1:0] in, input wire enable, output reg [3:0] out);
always @(*) begin
    if (enable)
        case(in)
            2'b00: out = 4'b0001;
            2'b01: out = 4'b0010;
            2'b10: out = 4'b0100;
            2'b11: out = 4'b1000;
        endcase
    else
        out = 4'b0000;
end
endmodule
module testbench;
reg [1:0] in; reg enable; wire [3:0] out;
decoder dut(.in(in), .enable(enable), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    enable=1; in=2'd0;#10; in=2'd1;#10; in=2'd2;#10; in=2'd3;#10;
    enable=0; in=2'd2;#10; enable=1; in=2'd1;#10; $finish;
end
initial $monitor("Time=%0t enable=%b in=%d out=%b", $time, enable, in, out);
endmodule''', True),

            ("4:1 Multiplexer", '''`timescale 1ns/1ps
module mux4to1(input wire [3:0] in, input wire [1:0] sel, output wire out);
assign out = in[sel];
endmodule
module testbench;
reg [3:0] in; reg [1:0] sel; wire out;
mux4to1 dut(.in(in), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b1010; sel=2'd0;#10; sel=2'd1;#10; sel=2'd2;#10; sel=2'd3;#10;
    in=4'b0110; sel=2'd0;#10; sel=2'd2;#10; $finish;
end
initial $monitor("Time=%0t in=%b sel=%d out=%b", $time, in, sel, out);
endmodule''', True),

            ("8-bit Even Parity Generator", '''`timescale 1ns/1ps
module parity_gen(input wire [7:0] data, output wire parity);
assign parity = ^data;
endmodule
module testbench;
reg [7:0] data; wire parity;
parity_gen dut(.data(data), .parity(parity));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    data=8'h00;#10; data=8'h01;#10; data=8'h03;#10; data=8'h07;#10;
    data=8'hFF;#10; data=8'hAA;#10; data=8'h55;#10; $finish;
end
initial $monitor("Time=%0t data=%h parity=%b", $time, data, parity);
endmodule''', True),

            ("4-bit Barrel Shifter", '''`timescale 1ns/1ps
module barrel_shifter(input wire [3:0] in, input wire [1:0] shift, input wire dir, output reg [3:0] out);
always @(*) begin
    if (dir) // Right shift
        out = in >> shift;
    else     // Left shift
        out = in << shift;
end
endmodule
module testbench;
reg [3:0] in; reg [1:0] shift; reg dir; wire [3:0] out;
barrel_shifter dut(.in(in), .shift(shift), .dir(dir), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b1101; dir=0; shift=2'd0;#10; shift=2'd1;#10; shift=2'd2;#10;
    dir=1; shift=2'd0;#10; shift=2'd1;#10; shift=2'd2;#10; $finish;
end
initial $monitor("Time=%0t in=%b dir=%b shift=%d out=%b", $time, in, dir, shift, out);
endmodule''', True),

            ("Simple 4-bit ALU", '''`timescale 1ns/1ps
module alu(input wire [3:0] a, b, input wire [1:0] op, output reg [3:0] out);
always @(*) begin
    case(op)
        2'b00: out = a + b;
        2'b01: out = a - b;
        2'b10: out = a & b;
        2'b11: out = a | b;
    endcase
end
endmodule
module testbench;
reg [3:0] a, b; reg [1:0] op; wire [3:0] out;
alu dut(.a(a), .b(b), .op(op), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd3; op=2'd0;#10; op=2'd1;#10; op=2'd2;#10; op=2'd3;#10;
    a=4'd12; b=4'd7; op=2'd0;#10; op=2'd1;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d op=%d out=%d", $time, a, b, op, out);
endmodule''', True),

            ("8:1 Multiplexer", '''`timescale 1ns/1ps
module mux8to1(input wire [7:0] in, input wire [2:0] sel, output wire out);
assign out = in[sel];
endmodule
module testbench;
reg [7:0] in; reg [2:0] sel; wire out;
mux8to1 dut(.in(in), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=8'b10101100; sel=3'd0;#10; sel=3'd1;#10; sel=3'd2;#10; sel=3'd3;#10;
    sel=3'd4;#10; sel=3'd5;#10; sel=3'd6;#10; sel=3'd7;#10; $finish;
end
initial $monitor("Time=%0t in=%b sel=%d out=%b", $time, in, sel, out);
endmodule''', True),

            ("4-bit Decrementer", '''`timescale 1ns/1ps
module decrementer(input wire [3:0] a, output wire [3:0] out, output wire underflow);
assign {underflow, out} = {1'b0, a} - 5'b00001;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire underflow;
decrementer dut(.a(a), .out(out), .underflow(underflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5;#10; a=4'd1;#10; a=4'd0;#10; a=4'd15;#10; a=4'd8;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d underflow=%b", $time, a, out, underflow);
endmodule''', True),
        ],

        # ==================== COMBINATIONAL LOGIC - BUGGY (10 circuits) ====================
        "Combinational - Buggy": [
            ("Incrementer (always adds 2)", '''`timescale 1ns/1ps
module bad_incrementer(input wire [3:0] a, output wire [3:0] out, output wire overflow);
assign {overflow, out} = a + 4'b0010;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire overflow;
bad_incrementer dut(.a(a), .out(out), .overflow(overflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd0;#10; a=4'd7;#10; a=4'd14;#10; a=4'd15;#10; a=4'd3;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d overflow=%b", $time, a, out, overflow);
endmodule''', False),

            ("Comparator (inverted gt-lt)", '''`timescale 1ns/1ps
module bad_comparator(input wire [3:0] a, b, output wire eq, gt, lt);
assign eq = (a == b);
assign gt = (a < b);
assign lt = (a > b);
endmodule
module testbench;
reg [3:0] a, b; wire eq, gt, lt;
bad_comparator dut(.a(a), .b(b), .eq(eq), .gt(gt), .lt(lt));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd5;#10; a=4'd8; b=4'd3;#10; a=4'd2; b=4'd7;#10;
    a=4'd15;b=4'd0;#10; a=4'd0; b=4'd15;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d eq=%b gt=%b lt=%b", $time, a, b, eq, gt, lt);
endmodule''', False),

            ("Priority Encoder (stuck output)", '''`timescale 1ns/1ps
module bad_priority_encoder(input wire [3:0] in, output reg [1:0] out, output reg valid);
always @(*) begin
    out = 2'd2;
    valid = (in != 4'b0000);
end
endmodule
module testbench;
reg [3:0] in; wire [1:0] out; wire valid;
bad_priority_encoder dut(.in(in), .out(out), .valid(valid));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b0000;#10; in=4'b0001;#10; in=4'b0010;#10; in=4'b0100;#10;
    in=4'b1000;#10; in=4'b1111;#10; in=4'b0101;#10; $finish;
end
initial $monitor("Time=%0t in=%b out=%d valid=%b", $time, in, out, valid);
endmodule''', False),

            ("Decoder (ignores enable)", '''`timescale 1ns/1ps
module bad_decoder(input wire [1:0] in, input wire enable, output reg [3:0] out);
always @(*) begin
    case(in)
        2'b00: out = 4'b0001;
        2'b01: out = 4'b0010;
        2'b10: out = 4'b0100;
        2'b11: out = 4'b1000;
    endcase
end
endmodule
module testbench;
reg [1:0] in; reg enable; wire [3:0] out;
bad_decoder dut(.in(in), .enable(enable), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    enable=1; in=2'd0;#10; in=2'd1;#10; in=2'd2;#10; in=2'd3;#10;
    enable=0; in=2'd2;#10; enable=1; in=2'd1;#10; $finish;
end
initial $monitor("Time=%0t enable=%b in=%d out=%b", $time, enable, in, out);
endmodule''', False),

            ("4:1 Mux (wrong bit order)", '''`timescale 1ns/1ps
module bad_mux4to1(input wire [3:0] in, input wire [1:0] sel, output wire out);
assign out = in[3-sel];
endmodule
module testbench;
reg [3:0] in; reg [1:0] sel; wire out;
bad_mux4to1 dut(.in(in), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b1010; sel=2'd0;#10; sel=2'd1;#10; sel=2'd2;#10; sel=2'd3;#10;
    in=4'b0110; sel=2'd0;#10; sel=2'd2;#10; $finish;
end
initial $monitor("Time=%0t in=%b sel=%d out=%b", $time, in, sel, out);
endmodule''', False),

            ("Parity Generator (always 0)", '''`timescale 1ns/1ps
module bad_parity_gen(input wire [7:0] data, output wire parity);
assign parity = 1'b0;
endmodule
module testbench;
reg [7:0] data; wire parity;
bad_parity_gen dut(.data(data), .parity(parity));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    data=8'h00;#10; data=8'h01;#10; data=8'h03;#10; data=8'h07;#10;
    data=8'hFF;#10; data=8'hAA;#10; data=8'h55;#10; $finish;
end
initial $monitor("Time=%0t data=%h parity=%b", $time, data, parity);
endmodule''', False),

            ("Barrel Shifter (wrong direction)", '''`timescale 1ns/1ps
module bad_barrel_shifter(input wire [3:0] in, input wire [1:0] shift, input wire dir, output reg [3:0] out);
always @(*) begin
    if (dir)
        out = in << shift;
    else
        out = in >> shift;
end
endmodule
module testbench;
reg [3:0] in; reg [1:0] shift; reg dir; wire [3:0] out;
bad_barrel_shifter dut(.in(in), .shift(shift), .dir(dir), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b1101; dir=0; shift=2'd0;#10; shift=2'd1;#10; shift=2'd2;#10;
    dir=1; shift=2'd0;#10; shift=2'd1;#10; shift=2'd2;#10; $finish;
end
initial $monitor("Time=%0t in=%b dir=%b shift=%d out=%b", $time, in, dir, shift, out);
endmodule''', False),

            ("ALU (wrong subtraction)", '''`timescale 1ns/1ps
module bad_alu(input wire [3:0] a, b, input wire [1:0] op, output reg [3:0] out);
always @(*) begin
    case(op)
        2'b00: out = a + b;
        2'b01: out = b - a;
        2'b10: out = a & b;
        2'b11: out = a | b;
    endcase
end
endmodule
module testbench;
reg [3:0] a, b; reg [1:0] op; wire [3:0] out;
bad_alu dut(.a(a), .b(b), .op(op), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd3; op=2'd0;#10; op=2'd1;#10; op=2'd2;#10; op=2'd3;#10;
    a=4'd12; b=4'd7; op=2'd0;#10; op=2'd1;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d op=%d out=%d", $time, a, b, op, out);
endmodule''', False),

            ("8:1 Mux (partial implementation)", '''`timescale 1ns/1ps
module bad_mux8to1(input wire [7:0] in, input wire [2:0] sel, output reg out);
always @(*) begin
    if (sel < 3'd4)
        out = in[sel];
    else
        out = 1'b0;
end
endmodule
module testbench;
reg [7:0] in; reg [2:0] sel; wire out;
bad_mux8to1 dut(.in(in), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=8'b10101100; sel=3'd0;#10; sel=3'd1;#10; sel=3'd2;#10; sel=3'd3;#10;
    sel=3'd4;#10; sel=3'd5;#10; sel=3'd6;#10; sel=3'd7;#10; $finish;
end
initial $monitor("Time=%0t in=%b sel=%d out=%b", $time, in, sel, out);
endmodule''', False),

            ("Decrementer (no underflow)", '''`timescale 1ns/1ps
module bad_decrementer(input wire [3:0] a, output wire [3:0] out, output wire underflow);
assign out = a - 4'b0001;
assign underflow = 1'b0;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire underflow;
bad_decrementer dut(.a(a), .out(out), .underflow(underflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5;#10; a=4'd1;#10; a=4'd0;#10; a=4'd15;#10; a=4'd8;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d underflow=%b", $time, a, out, underflow);
endmodule''', False),
        ],

        # ==================== SEQUENTIAL CIRCUITS - NORMAL (8 circuits) ====================
        "Sequential - Normal": [
            ("4-bit UpDown Counter", '''`timescale 1ns/1ps
module updown_counter(input wire clk, rst, up, enable, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else if (enable) begin
        if (up) count <= count + 1'b1;
        else count <= count - 1'b1;
    end
end
endmodule
module testbench;
reg clk, rst, up, enable; wire [3:0] count;
updown_counter dut(.clk(clk), .rst(rst), .up(up), .enable(enable), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; up=1; enable=1;#10; rst=0;#50;
    up=0;#40; enable=0;#20; enable=1;#20; $finish;
end
initial $monitor("Time=%0t rst=%b enable=%b up=%b count=%d", $time, rst, enable, up, count);
endmodule''', True),

            ("4-bit Ring Counter", '''`timescale 1ns/1ps
module ring_counter(input wire clk, rst, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0001;
    else q <= {q[2:0], q[3]};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
ring_counter dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#100; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule''', True),

            ("4-bit Johnson Counter", '''`timescale 1ns/1ps
module johnson_counter(input wire clk, rst, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0000;
    else q <= {q[2:0], ~q[3]};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
johnson_counter dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#120; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule''', True),

            ("4-bit LFSR", '''`timescale 1ns/1ps
module lfsr(input wire clk, rst, output reg [3:0] q);
wire feedback;
assign feedback = q[3] ^ q[2];
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0001;
    else q <= {q[2:0], feedback};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
lfsr dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#150; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule''', True),

            ("Loadable Shift Register", '''`timescale 1ns/1ps
module loadable_shift(input wire clk, rst, load, din, input wire [3:0] data, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0;
    else if (load) q <= data;
    else q <= {q[2:0], din};
end
endmodule
module testbench;
reg clk, rst, load, din; reg [3:0] data; wire [3:0] q;
loadable_shift dut(.clk(clk), .rst(rst), .load(load), .din(din), .data(data), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; load=0; din=0; data=4'b0;#10; rst=0;
    load=1; data=4'b1010;#10; load=0; din=1;#40; din=0;#20; $finish;
end
initial $monitor("Time=%0t rst=%b load=%b din=%b data=%b q=%b", $time, rst, load, din, data, q);
endmodule''', True),

            ("Edge Detector", '''`timescale 1ns/1ps
module edge_detector(input wire clk, rst, signal, output reg pulse);
reg signal_d;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        signal_d <= 1'b0;
        pulse <= 1'b0;
    end else begin
        signal_d <= signal;
        pulse <= signal & ~signal_d;
    end
end
endmodule
module testbench;
reg clk, rst, signal; wire pulse;
edge_detector dut(.clk(clk), .rst(rst), .signal(signal), .pulse(pulse));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; signal=0;#10; rst=0;#10;
    signal=1;#20; signal=0;#20; signal=1;#30; signal=0;#10; $finish;
end
initial $monitor("Time=%0t rst=%b signal=%b pulse=%b", $time, rst, signal, pulse);
endmodule''', True),

            ("Pulse Generator", '''`timescale 1ns/1ps
module pulse_gen(input wire clk, rst, trigger, output reg pulse);
reg [2:0] counter;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        counter <= 3'b0;
        pulse <= 1'b0;
    end else if (trigger && counter == 3'b0) begin
        counter <= 3'b100;
        pulse <= 1'b1;
    end else if (counter > 0) begin
        counter <= counter - 1'b1;
        pulse <= (counter > 1);
    end else begin
        pulse <= 1'b0;
    end
end
endmodule
module testbench;
reg clk, rst, trigger; wire pulse;
pulse_gen dut(.clk(clk), .rst(rst), .trigger(trigger), .pulse(pulse));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; trigger=0;#10; rst=0;#10;
    trigger=1;#10; trigger=0;#50; trigger=1;#10; trigger=0;#40; $finish;
end
initial $monitor("Time=%0t rst=%b trigger=%b pulse=%b", $time, rst, trigger, pulse);
endmodule''', True),

            ("Modulo-N Counter", '''`timescale 1ns/1ps
module mod_counter(input wire clk, rst, output reg [3:0] count, output reg overflow);
always @(posedge clk or posedge rst) begin
    if (rst) begin
        count <= 4'b0;
        overflow <= 1'b0;
    end else if (count == 4'd9) begin
        count <= 4'b0;
        overflow <= 1'b1;
    end else begin
        count <= count + 1'b1;
        overflow <= 1'b0;
    end
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count; wire overflow;
mod_counter dut(.clk(clk), .rst(rst), .count(count), .overflow(overflow));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#120; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d overflow=%b", $time, rst, count, overflow);
endmodule''', True),
        ],

        # ==================== SEQUENTIAL CIRCUITS - BUGGY (8 circuits) ====================
        "Sequential - Buggy": [
            ("UpDown Counter (no enable)", '''`timescale 1ns/1ps
module bad_updown_counter(input wire clk, rst, up, enable, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else begin
        if (up) count <= count + 1'b1;
        else count <= count - 1'b1;
    end
end
endmodule
module testbench;
reg clk, rst, up, enable; wire [3:0] count;
bad_updown_counter dut(.clk(clk), .rst(rst), .up(up), .enable(enable), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; up=1; enable=1;#10; rst=0;#50;
    up=0;#40; enable=0;#20; enable=1;#20; $finish;
end
initial $monitor("Time=%0t rst=%b enable=%b up=%b count=%d", $time, rst, enable, up, count);
endmodule''', False),

            ("Ring Counter (wrong initialization)", '''`timescale 1ns/1ps
module bad_ring_counter(input wire clk, rst, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0000;
    else q <= {q[2:0], q[3]};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
bad_ring_counter dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#100; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule''', False),

            ("Johnson Counter (no complement)", '''`timescale 1ns/1ps
module bad_johnson_counter(input wire clk, rst, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0000;
    else q <= {q[2:0], q[3]};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
bad_johnson_counter dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#120; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule''', False),

            ("LFSR (wrong feedback)", '''`timescale 1ns/1ps
module bad_lfsr(input wire clk, rst, output reg [3:0] q);
wire feedback;
assign feedback = q[3] & q[2];
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0001;
    else q <= {q[2:0], feedback};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
bad_lfsr dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#150; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule''', False),

            ("Loadable Shift (ignores load)", '''`timescale 1ns/1ps
module bad_loadable_shift(input wire clk, rst, load, din, input wire [3:0] data, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0;
    else q <= {q[2:0], din};
end
endmodule
module testbench;
reg clk, rst, load, din; reg [3:0] data; wire [3:0] q;
bad_loadable_shift dut(.clk(clk), .rst(rst), .load(load), .din(din), .data(data), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; load=0; din=0; data=4'b0;#10; rst=0;
    load=1; data=4'b1010;#10; load=0; din=1;#40; din=0;#20; $finish;
end
initial $monitor("Time=%0t rst=%b load=%b din=%b data=%b q=%b", $time, rst, load, din, data, q);
endmodule''', False),

            ("Edge Detector (no delay)", '''`timescale 1ns/1ps
module bad_edge_detector(input wire clk, rst, signal, output reg pulse);
always @(posedge clk or posedge rst) begin
    if (rst) pulse <= 1'b0;
    else pulse <= signal;
end
endmodule
module testbench;
reg clk, rst, signal; wire pulse;
bad_edge_detector dut(.clk(clk), .rst(rst), .signal(signal), .pulse(pulse));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; signal=0;#10; rst=0;#10;
    signal=1;#20; signal=0;#20; signal=1;#30; signal=0;#10; $finish;
end
initial $monitor("Time=%0t rst=%b signal=%b pulse=%b", $time, rst, signal, pulse);
endmodule''', False),

            ("Pulse Generator (stuck high)", '''`timescale 1ns/1ps
module bad_pulse_gen(input wire clk, rst, trigger, output reg pulse);
reg [2:0] counter;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        counter <= 3'b0;
        pulse <= 1'b0;
    end else if (trigger && counter == 3'b0) begin
        counter <= 3'b100;
        pulse <= 1'b1;
    end else if (counter > 0) begin
        counter <= counter - 1'b1;
        pulse <= 1'b1;
    end else begin
        pulse <= 1'b0;
    end
end
endmodule
module testbench;
reg clk, rst, trigger; wire pulse;
bad_pulse_gen dut(.clk(clk), .rst(rst), .trigger(trigger), .pulse(pulse));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; trigger=0;#10; rst=0;#10;
    trigger=1;#10; trigger=0;#50; trigger=1;#10; trigger=0;#40; $finish;
end
initial $monitor("Time=%0t rst=%b trigger=%b pulse=%b", $time, rst, trigger, pulse);
endmodule''', False),

            ("Modulo-N Counter (wrong limit)", '''`timescale 1ns/1ps
module bad_mod_counter(input wire clk, rst, output reg [3:0] count, output reg overflow);
always @(posedge clk or posedge rst) begin
    if (rst) begin
        count <= 4'b0;
        overflow <= 1'b0;
    end else if (count == 4'd10) begin
        count <= 4'b0;
        overflow <= 1'b1;
    end else begin
        count <= count + 1'b1;
        overflow <= 1'b0;
    end
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count; wire overflow;
bad_mod_counter dut(.clk(clk), .rst(rst), .count(count), .overflow(overflow));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#120; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d overflow=%b", $time, rst, count, overflow);
endmodule''', False),
        ],

        # ==================== ARITHMETIC - NORMAL (6 circuits) ====================
        "Arithmetic - Normal": [
            ("4-bit Ripple Carry Adder", '''`timescale 1ns/1ps
module ripple_carry_adder(input wire [3:0] a, b, input wire cin, output wire [3:0] sum, output wire cout);
wire [3:0] carry;
assign {carry[0], sum[0]} = a[0] + b[0] + cin;
assign {carry[1], sum[1]} = a[1] + b[1] + carry[0];
assign {carry[2], sum[2]} = a[2] + b[2] + carry[1];
assign {carry[3], sum[3]} = a[3] + b[3] + carry[2];
assign cout = carry[3];
endmodule
module testbench;
reg [3:0] a, b; reg cin; wire [3:0] sum; wire cout;
ripple_carry_adder dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd3; cin=0;#10; cin=1;#10;
    a=4'd15; b=4'd1; cin=0;#10; a=4'd7; b=4'd8; cin=0;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d cin=%b sum=%d cout=%b", $time, a, b, cin, sum, cout);
endmodule''', True),

            ("4-bit Subtractor", '''`timescale 1ns/1ps
module subtractor(input wire [3:0] a, b, output wire [3:0] diff, output wire borrow);
assign {borrow, diff} = a - b;
endmodule
module testbench;
reg [3:0] a, b; wire [3:0] diff; wire borrow;
subtractor dut(.a(a), .b(b), .diff(diff), .borrow(borrow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd8; b=4'd3;#10; a=4'd5; b=4'd7;#10;
    a=4'd15; b=4'd1;#10; a=4'd0; b=4'd5;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d diff=%d borrow=%b", $time, a, b, diff, borrow);
endmodule''', True),

            ("2-bit Multiplier", '''`timescale 1ns/1ps
module multiplier(input wire [1:0] a, b, output wire [3:0] product);
assign product = a * b;
endmodule
module testbench;
reg [1:0] a, b; wire [3:0] product;
multiplier dut(.a(a), .b(b), .product(product));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=2'd0; b=2'd0;#10; a=2'd1; b=2'd2;#10; a=2'd2; b=2'd3;#10;
    a=2'd3; b=2'd3;#10; a=2'd2; b=2'd2;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d product=%d", $time, a, b, product);
endmodule''', True),

            ("4-bit Magnitude Comparator", '''`timescale 1ns/1ps
module mag_comparator(input wire [3:0] a, b, output reg eq, gt, lt);
always @(*) begin
    eq = (a == b);
    gt = (a > b);
    lt = (a < b);
end
endmodule
module testbench;
reg [3:0] a, b; wire eq, gt, lt;
mag_comparator dut(.a(a), .b(b), .eq(eq), .gt(gt), .lt(lt));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd5;#10; a=4'd12; b=4'd7;#10; a=4'd3; b=4'd9;#10;
    a=4'd15; b=4'd0;#10; a=4'd0; b=4'd15;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d eq=%b gt=%b lt=%b", $time, a, b, eq, gt, lt);
endmodule''', True),

            ("BCD to Binary Converter", '''`timescale 1ns/1ps
module bcd_to_binary(input wire [3:0] bcd, output reg [3:0] binary, output reg error);
always @(*) begin
    if (bcd > 4'd9) begin
        binary = 4'd0;
        error = 1'b1;
    end else begin
        binary = bcd;
        error = 1'b0;
    end
end
endmodule
module testbench;
reg [3:0] bcd; wire [3:0] binary; wire error;
bcd_to_binary dut(.bcd(bcd), .binary(binary), .error(error));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    bcd=4'd0;#10; bcd=4'd5;#10; bcd=4'd9;#10; bcd=4'd10;#10; bcd=4'd15;#10; $finish;
end
initial $monitor("Time=%0t bcd=%d binary=%d error=%b", $time, bcd, binary, error);
endmodule''', True),

            ("4-bit Gray Code Converter", '''`timescale 1ns/1ps
module gray_converter(input wire [3:0] binary, output wire [3:0] gray);
assign gray[3] = binary[3];
assign gray[2] = binary[3] ^ binary[2];
assign gray[1] = binary[2] ^ binary[1];
assign gray[0] = binary[1] ^ binary[0];
endmodule
module testbench;
reg [3:0] binary; wire [3:0] gray;
gray_converter dut(.binary(binary), .gray(gray));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    binary=4'd0;#10; binary=4'd1;#10; binary=4'd2;#10; binary=4'd3;#10;
    binary=4'd7;#10; binary=4'd15;#10; $finish;
end
initial $monitor("Time=%0t binary=%b gray=%b", $time, binary, gray);
endmodule''', True),
        ],

        # ==================== ARITHMETIC - BUGGY (6 circuits) ====================
        "Arithmetic - Buggy": [
            ("Ripple Carry Adder (broken carry chain)", '''`timescale 1ns/1ps
module bad_ripple_carry_adder(input wire [3:0] a, b, input wire cin, output wire [3:0] sum, output wire cout);
wire [3:0] carry;
assign {carry[0], sum[0]} = a[0] + b[0] + cin;
assign {carry[1], sum[1]} = a[1] + b[1];
assign {carry[2], sum[2]} = a[2] + b[2];
assign {carry[3], sum[3]} = a[3] + b[3];
assign cout = carry[3];
endmodule
module testbench;
reg [3:0] a, b; reg cin; wire [3:0] sum; wire cout;
bad_ripple_carry_adder dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd3; cin=0;#10; cin=1;#10;
    a=4'd15; b=4'd1; cin=0;#10; a=4'd7; b=4'd8; cin=0;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d cin=%b sum=%d cout=%b", $time, a, b, cin, sum, cout);
endmodule''', False),

            ("Subtractor (no borrow)", '''`timescale 1ns/1ps
module bad_subtractor(input wire [3:0] a, b, output wire [3:0] diff, output wire borrow);
assign diff = a - b;
assign borrow = 1'b0;
endmodule
module testbench;
reg [3:0] a, b; wire [3:0] diff; wire borrow;
bad_subtractor dut(.a(a), .b(b), .diff(diff), .borrow(borrow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd8; b=4'd3;#10; a=4'd5; b=4'd7;#10;
    a=4'd15; b=4'd1;#10; a=4'd0; b=4'd5;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d diff=%d borrow=%b", $time, a, b, diff, borrow);
endmodule''', False),

            ("Multiplier (uses addition)", '''`timescale 1ns/1ps
module bad_multiplier(input wire [1:0] a, b, output wire [3:0] product);
assign product = {2'b0, a} + {2'b0, b};
endmodule
module testbench;
reg [1:0] a, b; wire [3:0] product;
bad_multiplier dut(.a(a), .b(b), .product(product));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=2'd0; b=2'd0;#10; a=2'd1; b=2'd2;#10; a=2'd2; b=2'd3;#10;
    a=2'd3; b=2'd3;#10; a=2'd2; b=2'd2;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d product=%d", $time, a, b, product);
endmodule''', False),

            ("Magnitude Comparator (missing lt)", '''`timescale 1ns/1ps
module bad_mag_comparator(input wire [3:0] a, b, output reg eq, gt, lt);
always @(*) begin
    eq = (a == b);
    gt = (a > b);
    lt = 1'b0;
end
endmodule
module testbench;
reg [3:0] a, b; wire eq, gt, lt;
bad_mag_comparator dut(.a(a), .b(b), .eq(eq), .gt(gt), .lt(lt));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd5;#10; a=4'd12; b=4'd7;#10; a=4'd3; b=4'd9;#10;
    a=4'd15; b=4'd0;#10; a=4'd0; b=4'd15;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d eq=%b gt=%b lt=%b", $time, a, b, eq, gt, lt);
endmodule''', False),

            ("BCD Converter (no error checking)", '''`timescale 1ns/1ps
module bad_bcd_to_binary(input wire [3:0] bcd, output reg [3:0] binary, output reg error);
always @(*) begin
    binary = bcd;
    error = 1'b0;
end
endmodule
module testbench;
reg [3:0] bcd; wire [3:0] binary; wire error;
bad_bcd_to_binary dut(.bcd(bcd), .binary(binary), .error(error));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    bcd=4'd0;#10; bcd=4'd5;#10; bcd=4'd9;#10; bcd=4'd10;#10; bcd=4'd15;#10; $finish;
end
initial $monitor("Time=%0t bcd=%d binary=%d error=%b", $time, bcd, binary, error);
endmodule''', False),

            ("Gray Code Converter (missing XOR)", '''`timescale 1ns/1ps
module bad_gray_converter(input wire [3:0] binary, output wire [3:0] gray);
assign gray[3] = binary[3];
assign gray[2] = binary[2];
assign gray[1] = binary[1];
assign gray[0] = binary[0];
endmodule
module testbench;
reg [3:0] binary; wire [3:0] gray;
bad_gray_converter dut(.binary(binary), .gray(gray));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    binary=4'd0;#10; binary=4'd1;#10; binary=4'd2;#10; binary=4'd3;#10;
    binary=4'd7;#10; binary=4'd15;#10; $finish;
end
initial $monitor("Time=%0t binary=%b gray=%b", $time, binary, gray);
endmodule''', False),
        ],
    }
    
    return circuits


# Standalone test functionality (optional)
if __name__ == "__main__":
    circuits = get_extended_test_circuits()
    
    print("="*80)
    print("EXTENDED TEST SUITE - Circuit Inventory")
    print("="*80)
    
    total = 0
    for category, tests in circuits.items():
        print(f"\n{category}: {len(tests)} circuits")
        for name, _, is_normal in tests:
            status = "✓ NORMAL" if is_normal else "✗ BUGGY"
            print(f"  - {name:40s} [{status}]")
        total += len(tests)
    
    print(f"\n{'='*80}")
    print(f"Total circuits: {total}")
    print(f"{'='*80}")