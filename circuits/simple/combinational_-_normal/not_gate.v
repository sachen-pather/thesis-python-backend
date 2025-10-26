/*
 * Circuit: NOT Gate
 * Category: Combinational - Normal
 * Complexity: SIMPLE
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule