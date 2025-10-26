/*
 * Circuit: Inverted AND (NAND)
 * Category: Combinational - Buggy
 * Complexity: SIMPLE
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule