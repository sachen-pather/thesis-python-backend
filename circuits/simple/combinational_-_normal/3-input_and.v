/*
 * Circuit: 3-Input AND
 * Category: Combinational - Normal
 * Complexity: SIMPLE
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule