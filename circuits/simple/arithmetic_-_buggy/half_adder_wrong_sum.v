/*
 * Circuit: Half Adder (wrong sum)
 * Category: Arithmetic - Buggy
 * Complexity: SIMPLE
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule