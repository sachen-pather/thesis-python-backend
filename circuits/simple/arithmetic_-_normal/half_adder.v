/*
 * Circuit: Half Adder
 * Category: Arithmetic - Normal
 * Complexity: SIMPLE
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule