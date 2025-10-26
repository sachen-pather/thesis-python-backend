/*
 * Circuit: 2-bit Multiplier
 * Category: Arithmetic - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule