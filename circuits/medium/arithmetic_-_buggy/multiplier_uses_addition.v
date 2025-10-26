/*
 * Circuit: Multiplier (uses addition)
 * Category: Arithmetic - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule