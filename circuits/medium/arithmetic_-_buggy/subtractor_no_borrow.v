/*
 * Circuit: Subtractor (no borrow)
 * Category: Arithmetic - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule