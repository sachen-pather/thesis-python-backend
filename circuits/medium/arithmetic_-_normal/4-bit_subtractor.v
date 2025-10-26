/*
 * Circuit: 4-bit Subtractor
 * Category: Arithmetic - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
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
endmodule