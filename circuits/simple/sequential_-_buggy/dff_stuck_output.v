/*
 * Circuit: DFF (stuck output)
 * Category: Sequential - Buggy
 * Complexity: SIMPLE
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_dff(input wire clk, rst, d, output reg q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 1'b0;
    else q <= 1'b0;
end
endmodule
module testbench;
reg clk, rst, d; wire q;
bad_dff dut(.clk(clk), .rst(rst), .d(d), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; d=0;#10; rst=0;
    d=1;#10; d=0;#10; d=1;#10; $finish;
end
initial $monitor("Time=%0t rst=%b d=%b q=%b", $time, rst, d, q);
endmodule