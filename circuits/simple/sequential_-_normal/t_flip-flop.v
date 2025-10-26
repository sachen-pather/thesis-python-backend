/*
 * Circuit: T Flip-Flop
 * Category: Sequential - Normal
 * Complexity: SIMPLE
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module tff(input wire clk, rst, t, output reg q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 1'b0;
    else if (t) q <= ~q;
end
endmodule
module testbench;
reg clk, rst, t; wire q;
tff dut(.clk(clk), .rst(rst), .t(t), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; t=0;#10; rst=0;
    t=1;#20; t=0;#10; t=1;#20; $finish;
end
initial $monitor("Time=%0t rst=%b t=%b q=%b", $time, rst, t, q);
endmodule