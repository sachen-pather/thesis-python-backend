/*
 * Circuit: 4-bit Counter
 * Category: Sequential - Normal
 * Complexity: SIMPLE
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module counter(input wire clk, rst, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else count <= count + 1'b1;
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count;
counter dut(.clk(clk), .rst(rst), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#100; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d", $time, rst, count);
endmodule