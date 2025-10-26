/*
 * Circuit: Counter (no reset)
 * Category: Sequential - Buggy
 * Complexity: SIMPLE
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_counter2(input wire clk, rst, output reg [3:0] count);
always @(posedge clk) begin
    count <= count + 1'b1;
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count;
bad_counter2 dut(.clk(clk), .rst(rst), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#100; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d", $time, rst, count);
endmodule