/*
 * Circuit: UpDown Counter (no enable)
 * Category: Sequential - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_updown_counter(input wire clk, rst, up, enable, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else begin
        if (up) count <= count + 1'b1;
        else count <= count - 1'b1;
    end
end
endmodule
module testbench;
reg clk, rst, up, enable; wire [3:0] count;
bad_updown_counter dut(.clk(clk), .rst(rst), .up(up), .enable(enable), .count(count));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; up=1; enable=1;#10; rst=0;#50;
    up=0;#40; enable=0;#20; enable=1;#20; $finish;
end
initial $monitor("Time=%0t rst=%b enable=%b up=%b count=%d", $time, rst, enable, up, count);
endmodule