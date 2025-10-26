/*
 * Circuit: Edge Detector (no delay)
 * Category: Sequential - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_edge_detector(input wire clk, rst, signal, output reg pulse);
always @(posedge clk or posedge rst) begin
    if (rst) pulse <= 1'b0;
    else pulse <= signal;
end
endmodule
module testbench;
reg clk, rst, signal; wire pulse;
bad_edge_detector dut(.clk(clk), .rst(rst), .signal(signal), .pulse(pulse));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; signal=0;#10; rst=0;#10;
    signal=1;#20; signal=0;#20; signal=1;#30; signal=0;#10; $finish;
end
initial $monitor("Time=%0t rst=%b signal=%b pulse=%b", $time, rst, signal, pulse);
endmodule