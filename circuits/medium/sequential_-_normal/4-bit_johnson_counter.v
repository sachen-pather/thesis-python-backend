/*
 * Circuit: 4-bit Johnson Counter
 * Category: Sequential - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module johnson_counter(input wire clk, rst, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0000;
    else q <= {q[2:0], ~q[3]};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
johnson_counter dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#120; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule