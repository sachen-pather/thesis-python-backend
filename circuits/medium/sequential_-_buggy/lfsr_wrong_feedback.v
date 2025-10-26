/*
 * Circuit: LFSR (wrong feedback)
 * Category: Sequential - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_lfsr(input wire clk, rst, output reg [3:0] q);
wire feedback;
assign feedback = q[3] & q[2];
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0001;
    else q <= {q[2:0], feedback};
end
endmodule
module testbench;
reg clk, rst; wire [3:0] q;
bad_lfsr dut(.clk(clk), .rst(rst), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#150; $finish;
end
initial $monitor("Time=%0t rst=%b q=%b", $time, rst, q);
endmodule