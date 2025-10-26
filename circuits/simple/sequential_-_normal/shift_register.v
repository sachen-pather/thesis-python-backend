/*
 * Circuit: Shift Register
 * Category: Sequential - Normal
 * Complexity: SIMPLE
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module shift_reg(input wire clk, rst, din, output reg [3:0] dout);
always @(posedge clk or posedge rst) begin
    if (rst) dout <= 4'b0;
    else dout <= {dout[2:0], din};
end
endmodule
module testbench;
reg clk, rst, din; wire [3:0] dout;
shift_reg dut(.clk(clk), .rst(rst), .din(din), .dout(dout));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; din=0;#10; rst=0;
    din=1;#10; din=0;#10; din=1;#10; din=1;#10; $finish;
end
initial $monitor("Time=%0t rst=%b din=%b dout=%b", $time, rst, din, dout);
endmodule