/*
 * Circuit: Loadable Shift Register
 * Category: Sequential - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module loadable_shift(input wire clk, rst, load, din, input wire [3:0] data, output reg [3:0] q);
always @(posedge clk or posedge rst) begin
    if (rst) q <= 4'b0;
    else if (load) q <= data;
    else q <= {q[2:0], din};
end
endmodule
module testbench;
reg clk, rst, load, din; reg [3:0] data; wire [3:0] q;
loadable_shift dut(.clk(clk), .rst(rst), .load(load), .din(din), .data(data), .q(q));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; load=0; din=0; data=4'b0;#10; rst=0;
    load=1; data=4'b1010;#10; load=0; din=1;#40; din=0;#20; $finish;
end
initial $monitor("Time=%0t rst=%b load=%b din=%b data=%b q=%b", $time, rst, load, din, data, q);
endmodule