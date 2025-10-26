/*
 * Circuit: 8-bit Even Parity Generator
 * Category: Combinational - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module parity_gen(input wire [7:0] data, output wire parity);
assign parity = ^data;
endmodule
module testbench;
reg [7:0] data; wire parity;
parity_gen dut(.data(data), .parity(parity));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    data=8'h00;#10; data=8'h01;#10; data=8'h03;#10; data=8'h07;#10;
    data=8'hFF;#10; data=8'hAA;#10; data=8'h55;#10; $finish;
end
initial $monitor("Time=%0t data=%h parity=%b", $time, data, parity);
endmodule