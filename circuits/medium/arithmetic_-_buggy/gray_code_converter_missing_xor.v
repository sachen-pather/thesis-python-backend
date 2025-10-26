/*
 * Circuit: Gray Code Converter (missing XOR)
 * Category: Arithmetic - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_gray_converter(input wire [3:0] binary, output wire [3:0] gray);
assign gray[3] = binary[3];
assign gray[2] = binary[2];
assign gray[1] = binary[1];
assign gray[0] = binary[0];
endmodule
module testbench;
reg [3:0] binary; wire [3:0] gray;
bad_gray_converter dut(.binary(binary), .gray(gray));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    binary=4'd0;#10; binary=4'd1;#10; binary=4'd2;#10; binary=4'd3;#10;
    binary=4'd7;#10; binary=4'd15;#10; $finish;
end
initial $monitor("Time=%0t binary=%b gray=%b", $time, binary, gray);
endmodule