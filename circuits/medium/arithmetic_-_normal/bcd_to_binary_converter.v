/*
 * Circuit: BCD to Binary Converter
 * Category: Arithmetic - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bcd_to_binary(input wire [3:0] bcd, output reg [3:0] binary, output reg error);
always @(*) begin
    if (bcd > 4'd9) begin
        binary = 4'd0;
        error = 1'b1;
    end else begin
        binary = bcd;
        error = 1'b0;
    end
end
endmodule
module testbench;
reg [3:0] bcd; wire [3:0] binary; wire error;
bcd_to_binary dut(.bcd(bcd), .binary(binary), .error(error));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    bcd=4'd0;#10; bcd=4'd5;#10; bcd=4'd9;#10; bcd=4'd10;#10; bcd=4'd15;#10; $finish;
end
initial $monitor("Time=%0t bcd=%d binary=%d error=%b", $time, bcd, binary, error);
endmodule