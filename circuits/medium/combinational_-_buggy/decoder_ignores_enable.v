/*
 * Circuit: Decoder (ignores enable)
 * Category: Combinational - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_decoder(input wire [1:0] in, input wire enable, output reg [3:0] out);
always @(*) begin
    case(in)
        2'b00: out = 4'b0001;
        2'b01: out = 4'b0010;
        2'b10: out = 4'b0100;
        2'b11: out = 4'b1000;
    endcase
end
endmodule
module testbench;
reg [1:0] in; reg enable; wire [3:0] out;
bad_decoder dut(.in(in), .enable(enable), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    enable=1; in=2'd0;#10; in=2'd1;#10; in=2'd2;#10; in=2'd3;#10;
    enable=0; in=2'd2;#10; enable=1; in=2'd1;#10; $finish;
end
initial $monitor("Time=%0t enable=%b in=%d out=%b", $time, enable, in, out);
endmodule