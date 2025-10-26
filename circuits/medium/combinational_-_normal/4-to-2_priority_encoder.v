/*
 * Circuit: 4-to-2 Priority Encoder
 * Category: Combinational - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module priority_encoder(input wire [3:0] in, output reg [1:0] out, output reg valid);
always @(*) begin
    casez(in)
        4'b1???: begin out = 2'd3; valid = 1'b1; end
        4'b01??: begin out = 2'd2; valid = 1'b1; end
        4'b001?: begin out = 2'd1; valid = 1'b1; end
        4'b0001: begin out = 2'd0; valid = 1'b1; end
        default: begin out = 2'd0; valid = 1'b0; end
    endcase
end
endmodule
module testbench;
reg [3:0] in; wire [1:0] out; wire valid;
priority_encoder dut(.in(in), .out(out), .valid(valid));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b0000;#10; in=4'b0001;#10; in=4'b0010;#10; in=4'b0100;#10;
    in=4'b1000;#10; in=4'b1111;#10; in=4'b0101;#10; $finish;
end
initial $monitor("Time=%0t in=%b out=%d valid=%b", $time, in, out, valid);
endmodule