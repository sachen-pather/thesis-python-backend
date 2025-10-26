/*
 * Circuit: Sequence Detector (wrong pattern)
 * Category: State Machines - Buggy
 * Complexity: COMPLEX
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_sequence_detector(input wire clk, rst, din, output reg detected);
localparam S0=0, S1=1, S10=2, S101=3, S1011=4;
reg [2:0] state;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= S0;
        detected <= 0;
    end else begin
        detected <= 0;
        case (state)
            S0: state <= din ? S1 : S0;
            S1: state <= din ? S1 : S10;
            S10: state <= din ? S101 : S0;
            S101: begin
                // BUG: detects on 0 instead of 1
                if (!din) begin state <= S1011; detected <= 1; end
                else state <= S1;
            end
            S1011: state <= din ? S1 : S10;
        endcase
    end
end
endmodule

module testbench;
reg clk, rst, din; wire detected;
bad_sequence_detector dut(.clk(clk), .rst(rst), .din(din), .detected(detected));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; din=0; #10; rst=0;
    din=1;#10; din=0;#10; din=1;#10; din=1;#10;
    din=0;#10; din=1;#10; din=0;#10; din=0;#10;
    din=1;#10; din=0;#10; din=1;#10; din=1;#10;
    $finish;
end
initial $monitor("Time=%0t din=%b state=%d detected=%b", $time, din, dut.state, detected);
endmodule