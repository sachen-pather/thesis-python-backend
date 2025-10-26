/*
 * Circuit: Pulse Generator
 * Category: Sequential - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module pulse_gen(input wire clk, rst, trigger, output reg pulse);
reg [2:0] counter;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        counter <= 3'b0;
        pulse <= 1'b0;
    end else if (trigger && counter == 3'b0) begin
        counter <= 3'b100;
        pulse <= 1'b1;
    end else if (counter > 0) begin
        counter <= counter - 1'b1;
        pulse <= (counter > 1);
    end else begin
        pulse <= 1'b0;
    end
end
endmodule
module testbench;
reg clk, rst, trigger; wire pulse;
pulse_gen dut(.clk(clk), .rst(rst), .trigger(trigger), .pulse(pulse));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; trigger=0;#10; rst=0;#10;
    trigger=1;#10; trigger=0;#50; trigger=1;#10; trigger=0;#40; $finish;
end
initial $monitor("Time=%0t rst=%b trigger=%b pulse=%b", $time, rst, trigger, pulse);
endmodule