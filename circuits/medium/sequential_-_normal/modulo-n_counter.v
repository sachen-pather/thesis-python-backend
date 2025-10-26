/*
 * Circuit: Modulo-N Counter
 * Category: Sequential - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module mod_counter(input wire clk, rst, output reg [3:0] count, output reg overflow);
always @(posedge clk or posedge rst) begin
    if (rst) begin
        count <= 4'b0;
        overflow <= 1'b0;
    end else if (count == 4'd9) begin
        count <= 4'b0;
        overflow <= 1'b1;
    end else begin
        count <= count + 1'b1;
        overflow <= 1'b0;
    end
end
endmodule
module testbench;
reg clk, rst; wire [3:0] count; wire overflow;
mod_counter dut(.clk(clk), .rst(rst), .count(count), .overflow(overflow));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1;#10; rst=0;#120; $finish;
end
initial $monitor("Time=%0t rst=%b count=%d overflow=%b", $time, rst, count, overflow);
endmodule