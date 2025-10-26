/*
 * Circuit: 4-bit Ripple Carry Adder
 * Category: Arithmetic - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module ripple_carry_adder(input wire [3:0] a, b, input wire cin, output wire [3:0] sum, output wire cout);
wire [3:0] carry;
assign {carry[0], sum[0]} = a[0] + b[0] + cin;
assign {carry[1], sum[1]} = a[1] + b[1] + carry[0];
assign {carry[2], sum[2]} = a[2] + b[2] + carry[1];
assign {carry[3], sum[3]} = a[3] + b[3] + carry[2];
assign cout = carry[3];
endmodule
module testbench;
reg [3:0] a, b; reg cin; wire [3:0] sum; wire cout;
ripple_carry_adder dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd3; cin=0;#10; cin=1;#10;
    a=4'd15; b=4'd1; cin=0;#10; a=4'd7; b=4'd8; cin=0;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d cin=%b sum=%d cout=%b", $time, a, b, cin, sum, cout);
endmodule