/*
 * Circuit: Full Adder (no carry)
 * Category: Arithmetic - Buggy
 * Complexity: SIMPLE
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_full_adder(input wire a, b, cin, output wire sum, cout);
assign sum = a ^ b ^ cin;
assign cout = 1'b0;
endmodule
module testbench;
reg a, b, cin; wire sum, cout;
bad_full_adder dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=0;cin=0;#10; a=0;b=1;cin=0;#10; a=1;b=1;cin=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b cin=%b sum=%b cout=%b", $time, a, b, cin, sum, cout);
endmodule