`timescale 1ns/1ps

module and_gate(
    input wire a,
    input wire b,
    input wire clk,
    input wire rst_n,
    output reg out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        out <= 1'b0;
    else
        out <= a & b;
end

endmodule

module testbench;
    reg a, b, clk, rst_n;
    wire out;

    and_gate dut (
        .a(a),
        .b(b),
        .clk(clk),
        .rst_n(rst_n),
        .out(out)
    );

    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);

        a = 1'b0;
        b = 1'b0;
        clk = 1'b0;
        rst_n = 1'b0;

        #20 rst_n = 1'b1;
        #10 a = 1'b1;
        #10 b = 1'b1;
        #10 a = 1'b0;
        #10 b = 1'b0;
        #20 a = 1'b1;
        #20 b = 1'b1;
        #20 rst_n = 1'b0;
        #20 rst_n = 1'b1;
        #50 $finish;
    end

    always #5 clk = ~clk;

    initial begin
        $monitor("Time=%0t clk=%b rst_n=%b a=%b b=%b out=%b", 
                 $time, clk, rst_n, a, b, out);
    end
endmodule