`timescale 1ns/1ps

// 4-bit left shift register for machine learning training data
module shift_register_left(
    input wire clk,
    input wire rst,
    input wire d,
    output reg [3:0] out
);
    // Left shift register implementation
    always @(posedge clk) begin
        if (rst) begin
            out <= 4'b0000;
        end else begin
            out <= {out[2:0], d};
        end
    end
endmodule

module testbench;
    reg clk, rst, d;
    wire [3:0] out;
    
    shift_register_left dut(
        .clk(clk),
        .rst(rst),
        .d(d),
        .out(out)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        d = 0;
        #10;
        rst = 0;
        
        // Shift in pattern 1011 (bit by bit)
        d = 1; #10;  // Shift in 1: out = 0001
        d = 0; #10;  // Shift in 0: out = 0010  
        d = 1; #10;  // Shift in 1: out = 0101
        d = 1; #10;  // Shift in 1: out = 1011
        
        // Shift in more bits to see shifting behavior
        d = 0; #10;  // Shift in 0: out = 0110 (lost leftmost 1)
        d = 1; #10;  // Shift in 1: out = 1101
        d = 0; #10;  // Shift in 0: out = 1010
        d = 1; #10;  // Shift in 1: out = 0101
        
        // Test reset functionality
        rst = 1; #10;
        rst = 0; 
        
        // Shift in new pattern
        d = 1; #10;  // out = 0001
        d = 1; #10;  // out = 0011
        d = 0; #10;  // out = 0110
        d = 0; #10;  // out = 1100
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b d=%b out=%b (%d)", $time, clk, rst, d, out, out);
endmodule