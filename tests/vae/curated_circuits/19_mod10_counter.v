`timescale 1ns/1ps

// Mod-10 (Decade) counter for machine learning training data
module mod10_counter(
    input wire clk,
    input wire rst,
    output reg [3:0] count
);
    // Counter that counts from 0 to 9 and wraps
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 4'b0000;
        else if (count == 4'b1001)  // When count reaches 9
            count <= 4'b0000;        // Reset to 0
        else
            count <= count + 1;
    end
endmodule

module testbench;
    reg clk, rst;
    wire [3:0] count;
    
    mod10_counter dut(
        .clk(clk),
        .rst(rst),
        .count(count)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        #10;
        
        // Release reset and count
        rst = 0;
        #120;  // Count through 0-9 and wrap around
        
        // Test reset
        rst = 1; #10;
        rst = 0; #30;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b count=%b(%d)", 
                     $time, clk, rst, count, count);
endmodule
