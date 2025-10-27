`timescale 1ns/1ps

// 4-bit ring counter for machine learning training data
module ring_counter_4bit(
    input wire clk,
    input wire rst,
    output reg [3:0] count
);
    // Ring counter - circulates a single 1 bit
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 4'b0001;  // Initialize with single 1
        else
            count <= {count[2:0], count[3]};  // Rotate left
    end
endmodule

module testbench;
    reg clk, rst;
    wire [3:0] count;
    
    ring_counter_4bit dut(
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
        #100;  // Run through multiple cycles
        
        // Reset again
        rst = 1; #10;
        rst = 0; #40;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b count=%b", 
                     $time, clk, rst, count);
endmodule
