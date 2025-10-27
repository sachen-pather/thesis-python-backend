`timescale 1ns/1ps

// 3-bit up-down counter for machine learning training data
module updown_counter_3bit(
    input wire clk,
    input wire rst,
    input wire up_down,
    output reg [2:0] count
);
    // Counter with direction control
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 3'b000;
        else if (up_down)
            count <= count + 1;
        else
            count <= count - 1;
    end
endmodule

module testbench;
    reg clk, rst, up_down;
    wire [2:0] count;
    
    updown_counter_3bit dut(
        .clk(clk),
        .rst(rst),
        .up_down(up_down),
        .count(count)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        up_down = 1;
        #10;
        
        // Release reset and count up
        rst = 0;
        up_down = 1;  // Count up
        #60;
        
        // Change to count down
        up_down = 0;  // Count down
        #60;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b up_down=%b count=%b (%d)", 
                     $time, clk, rst, up_down, count, count);
endmodule
