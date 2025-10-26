`timescale 1ns/1ps

// 3-stage pipeline for machine learning training data
// Computes f = (a + b + c - d) * d with pipeline stages
module pipeline(
    input wire clk,
    input wire [7:0] a, b, c, d,
    output reg [7:0] f
);
    // Pipeline registers
    reg [7:0] y1, y2, y3, d1, d2;
    
    // Stage 1: Parallel arithmetic operations
    always @(posedge clk) begin
        y1 <= a + b;
        y2 <= c - d;
        d1 <= d;
    end
    
    // Stage 2: Combine stage 1 results
    always @(posedge clk) begin
        y3 <= y1 + y2;
        d2 <= d1;
    end
    
    // Stage 3: Final multiplication
    always @(posedge clk) begin
        f <= y3 * d2;
    end
endmodule

module testbench;
    reg clk;
    reg [7:0] a, b, c, d;
    wire [7:0] f;
    
    pipeline dut(
        .clk(clk),
        .a(a), .b(b), .c(c), .d(d),
        .f(f)
    );
    
    initial begin
        // Initialize
        clk = 0;
        a = 0; b = 0; c = 0; d = 0;
        #10;
        
        // Test case 1: a=10, b=5, c=20, d=3
        // Expected: f = (10+5+20-3)*3 = 32*3 = 96 (after 3 clock cycles)
        a = 10; b = 5; c = 20; d = 3; #10;
        
        // Test case 2: a=8, b=12, c=6, d=2  
        // Expected: f = (8+12+6-2)*2 = 24*2 = 48
        a = 8; b = 12; c = 6; d = 2; #10;
        
        // Test case 3: a=15, b=7, c=10, d=4
        // Expected: f = (15+7+10-4)*4 = 28*4 = 112
        a = 15; b = 7; c = 10; d = 4; #10;
        
        // Test case 4: a=1, b=2, c=3, d=1
        // Expected: f = (1+2+3-1)*1 = 5*1 = 5
        a = 1; b = 2; c = 3; d = 1; #10;
        
        // Keep inputs constant to see pipeline drain
        a = 0; b = 0; c = 0; d = 0;
        #40;  // Wait for pipeline to finish processing
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b a=%d b=%d c=%d d=%d f=%d", 
                     $time, clk, a, b, c, d, f);
endmodule