`timescale 1ns/1ps

// Sequence detector (101) for machine learning training data
module sequence_detector_101(
    input wire clk,
    input wire rst,
    input wire in,
    output reg detected
);
    // State encoding
    reg [1:0] state;
    parameter S0 = 2'b00;  // Initial/waiting for first 1
    parameter S1 = 2'b01;  // Got 1
    parameter S2 = 2'b10;  // Got 10
    
    // Mealy state machine
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S0;
            detected <= 0;
        end else begin
            detected <= 0;
            case (state)
                S0: begin
                    if (in)
                        state <= S1;
                    else
                        state <= S0;
                end
                S1: begin
                    if (in)
                        state <= S1;
                    else
                        state <= S2;
                end
                S2: begin
                    if (in) begin
                        state <= S1;
                        detected <= 1;  // Pattern 101 detected!
                    end else begin
                        state <= S0;
                    end
                end
            endcase
        end
    end
endmodule

module testbench;
    reg clk, rst, in;
    wire detected;
    
    sequence_detector_101 dut(
        .clk(clk),
        .rst(rst),
        .in(in),
        .detected(detected)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        in = 0;
        #10;
        
        // Test sequence: contains multiple 101 patterns
        rst = 0;
        in = 1; #10;  // 1
        in = 0; #10;  // 0
        in = 1; #10;  // 1 -> detected!
        in = 0; #10;  // 0
        in = 1; #10;  // 1 -> detected!
        in = 1; #10;  // 1
        in = 0; #10;  // 0
        in = 0; #10;  // 0
        in = 1; #10;  // 1
        in = 0; #10;  // 0
        in = 1; #10;  // 1 -> detected!
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b in=%b detected=%b", 
                     $time, clk, rst, in, detected);
endmodule
