`timescale 1ns/1ps

// Moore pattern detector (00110) state machine for machine learning training data
module moore_pattern_detector(
    input wire clk,
    input wire rst,
    input wire in,
    output reg found
);
    // State encoding
    reg [2:0] state;
    parameter [2:0] START = 3'b000;
    parameter [2:0] ZERO  = 3'b001;
    parameter [2:0] WAIT  = 3'b010;
    parameter [2:0] S1    = 3'b011;
    parameter [2:0] S2    = 3'b100;
    parameter [2:0] S3    = 3'b101;
    parameter [2:0] FOUND = 3'b110;
    
    // Moore state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= START;
        end else begin
            case (state)
                START: begin
                    if (in == 0) state <= ZERO;
                    else         state <= START;
                end
                ZERO: begin
                    if (in == 0) state <= WAIT;
                    else         state <= START;
                end
                WAIT: begin
                    if (in == 1) state <= S1;
                    else         state <= WAIT;
                end
                S1: begin
                    if (in == 1) state <= S2;
                    else         state <= ZERO;
                end
                S2: begin
                    if (in == 0) state <= S3;
                    else         state <= START;
                end
                S3: begin
                    if (in == 1) state <= FOUND;
                    else         state <= WAIT;
                end
                FOUND: begin
                    if (in == 1) state <= START;
                    else         state <= ZERO;
                end
            endcase
        end
    end
    
    // Output logic (Moore machine - output depends only on state)
    always @(posedge clk) begin
        if (rst) begin
            found <= 1'b0;
        end else begin
            case (state)
                FOUND:   found <= 1'b1;
                default: found <= 1'b0;
            endcase
        end
    end
endmodule

module testbench;
    reg clk, rst, in;
    wire found;
    
    moore_pattern_detector dut(
        .clk(clk),
        .rst(rst),
        .in(in),
        .found(found)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        in = 0;
        #10;
        rst = 0;
        
        // Test sequence containing pattern 00110
        // Pattern: 1001100110101100110
        //                ^     ^    ^  (pattern detected here)
        
        in = 1; #10;  // 1
        in = 0; #10;  // 0 
        in = 0; #10;  // 0
        in = 1; #10;  // 1
        in = 1; #10;  // 1
        in = 0; #10;  // 0 -> Pattern 00110 completed, found=1 next cycle
        in = 0; #10;  // 0 
        in = 1; #10;  // 1
        in = 1; #10;  // 1
        in = 0; #10;  // 0 -> Pattern 00110 completed again, found=1 next cycle
        in = 1; #10;  // 1
        in = 0; #10;  // 0
        in = 1; #10;  // 1
        in = 1; #10;  // 1
        in = 0; #10;  // 0
        in = 0; #10;  // 0
        in = 1; #10;  // 1
        in = 1; #10;  // 1
        in = 0; #10;  // 0 -> Pattern 00110 completed third time, found=1 next cycle
        
        #20;  // Extra cycles to see output
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b in=%b found=%b", $time, clk, rst, in, found);
endmodule