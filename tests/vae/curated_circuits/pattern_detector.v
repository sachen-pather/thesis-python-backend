`timescale 1ns/1ps

// Pattern detector (00110) Mealy state machine for machine learning training data
module pattern_detector(
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
    
    // Mealy state machine
    always @(posedge clk) begin
        if (rst) begin
            state <= START;
            found <= 1'b0;
        end else begin
            case (state)
                START: begin
                    if (in == 0) begin state <= ZERO;  found <= 1'b0; end
                    else         begin state <= START; found <= 1'b0; end
                end
                ZERO: begin
                    if (in == 0) begin state <= WAIT;  found <= 1'b0; end
                    else         begin state <= START; found <= 1'b0; end
                end
                WAIT: begin
                    if (in == 1) begin state <= S1;    found <= 1'b0; end
                    else         begin state <= WAIT;  found <= 1'b0; end
                end
                S1: begin
                    if (in == 1) begin state <= S2;    found <= 1'b0; end
                    else         begin state <= ZERO;  found <= 1'b0; end
                end
                S2: begin
                    if (in == 0) begin state <= S3;    found <= 1'b0; end
                    else         begin state <= START; found <= 1'b0; end
                end
                S3: begin
                    if (in == 1) begin state <= START; found <= 1'b1; end
                    else         begin state <= WAIT;  found <= 1'b0; end
                end
            endcase
        end
    end
endmodule

module testbench;
    reg clk, rst, in;
    wire found;
    
    pattern_detector dut(
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
        in = 0; #10;  // 0 -> Pattern 00110 detected!
        in = 0; #10;  // 0
        in = 1; #10;  // 1
        in = 1; #10;  // 1
        in = 0; #10;  // 0 -> Pattern 00110 detected again!
        in = 1; #10;  // 1
        in = 0; #10;  // 0
        in = 1; #10;  // 1
        in = 1; #10;  // 1
        in = 0; #10;  // 0
        in = 0; #10;  // 0
        in = 1; #10;  // 1
        in = 1; #10;  // 1
        in = 0; #10;  // 0 -> Pattern 00110 detected third time!
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b in=%b found=%b", $time, clk, rst, in, found);
endmodule