`timescale 1ns/1ps

// Vending machine FSM (costs 15 cents, accepts 5 and 10 cent coins)
// for machine learning training data
module vending_machine(
    input wire clk,
    input wire rst,
    input wire nickel,   // 5 cent coin
    input wire dime,     // 10 cent coin
    output reg dispense,
    output reg [3:0] total
);
    // State encoding (total cents deposited)
    reg [1:0] state;
    parameter S0  = 2'b00;  // 0 cents
    parameter S5  = 2'b01;  // 5 cents
    parameter S10 = 2'b10;  // 10 cents
    parameter S15 = 2'b11;  // 15 cents (dispense)
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S0;
            total <= 4'd0;
            dispense <= 0;
        end else begin
            dispense <= 0;  // Default
            case(state)
                S0: begin
                    if (nickel) begin
                        state <= S5;
                        total <= 4'd5;
                    end else if (dime) begin
                        state <= S10;
                        total <= 4'd10;
                    end
                end
                S5: begin
                    if (nickel) begin
                        state <= S10;
                        total <= 4'd10;
                    end else if (dime) begin
                        state <= S15;
                        total <= 4'd15;
                        dispense <= 1;
                    end
                end
                S10: begin
                    if (nickel || dime) begin
                        state <= S15;
                        total <= 4'd15;
                        dispense <= 1;
                    end
                end
                S15: begin
                    state <= S0;  // Return to start after dispensing
                    total <= 4'd0;
                end
            endcase
        end
    end
endmodule

module testbench;
    reg clk, rst, nickel, dime;
    wire dispense;
    wire [3:0] total;
    
    vending_machine dut(
        .clk(clk),
        .rst(rst),
        .nickel(nickel),
        .dime(dime),
        .dispense(dispense),
        .total(total)
    );
    
    initial begin
        clk = 0;
        rst = 1;
        nickel = 0;
        dime = 0;
        #10;
        
        rst = 0;
        
        // Test case 1: Three nickels (5+5+5=15)
        nickel = 1; dime = 0; #10;
        nickel = 0; #10;
        nickel = 1; #10;
        nickel = 0; #10;
        nickel = 1; #10;
        nickel = 0; #20;  // Wait for dispense and reset
        
        // Test case 2: One dime + one nickel (10+5=15)
        dime = 1; nickel = 0; #10;
        dime = 0; #10;
        nickel = 1; #10;
        nickel = 0; #20;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t rst=%b nickel=%b dime=%b total=%d dispense=%b", 
                     $time, rst, nickel, dime, total, dispense);
endmodule
