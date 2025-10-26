`timescale 1ns/1ps

// 8-bit register with load and clear for machine learning training data
module register_8bit(
    input wire [7:0] data_in,
    input wire clk,
    input wire ld_bar,
    input wire clr_bar,
    output reg [7:0] data_out
);
    // 8-bit register implementation
    always @(posedge clk) begin
        if (~clr_bar) begin
            data_out <= 8'b00000000;
        end else if (~ld_bar) begin
            data_out <= data_in;
        end
        // If both ld_bar and clr_bar are high, hold current value
    end
endmodule

module testbench;
    reg [7:0] data_in;
    reg clk, ld_bar, clr_bar;
    wire [7:0] data_out;
    
    register_8bit dut(
        .data_in(data_in),
        .clk(clk),
        .ld_bar(ld_bar),
        .clr_bar(clr_bar),
        .data_out(data_out)
    );
    
    initial begin
        // Initialize
        clk = 0;
        ld_bar = 1;
        clr_bar = 1;
        data_in = 8'b00000000;
        #10;
        
        // Test clear function
        clr_bar = 0; #10;
        clr_bar = 1; #10;
        
        // Test load function - load different values
        data_in = 8'b10101010;  // 170 decimal
        ld_bar = 0; #10;
        ld_bar = 1; #20;  // Hold value
        
        data_in = 8'b11001100;  // 204 decimal
        ld_bar = 0; #10;
        ld_bar = 1; #20;  // Hold value
        
        data_in = 8'b11110000;  // 240 decimal  
        ld_bar = 0; #10;
        ld_bar = 1; #20;  // Hold value
        
        // Test clear while data is present
        clr_bar = 0; #10;
        clr_bar = 1; #10;
        
        // Load final value
        data_in = 8'b01010101;  // 85 decimal
        ld_bar = 0; #10;
        ld_bar = 1; #20;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b ld_bar=%b clr_bar=%b data_in=%b data_out=%b (%d)", 
                     $time, clk, ld_bar, clr_bar, data_in, data_out, data_out);
endmodule