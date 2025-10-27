`timescale 1ns/1ps

// 8-bit register with enable for machine learning training data
module register_8bit_en(
    input wire clk,
    input wire rst,
    input wire en,
    input wire [7:0] data_in,
    output reg [7:0] data_out
);
    // 8-bit register with enable control
    always @(posedge clk or posedge rst) begin
        if (rst)
            data_out <= 8'b00000000;
        else if (en)
            data_out <= data_in;
    end
endmodule

module testbench;
    reg clk, rst, en;
    reg [7:0] data_in;
    wire [7:0] data_out;
    
    register_8bit_en dut(
        .clk(clk),
        .rst(rst),
        .en(en),
        .data_in(data_in),
        .data_out(data_out)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        en = 0;
        data_in = 8'b00000000;
        #10;
        
        // Release reset
        rst = 0; #10;
        
        // Test load with enable
        en = 1; data_in = 8'b10101010; #10;
        en = 0; data_in = 8'b11111111; #10;  // Should hold previous
        en = 1; data_in = 8'b11001100; #10;
        en = 0; #10;
        en = 1; data_in = 8'b00110011; #10;
        
        // Test reset
        rst = 1; #10;
        rst = 0; #10;
        
        en = 1; data_in = 8'b11110000; #10;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b en=%b data_in=%b data_out=%b", 
                     $time, clk, rst, en, data_in, data_out);
endmodule
