`timescale 1ns/1ps

// 4-bit SISO shift register for machine learning training data
module siso_shift_register_4bit(
    input wire clk,
    input wire rst,
    input wire si,
    output wire so
);
    reg [3:0] shift_reg;
    
    assign so = shift_reg[3];
    
    always @(posedge clk or posedge rst) begin
        if (rst)
            shift_reg <= 4'b0000;
        else
            shift_reg <= {shift_reg[2:0], si};
    end
endmodule

module testbench;
    reg clk, rst, si;
    wire so;
    
    siso_shift_register_4bit dut(
        .clk(clk),
        .rst(rst),
        .si(si),
        .so(so)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        si = 0;
        #10;
        
        rst = 0;
        
        // Shift in pattern 1011
        si = 1; #10;  // Shift in 1
        si = 0; #10;  // Shift in 0
        si = 1; #10;  // Shift in 1
        si = 1; #10;  // Shift in 1
        
        // Continue shifting to see data come out
        si = 0; #10;  // Now 1 should appear at output
        si = 0; #10;  // Now 0 should appear at output
        si = 0; #10;  // Now 1 should appear at output
        si = 0; #10;  // Now 1 should appear at output
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b si=%b so=%b", 
                     $time, clk, rst, si, so);
endmodule
