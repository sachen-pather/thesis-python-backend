`timescale 1ns/1ps

// 4-bit PISO shift register for machine learning training data
module piso_shift_register_4bit(
    input wire clk,
    input wire load,
    input wire [3:0] parallel_in,
    output wire serial_out
);
    reg [3:0] shift_reg;
    
    assign serial_out = shift_reg[3];
    
    always @(posedge clk) begin
        if (load)
            shift_reg <= parallel_in;
        else
            shift_reg <= {shift_reg[2:0], 1'b0};
    end
endmodule

module testbench;
    reg clk, load;
    reg [3:0] parallel_in;
    wire serial_out;
    
    piso_shift_register_4bit dut(
        .clk(clk),
        .load(load),
        .parallel_in(parallel_in),
        .serial_out(serial_out)
    );
    
    initial begin
        // Initialize
        clk = 0;
        load = 0;
        parallel_in = 4'b0000;
        #5;
        
        // Load pattern 1010
        load = 1; parallel_in = 4'b1010; #10;
        
        // Shift out the data
        load = 0; #10;  // MSB (1) should appear
        #10;            // Next bit (0)
        #10;            // Next bit (1)
        #10;            // LSB (0)
        
        // Load another pattern 1101
        load = 1; parallel_in = 4'b1101; #10;
        
        // Shift out this data
        load = 0; #10;  // MSB (1)
        #10;            // Next bit (1)
        #10;            // Next bit (0)
        #10;            // LSB (1)
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b load=%b parallel_in=%b serial_out=%b", 
                     $time, clk, load, parallel_in, serial_out);
endmodule
