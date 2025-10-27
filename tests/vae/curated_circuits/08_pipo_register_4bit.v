`timescale 1ns/1ps

// 4-bit PIPO register for machine learning training data
module pipo_register_4bit(
    input wire clk,
    input wire load,
    input wire [3:0] d,
    output reg [3:0] q
);
    // Synchronous load on clock edge
    always @(posedge clk) begin
        if (load)
            q <= d;
    end
endmodule

module testbench;
    reg clk, load;
    reg [3:0] d;
    wire [3:0] q;
    
    pipo_register_4bit dut(
        .clk(clk),
        .load(load),
        .d(d),
        .q(q)
    );
    
    initial begin
        // Initialize
        clk = 0;
        load = 0;
        d = 4'b0000;
        #5;
        
        // Test load operations
        load = 1; d = 4'b1010; #10;
        load = 0; d = 4'b1111; #10;  // Should hold previous value
        load = 1; d = 4'b0101; #10;
        load = 0; #10;
        load = 1; d = 4'b1100; #10;
        load = 0; #10;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b load=%b d=%b q=%b", $time, clk, load, d, q);
endmodule
