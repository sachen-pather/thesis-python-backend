`timescale 1ns/1ps

// SR flip-flop for machine learning training data
module sr_flip_flop(
    input wire clk,
    input wire s, r,
    output reg q,
    output wire qbar
);
    // Behavioral implementation
    assign qbar = ~q;
    
    always @(posedge clk) begin
        case({s,r})
            2'b00: q <= q;      // Hold
            2'b01: q <= 1'b0;   // Reset
            2'b10: q <= 1'b1;   // Set
            2'b11: q <= 1'bx;   // Invalid state
        endcase
    end
endmodule

module testbench;
    reg clk, s, r;
    wire q, qbar;
    
    sr_flip_flop dut(
        .clk(clk),
        .s(s),
        .r(r),
        .q(q),
        .qbar(qbar)
    );
    
    initial begin
        // Initialize
        clk = 0;
        s = 0; r = 0;
        #5;
        
        // Test SR combinations (avoid s=1,r=1 invalid state mostly)
        s = 0; r = 0; #10;  // Hold state
        s = 1; r = 0; #10;  // Set (q=1)
        s = 0; r = 0; #10;  // Hold
        s = 0; r = 1; #10;  // Reset (q=0)
        s = 0; r = 0; #10;  // Hold
        s = 1; r = 0; #10;  // Set again
        s = 0; r = 1; #10;  // Reset again
        s = 1; r = 1; #10;  // Invalid state (demonstrate)
        s = 0; r = 0; #10;  // Back to hold
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b s=%b r=%b q=%b qbar=%b", $time, clk, s, r, q, qbar);
endmodule