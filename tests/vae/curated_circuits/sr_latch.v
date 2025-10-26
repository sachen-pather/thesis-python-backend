`timescale 1ns/1ps

// SR latch for machine learning training data
module sr_latch(
    input wire s, r,
    output wire q, qbar
);
    // Gate-level implementation using cross-coupled NANDs
    nand (q, s, qbar);
    nand (qbar, r, q);
endmodule

module testbench;
    reg s, r;
    wire q, qbar;
    
    sr_latch dut(
        .s(s),
        .r(r),
        .q(q),
        .qbar(qbar)
    );
    
    initial begin
        // Initialize to valid state
        s = 0; r = 1; #10;  // Start with reset state
        
        // Test SR latch combinations
        s = 1; r = 1; #10;  // Hold state (both inactive)
        s = 0; r = 1; #10;  // Set (q=1)
        s = 1; r = 1; #10;  // Hold
        s = 1; r = 0; #10;  // Reset (q=0)
        s = 1; r = 1; #10;  // Hold
        s = 0; r = 1; #10;  // Set again
        s = 1; r = 0; #10;  // Reset again
        s = 0; r = 0; #10;  // Invalid state (both active)
        s = 1; r = 1; #10;  // Back to hold
        
        $finish;
    end
    
    initial $monitor("Time=%0t s=%b r=%b q=%b qbar=%b", $time, s, r, q, qbar);
endmodule