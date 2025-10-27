`timescale 1ns/1ps

// D latch for machine learning training data
module d_latch(
    input wire en,
    input wire d,
    output reg q,
    output wire qbar
);
    // Level-sensitive latch
    assign qbar = ~q;
    
    always @(*) begin
        if (en)
            q = d;
    end
endmodule

module testbench;
    reg en, d;
    wire q, qbar;
    
    d_latch dut(
        .en(en),
        .d(d),
        .q(q),
        .qbar(qbar)
    );
    
    initial begin
        // Initialize
        en = 0; d = 0; #10;
        
        // Test transparent mode (en=1)
        en = 1; d = 1; #10;  // q follows d
        d = 0; #10;          // q follows d
        d = 1; #10;          // q follows d
        d = 0; #10;          // q follows d
        
        // Test latch mode (en=0)
        en = 0; d = 1; #10;  // q holds previous value
        d = 0; #10;          // q still holds
        d = 1; #10;          // q still holds
        
        // Back to transparent
        en = 1; d = 0; #10;  // q follows d
        d = 1; #10;          // q follows d
        
        $finish;
    end
    
    initial $monitor("Time=%0t en=%b d=%b q=%b qbar=%b", $time, en, d, q, qbar);
endmodule
