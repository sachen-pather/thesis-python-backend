`timescale 1ns/1ps

// 1:8 demultiplexer for machine learning training data
module demux_1x8(
    input wire data,
    input wire [2:0] sel,
    output reg [7:0] out
);
    // Demultiplexer - routes input to one of 8 outputs
    always @(*) begin
        out = 8'b00000000;  // Default all outputs low
        case(sel)
            3'b000: out[0] = data;
            3'b001: out[1] = data;
            3'b010: out[2] = data;
            3'b011: out[3] = data;
            3'b100: out[4] = data;
            3'b101: out[5] = data;
            3'b110: out[6] = data;
            3'b111: out[7] = data;
        endcase
    end
endmodule

module testbench;
    reg data;
    reg [2:0] sel;
    wire [7:0] out;
    
    demux_1x8 dut(
        .data(data),
        .sel(sel),
        .out(out)
    );
    
    initial begin
        // Test with data=0
        data = 0;
        sel = 3'b000; #10;
        sel = 3'b011; #10;
        sel = 3'b111; #10;
        
        // Test with data=1, routing to all outputs
        data = 1;
        sel = 3'b000; #10;
        sel = 3'b001; #10;
        sel = 3'b010; #10;
        sel = 3'b011; #10;
        sel = 3'b100; #10;
        sel = 3'b101; #10;
        sel = 3'b110; #10;
        sel = 3'b111; #10;
        
        $finish;
    end
    
    initial $monitor("Time=%0t data=%b sel=%b out=%b", $time, data, sel, out);
endmodule
