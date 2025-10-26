/*
 * Circuit: Traffic Light (stuck state)
 * Category: State Machines - Buggy
 * Complexity: COMPLEX
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_traffic_light(input wire clk, rst, emergency, output reg [1:0] ns_light, ew_light);
localparam RED=2'b00, YELLOW=2'b01, GREEN=2'b10;
localparam S_NS_GREEN=0, S_NS_YELLOW=1, S_EW_GREEN=2, S_EW_YELLOW=3;
reg [1:0] state;
reg [3:0] counter;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= S_NS_GREEN;
        counter <= 0;
        ns_light <= GREEN;
        ew_light <= RED;
    end else if (emergency) begin
        ns_light <= RED;
        ew_light <= RED;
    end else begin
        counter <= counter + 1;
        case (state)
            S_NS_GREEN: begin
                ns_light <= GREEN; ew_light <= RED;
                // BUG: Never transitions to yellow
            end
            S_NS_YELLOW: begin
                ns_light <= YELLOW; ew_light <= RED;
                if (counter == 2) begin state <= S_EW_GREEN; counter <= 0; end
            end
            S_EW_GREEN: begin
                ns_light <= RED; ew_light <= GREEN;
                if (counter == 8) begin state <= S_EW_YELLOW; counter <= 0; end
            end
            S_EW_YELLOW: begin
                ns_light <= RED; ew_light <= YELLOW;
                if (counter == 2) begin state <= S_NS_GREEN; counter <= 0; end
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst, emergency; wire [1:0] ns_light, ew_light;
bad_traffic_light dut(.clk(clk), .rst(rst), .emergency(emergency), .ns_light(ns_light), .ew_light(ew_light));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; emergency=0; #10; rst=0;
    #200; emergency=1; #20; emergency=0; #100; $finish;
end
initial $monitor("Time=%0t state=%d ns=%b ew=%b emerg=%b", $time, dut.state, ns_light, ew_light, emergency);
endmodule