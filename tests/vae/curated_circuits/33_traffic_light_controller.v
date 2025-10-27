`timescale 1ns/1ps

// Traffic light controller FSM for machine learning training data
module traffic_light_controller(
    input wire clk,
    input wire rst,
    output reg [2:0] lights  // {Red, Yellow, Green}
);
    // State encoding
    reg [1:0] state;
    parameter GREEN  = 2'b00;
    parameter YELLOW = 2'b01;
    parameter RED    = 2'b10;
    
    // State transition
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= RED;
        else begin
            case(state)
                GREEN:  state <= YELLOW;
                YELLOW: state <= RED;
                RED:    state <= GREEN;
                default: state <= RED;
            endcase
        end
    end
    
    // Output logic
    always @(*) begin
        case(state)
            GREEN:  lights = 3'b001;  // Green on
            YELLOW: lights = 3'b010;  // Yellow on
            RED:    lights = 3'b100;  // Red on
            default: lights = 3'b000;
        endcase
    end
endmodule

module testbench;
    reg clk, rst;
    wire [2:0] lights;
    
    traffic_light_controller dut(
        .clk(clk),
        .rst(rst),
        .lights(lights)
    );
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        #10;
        
        // Release reset and cycle through states
        rst = 0;
        #80;  // Go through multiple cycles
        
        // Test reset
        rst = 1; #10;
        rst = 0; #30;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b rst=%b lights=%b (R=%b Y=%b G=%b)", 
                     $time, clk, rst, lights, lights[2], lights[1], lights[0]);
endmodule
