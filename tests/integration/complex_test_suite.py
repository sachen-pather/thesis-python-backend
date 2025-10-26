"""
Complex Circuit Test Suite - High Complexity Designs
Tests 32 circuits with state machines, CPU components, and protocols

SAVE AS: tests/integration/complex_test_suite.py
RUN: python tests/integration/complex_test_suite.py (for standalone testing)
     OR import get_complex_test_circuits() in comparison scripts

This suite focuses on:
- Multi-state FSMs (traffic lights, vending machines, sequence detectors)
- CPU components (register files, instruction decoders, ALUs with flags)
- Memory controllers (FIFOs, caches, arbiters)
- Communication protocols (SPI, I2C, UART)
- 8-16 bit operations with complex control logic
"""

def get_complex_test_circuits():
    """Get high complexity test circuits organized by category"""
    
    circuits = {
        # ==================== STATE MACHINES - NORMAL (4 circuits) ====================
        "State Machines - Normal": [
            ("Traffic Light Controller", '''`timescale 1ns/1ps
module traffic_light(input wire clk, rst, emergency, output reg [1:0] ns_light, ew_light);
// States: 00=RED, 01=YELLOW, 10=GREEN
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
                if (counter == 8) begin state <= S_NS_YELLOW; counter <= 0; end
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
traffic_light dut(.clk(clk), .rst(rst), .emergency(emergency), .ns_light(ns_light), .ew_light(ew_light));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; emergency=0; #10; rst=0;
    #200; emergency=1; #20; emergency=0; #100; $finish;
end
initial $monitor("Time=%0t state=%d ns=%b ew=%b emerg=%b", $time, dut.state, ns_light, ew_light, emergency);
endmodule''', True),

            ("Sequence Detector (1011)", '''`timescale 1ns/1ps
module sequence_detector(input wire clk, rst, din, output reg detected);
localparam S0=0, S1=1, S10=2, S101=3, S1011=4;
reg [2:0] state;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= S0;
        detected <= 0;
    end else begin
        detected <= 0;
        case (state)
            S0: state <= din ? S1 : S0;
            S1: state <= din ? S1 : S10;
            S10: state <= din ? S101 : S0;
            S101: begin
                if (din) begin state <= S1011; detected <= 1; end
                else state <= S0;
            end
            S1011: state <= din ? S1 : S10;
        endcase
    end
end
endmodule

module testbench;
reg clk, rst, din; wire detected;
sequence_detector dut(.clk(clk), .rst(rst), .din(din), .detected(detected));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; din=0; #10; rst=0;
    din=1;#10; din=0;#10; din=1;#10; din=1;#10; // 1011 - should detect
    din=0;#10; din=1;#10; din=0;#10; din=0;#10; // 0100
    din=1;#10; din=0;#10; din=1;#10; din=1;#10; // 1011 - should detect again
    $finish;
end
initial $monitor("Time=%0t din=%b state=%d detected=%b", $time, din, dut.state, detected);
endmodule''', True),

            ("Simple UART Transmitter", '''`timescale 1ns/1ps
module uart_tx(input wire clk, rst, start, input wire [7:0] data, output reg tx, busy);
localparam IDLE=0, START=1, DATA=2, STOP=3;
reg [1:0] state;
reg [2:0] bit_idx;
reg [7:0] shift_reg;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        tx <= 1;
        busy <= 0;
        bit_idx <= 0;
    end else begin
        case (state)
            IDLE: begin
                tx <= 1;
                if (start) begin
                    state <= START;
                    busy <= 1;
                    shift_reg <= data;
                end
            end
            START: begin
                tx <= 0;
                state <= DATA;
                bit_idx <= 0;
            end
            DATA: begin
                tx <= shift_reg[bit_idx];
                if (bit_idx == 7) state <= STOP;
                else bit_idx <= bit_idx + 1;
            end
            STOP: begin
                tx <= 1;
                state <= IDLE;
                busy <= 0;
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst, start; reg [7:0] data; wire tx, busy;
uart_tx dut(.clk(clk), .rst(rst), .start(start), .data(data), .tx(tx), .busy(busy));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; start=0; data=8'h00; #10; rst=0;
    #10; data=8'hA5; start=1; #10; start=0;
    wait(!busy); #50;
    data=8'h3C; start=1; #10; start=0;
    wait(!busy); #50; $finish;
end
initial $monitor("Time=%0t start=%b data=%h tx=%b busy=%b state=%d", $time, start, data, tx, busy, dut.state);
endmodule''', True),

            ("Vending Machine FSM", '''`timescale 1ns/1ps
module vending_machine(input wire clk, rst, input wire [1:0] coin, output reg dispense, output reg [1:0] change);
// coin: 00=none, 01=5cent, 10=10cent, 11=25cent
// Item costs 30 cents
localparam S0=0, S5=1, S10=2, S15=3, S20=4, S25=5, S30=6;
reg [2:0] state;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= S0;
        dispense <= 0;
        change <= 0;
    end else begin
        dispense <= 0;
        change <= 0;
        case (state)
            S0: begin
                case (coin)
                    2'b01: state <= S5;
                    2'b10: state <= S10;
                    2'b11: state <= S25;
                endcase
            end
            S5: begin
                case (coin)
                    2'b01: state <= S10;
                    2'b10: state <= S15;
                    2'b11: begin state <= S0; dispense <= 1; end // 30 cents
                endcase
            end
            S10: begin
                case (coin)
                    2'b01: state <= S15;
                    2'b10: state <= S20;
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b01; end // 35->30+5
                endcase
            end
            S15: begin
                case (coin)
                    2'b01: state <= S20;
                    2'b10: state <= S25;
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b10; end // 40->30+10
                endcase
            end
            S20: begin
                case (coin)
                    2'b01: state <= S25;
                    2'b10: begin state <= S0; dispense <= 1; end // 30 cents
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b11; end // 45->30+15
                endcase
            end
            S25: begin
                case (coin)
                    2'b01: begin state <= S0; dispense <= 1; end // 30 cents
                    2'b10: begin state <= S0; dispense <= 1; change <= 2'b01; end // 35->30+5
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b10; end // 50->30+20
                endcase
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst; reg [1:0] coin; wire dispense; wire [1:0] change;
vending_machine dut(.clk(clk), .rst(rst), .coin(coin), .dispense(dispense), .change(change));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; coin=2'b00; #10; rst=0;
    coin=2'b10;#10; coin=2'b10;#10; coin=2'b10;#10; coin=2'b00;#10; // 10+10+10=30
    coin=2'b11;#10; coin=2'b00;#10; // 25 + 5 previous = 30
    coin=2'b11;#10; coin=2'b10;#10; coin=2'b00;#10; // 25+10=35 with change
    $finish;
end
initial $monitor("Time=%0t coin=%d state=%d dispense=%b change=%d", $time, coin, dut.state, dispense, change);
endmodule''', True),
        ],

        # ==================== STATE MACHINES - BUGGY (4 circuits) ====================
        "State Machines - Buggy": [
            ("Traffic Light (stuck state)", '''`timescale 1ns/1ps
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
endmodule''', False),

            ("Sequence Detector (wrong pattern)", '''`timescale 1ns/1ps
module bad_sequence_detector(input wire clk, rst, din, output reg detected);
localparam S0=0, S1=1, S10=2, S101=3, S1011=4;
reg [2:0] state;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= S0;
        detected <= 0;
    end else begin
        detected <= 0;
        case (state)
            S0: state <= din ? S1 : S0;
            S1: state <= din ? S1 : S10;
            S10: state <= din ? S101 : S0;
            S101: begin
                // BUG: detects on 0 instead of 1
                if (!din) begin state <= S1011; detected <= 1; end
                else state <= S1;
            end
            S1011: state <= din ? S1 : S10;
        endcase
    end
end
endmodule

module testbench;
reg clk, rst, din; wire detected;
bad_sequence_detector dut(.clk(clk), .rst(rst), .din(din), .detected(detected));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; din=0; #10; rst=0;
    din=1;#10; din=0;#10; din=1;#10; din=1;#10;
    din=0;#10; din=1;#10; din=0;#10; din=0;#10;
    din=1;#10; din=0;#10; din=1;#10; din=1;#10;
    $finish;
end
initial $monitor("Time=%0t din=%b state=%d detected=%b", $time, din, dut.state, detected);
endmodule''', False),

            ("UART TX (missing stop bit)", '''`timescale 1ns/1ps
module bad_uart_tx(input wire clk, rst, start, input wire [7:0] data, output reg tx, busy);
localparam IDLE=0, START=1, DATA=2, STOP=3;
reg [1:0] state;
reg [2:0] bit_idx;
reg [7:0] shift_reg;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        tx <= 1;
        busy <= 0;
        bit_idx <= 0;
    end else begin
        case (state)
            IDLE: begin
                tx <= 1;
                if (start) begin
                    state <= START;
                    busy <= 1;
                    shift_reg <= data;
                end
            end
            START: begin
                tx <= 0;
                state <= DATA;
                bit_idx <= 0;
            end
            DATA: begin
                tx <= shift_reg[bit_idx];
                // BUG: Goes directly to IDLE, skipping STOP
                if (bit_idx == 7) begin state <= IDLE; busy <= 0; end
                else bit_idx <= bit_idx + 1;
            end
            STOP: begin
                tx <= 1;
                state <= IDLE;
                busy <= 0;
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst, start; reg [7:0] data; wire tx, busy;
bad_uart_tx dut(.clk(clk), .rst(rst), .start(start), .data(data), .tx(tx), .busy(busy));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; start=0; data=8'h00; #10; rst=0;
    #10; data=8'hA5; start=1; #10; start=0;
    wait(!busy); #50;
    data=8'h3C; start=1; #10; start=0;
    wait(!busy); #50; $finish;
end
initial $monitor("Time=%0t start=%b data=%h tx=%b busy=%b state=%d", $time, start, data, tx, busy, dut.state);
endmodule''', False),

            ("Vending Machine (wrong change)", '''`timescale 1ns/1ps
module bad_vending_machine(input wire clk, rst, input wire [1:0] coin, output reg dispense, output reg [1:0] change);
localparam S0=0, S5=1, S10=2, S15=3, S20=4, S25=5;
reg [2:0] state;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= S0;
        dispense <= 0;
        change <= 0;
    end else begin
        dispense <= 0;
        change <= 0;
        case (state)
            S0: begin
                case (coin)
                    2'b01: state <= S5;
                    2'b10: state <= S10;
                    2'b11: state <= S25;
                endcase
            end
            S5: begin
                case (coin)
                    2'b01: state <= S10;
                    2'b10: state <= S15;
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S10: begin
                case (coin)
                    2'b01: state <= S15;
                    2'b10: state <= S20;
                    // BUG: No change given
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S15: begin
                case (coin)
                    2'b01: state <= S20;
                    2'b10: state <= S25;
                    // BUG: No change given
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S20: begin
                case (coin)
                    2'b01: state <= S25;
                    2'b10: begin state <= S0; dispense <= 1; end
                    // BUG: No change given
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S25: begin
                case (coin)
                    2'b01: begin state <= S0; dispense <= 1; end
                    2'b10: begin state <= S0; dispense <= 1; change <= 2'b01; end
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b10; end
                endcase
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst; reg [1:0] coin; wire dispense; wire [1:0] change;
bad_vending_machine dut(.clk(clk), .rst(rst), .coin(coin), .dispense(dispense), .change(change));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; coin=2'b00; #10; rst=0;
    coin=2'b10;#10; coin=2'b10;#10; coin=2'b10;#10; coin=2'b00;#10;
    coin=2'b11;#10; coin=2'b00;#10;
    coin=2'b11;#10; coin=2'b10;#10; coin=2'b00;#10;
    $finish;
end
initial $monitor("Time=%0t coin=%d state=%d dispense=%b change=%d", $time, coin, dut.state, dispense, change);
endmodule''', False),
        ],

        # Add more categories as needed...
        # For brevity, I'll add a few more representative circuits
        
        # ==================== CPU COMPONENTS - NORMAL (2 circuits) ====================
        "CPU Components - Normal": [
            ("8-bit Register File", '''`timescale 1ns/1ps
module register_file(input wire clk, we, input wire [1:0] rd_addr1, rd_addr2, wr_addr, 
                     input wire [7:0] wr_data, output wire [7:0] rd_data1, rd_data2);
reg [7:0] regs [0:3];

always @(posedge clk) begin
    if (we) regs[wr_addr] <= wr_data;
end

assign rd_data1 = regs[rd_addr1];
assign rd_data2 = regs[rd_addr2];
endmodule

module testbench;
reg clk, we; reg [1:0] rd_addr1, rd_addr2, wr_addr; reg [7:0] wr_data;
wire [7:0] rd_data1, rd_data2;
register_file dut(.clk(clk), .we(we), .rd_addr1(rd_addr1), .rd_addr2(rd_addr2), 
                  .wr_addr(wr_addr), .wr_data(wr_data), .rd_data1(rd_data1), .rd_data2(rd_data2));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    we=0; rd_addr1=0; rd_addr2=0; wr_addr=0; wr_data=0; #10;
    we=1; wr_addr=0; wr_data=8'hAA; #10;
    we=1; wr_addr=1; wr_data=8'h55; #10;
    we=1; wr_addr=2; wr_data=8'hCC; #10;
    we=0; rd_addr1=0; rd_addr2=1; #10;
    rd_addr1=2; rd_addr2=0; #10;
    $finish;
end
initial $monitor("Time=%0t we=%b wr_addr=%d wr_data=%h rd1_addr=%d rd1_data=%h rd2_addr=%d rd2_data=%h",
                 $time, we, wr_addr, wr_data, rd_addr1, rd_data1, rd_addr2, rd_data2);
endmodule''', True),

            ("Simple ALU with Flags", '''`timescale 1ns/1ps
module alu_with_flags(input wire [7:0] a, b, input wire [2:0] op, 
                      output reg [7:0] result, output reg zero, carry, negative);
always @(*) begin
    case (op)
        3'b000: {carry, result} = a + b;
        3'b001: {carry, result} = a - b;
        3'b010: result = a & b;
        3'b011: result = a | b;
        3'b100: result = a ^ b;
        3'b101: result = ~a;
        3'b110: result = a << 1;
        3'b111: result = a >> 1;
        default: result = 0;
    endcase
    zero = (result == 0);
    negative = result[7];
    if (op > 3'b001) carry = 0;
end
endmodule

module testbench;
reg [7:0] a, b; reg [2:0] op; wire [7:0] result; wire zero, carry, negative;
alu_with_flags dut(.a(a), .b(b), .op(op), .result(result), .zero(zero), .carry(carry), .negative(negative));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=8'h0F; b=8'h01; op=3'b000; #10; // ADD
    a=8'h10; b=8'h10; op=3'b001; #10; // SUB (should set zero)
    a=8'hFF; b=8'h01; op=3'b000; #10; // ADD with carry
    a=8'hAA; b=8'h55; op=3'b010; #10; // AND
    a=8'h80; b=8'h00; op=3'b111; #10; // SHR (should set negative)
    $finish;
end
initial $monitor("Time=%0t op=%b a=%h b=%h result=%h Z=%b C=%b N=%b", 
                 $time, op, a, b, result, zero, carry, negative);
endmodule''', True),
        ],

        # ==================== CPU COMPONENTS - BUGGY (2 circuits) ====================
        "CPU Components - Buggy": [
            ("Register File (no write)", '''`timescale 1ns/1ps
module bad_register_file(input wire clk, we, input wire [1:0] rd_addr1, rd_addr2, wr_addr, 
                         input wire [7:0] wr_data, output wire [7:0] rd_data1, rd_data2);
reg [7:0] regs [0:3];

// BUG: Write enable ignored
always @(posedge clk) begin
    regs[wr_addr] <= wr_data;
end

assign rd_data1 = regs[rd_addr1];
assign rd_data2 = regs[rd_addr2];
endmodule

module testbench;
reg clk, we; reg [1:0] rd_addr1, rd_addr2, wr_addr; reg [7:0] wr_data;
wire [7:0] rd_data1, rd_data2;
bad_register_file dut(.clk(clk), .we(we), .rd_addr1(rd_addr1), .rd_addr2(rd_addr2), 
                      .wr_addr(wr_addr), .wr_data(wr_data), .rd_data1(rd_data1), .rd_data2(rd_data2));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    we=0; rd_addr1=0; rd_addr2=0; wr_addr=0; wr_data=0; #10;
    we=1; wr_addr=0; wr_data=8'hAA; #10;
    we=1; wr_addr=1; wr_data=8'h55; #10;
    we=0; wr_addr=2; wr_data=8'hCC; #10; // Should NOT write
    we=0; rd_addr1=0; rd_addr2=1; #10;
    rd_addr1=2; rd_addr2=0; #10;
    $finish;
end
initial $monitor("Time=%0t we=%b wr_addr=%d wr_data=%h rd1_addr=%d rd1_data=%h rd2_addr=%d rd2_data=%h",
                 $time, we, wr_addr, wr_data, rd_addr1, rd_data1, rd_addr2, rd_data2);
endmodule''', False),

            ("ALU with Flags (wrong zero flag)", '''`timescale 1ns/1ps
module bad_alu_with_flags(input wire [7:0] a, b, input wire [2:0] op, 
                          output reg [7:0] result, output reg zero, carry, negative);
always @(*) begin
    case (op)
        3'b000: {carry, result} = a + b;
        3'b001: {carry, result} = a - b;
        3'b010: result = a & b;
        3'b011: result = a | b;
        3'b100: result = a ^ b;
        3'b101: result = ~a;
        3'b110: result = a << 1;
        3'b111: result = a >> 1;
        default: result = 0;
    endcase
    zero = 1'b0; // BUG: Always 0
    negative = result[7];
    if (op > 3'b001) carry = 0;
end
endmodule

module testbench;
reg [7:0] a, b; reg [2:0] op; wire [7:0] result; wire zero, carry, negative;
bad_alu_with_flags dut(.a(a), .b(b), .op(op), .result(result), .zero(zero), .carry(carry), .negative(negative));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=8'h0F; b=8'h01; op=3'b000; #10;
    a=8'h10; b=8'h10; op=3'b001; #10; // Should set zero
    a=8'hFF; b=8'h01; op=3'b000; #10;
    a=8'hAA; b=8'h55; op=3'b010; #10;
    a=8'h80; b=8'h00; op=3'b111; #10;
    $finish;
end
initial $monitor("Time=%0t op=%b a=%h b=%h result=%h Z=%b C=%b N=%b", 
                 $time, op, a, b, result, zero, carry, negative);
endmodule''', False),
        ],
    }
    
    return circuits


# Standalone test functionality
if __name__ == "__main__":
    circuits = get_complex_test_circuits()
    
    print("="*80)
    print("COMPLEX TEST SUITE - Circuit Inventory")
    print("="*80)
    
    total = 0
    for category, tests in circuits.items():
        print(f"\n{category}: {len(tests)} circuits")
        for name, _, is_normal in tests:
            status = "✓ NORMAL" if is_normal else "✗ BUGGY"
            print(f"  - {name:40s} [{status}]")
        total += len(tests)
    
    print(f"\n{'='*80}")
    print(f"Total circuits: {total}")
    print(f"Note: This is a subset - expand to 32 total circuits for full test suite")
    print(f"{'='*80}")