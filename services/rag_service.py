# services/rag_service.py - MVP RAG Implementation
import os
import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class VerilogRAGService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
        self.knowledge_base = []
        self.embeddings = None
        self.rag_data_file = "data/verilog_rag_data.pkl"
        
        # Initialize with basic examples
        self._initialize_knowledge_base()
        self._load_or_create_embeddings()
    
    def _initialize_knowledge_base(self):
        """Initialize with basic Verilog examples"""
        self.knowledge_base = [
            {
    "description": "8-bit up counter with synchronous load and enable",
    "mermaid": """graph TD
    CLK[Clock] --> COUNTER[8-bit Counter]
    RST[Reset] --> COUNTER
    LOAD[Load] --> COUNTER
    EN[Enable] --> COUNTER
    DATA[Data In 8-bit] --> COUNTER
    COUNTER --> COUNT[Count Out 8-bit]""",
    "verilog": """`timescale 1ns/1ps

module counter_8bit_load(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire load,
    input wire [7:0] data_in,
    output reg [7:0] count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 8'b0;
    else if (load)
        count <= data_in;
    else if (enable)
        count <= count + 1'b1;
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    reg load;
    reg [7:0] data_in;
    wire [7:0] count;
    
    counter_8bit_load dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .load(load),
        .data_in(data_in),
        .count(count)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        load = 0;
        data_in = 8'h00;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #100;
        
        load = 1;
        data_in = 8'hA5;
        #10 load = 0;
        
        #100;
        
        enable = 0;
        #50;
        
        enable = 1;
        #100;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "4-bit down counter with terminal count output",
    "mermaid": """graph TD
    CLK[Clock] --> COUNTER[4-bit Down Counter]
    RST[Reset] --> COUNTER
    EN[Enable] --> COUNTER
    COUNTER --> COUNT[Count Out 4-bit]
    COUNTER --> TC[Terminal Count]""",
    "verilog": """`timescale 1ns/1ps

module down_counter_4bit(
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [3:0] count,
    output wire tc
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 4'b1111;
    else if (enable)
        count <= count - 1'b1;
end

assign tc = (count == 4'b0000) ? 1'b1 : 1'b0;

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    wire [3:0] count;
    wire tc;
    
    down_counter_4bit dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .count(count),
        .tc(tc)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #200;
        
        enable = 0;
        #50;
        
        enable = 1;
        #100;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "BCD decade counter with carry out for cascading",
    "mermaid": """graph TD
    CLK[Clock] --> BCD[BCD Counter 0-9]
    RST[Reset] --> BCD
    EN[Enable] --> BCD
    BCD --> COUNT[BCD Out 4-bit]
    BCD --> CO[Carry Out]""",
    "verilog": """`timescale 1ns/1ps

module bcd_counter(
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [3:0] bcd_count,
    output reg carry_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bcd_count <= 4'b0000;
        carry_out <= 1'b0;
    end
    else if (enable) begin
        if (bcd_count == 4'd9) begin
            bcd_count <= 4'b0000;
            carry_out <= 1'b1;
        end
        else begin
            bcd_count <= bcd_count + 1'b1;
            carry_out <= 1'b0;
        end
    end
    else begin
        carry_out <= 1'b0;
    end
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    wire [3:0] bcd_count;
    wire carry_out;
    
    bcd_counter dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .bcd_count(bcd_count),
        .carry_out(carry_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #250;
        
        enable = 0;
        #30;
        
        enable = 1;
        #150;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},
{
    "description": "16-bit ripple counter with asynchronous reset",
    "mermaid": """graph TD
    CLK[Clock] --> COUNTER[16-bit Ripple Counter]
    RST[Async Reset] --> COUNTER
    COUNTER --> COUNT[Count Out 16-bit]""",
    "verilog": """`timescale 1ns/1ps

module ripple_counter_16bit(
    input wire clk,
    input wire rst_n,
    output reg [15:0] count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 16'b0;
    else
        count <= count + 1'b1;
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    wire [15:0] count;
    
    ripple_counter_16bit dut(
        .clk(clk),
        .rst_n(rst_n),
        .count(count)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        
        #25 rst_n = 1;
        
        #500;
        
        rst_n = 0;
        #20;
        rst_n = 1;
        
        #300;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "Modulo-N counter with programmable limit",
    "mermaid": """graph TD
    CLK[Clock] --> COUNTER[Modulo-N Counter]
    RST[Reset] --> COUNTER
    EN[Enable] --> COUNTER
    LIMIT[Limit 8-bit] --> COUNTER
    COUNTER --> COUNT[Count Out 8-bit]
    COUNTER --> WRAP[Wrap Flag]""",
    "verilog": """`timescale 1ns/1ps

module modulo_n_counter(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [7:0] limit,
    output reg [7:0] count,
    output reg wrap_flag
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        count <= 8'b0;
        wrap_flag <= 1'b0;
    end
    else if (enable) begin
        if (count >= limit - 1) begin
            count <= 8'b0;
            wrap_flag <= 1'b1;
        end
        else begin
            count <= count + 1'b1;
            wrap_flag <= 1'b0;
        end
    end
    else begin
        wrap_flag <= 1'b0;
    end
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    reg [7:0] limit;
    wire [7:0] count;
    wire wrap_flag;
    
    modulo_n_counter dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .limit(limit),
        .count(count),
        .wrap_flag(wrap_flag)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        limit = 8'd10;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #150;
        
        limit = 8'd5;
        #80;
        
        enable = 0;
        #30;
        enable = 1;
        
        #100;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "Gray code counter for 4-bit values",
    "mermaid": """graph TD
    CLK[Clock] --> COUNTER[Gray Code Counter]
    RST[Reset] --> COUNTER
    EN[Enable] --> COUNTER
    COUNTER --> GRAY[Gray Code 4-bit]
    COUNTER --> BIN[Binary 4-bit]""",
    "verilog": """`timescale 1ns/1ps

module gray_counter_4bit(
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [3:0] gray_count,
    output wire [3:0] binary_count
);

reg [3:0] binary_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        binary_reg <= 4'b0;
    else if (enable)
        binary_reg <= binary_reg + 1'b1;
end

always @(*) begin
    gray_count = binary_reg ^ (binary_reg >> 1);
end

assign binary_count = binary_reg;

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    wire [3:0] gray_count;
    wire [3:0] binary_count;
    
    gray_counter_4bit dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .gray_count(gray_count),
        .binary_count(binary_count)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #200;
        
        enable = 0;
        #40;
        enable = 1;
        
        #150;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "Up-down counter with direction control and overflow flags",
    "mermaid": """graph TD
    CLK[Clock] --> COUNTER[Up-Down Counter]
    RST[Reset] --> COUNTER
    EN[Enable] --> COUNTER
    DIR[Direction] --> COUNTER
    COUNTER --> COUNT[Count 8-bit]
    COUNTER --> OVF[Overflow]
    COUNTER --> UNF[Underflow]""",
    "verilog": """`timescale 1ns/1ps

module updown_counter(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire direction,
    output reg [7:0] count,
    output reg overflow,
    output reg underflow
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        count <= 8'b0;
        overflow <= 1'b0;
        underflow <= 1'b0;
    end
    else if (enable) begin
        if (direction) begin
            if (count == 8'hFF) begin
                count <= 8'h00;
                overflow <= 1'b1;
                underflow <= 1'b0;
            end
            else begin
                count <= count + 1'b1;
                overflow <= 1'b0;
                underflow <= 1'b0;
            end
        end
        else begin
            if (count == 8'h00) begin
                count <= 8'hFF;
                underflow <= 1'b1;
                overflow <= 1'b0;
            end
            else begin
                count <= count - 1'b1;
                overflow <= 1'b0;
                underflow <= 1'b0;
            end
        end
    end
    else begin
        overflow <= 1'b0;
        underflow <= 1'b0;
    end
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    reg direction;
    wire [7:0] count;
    wire overflow;
    wire underflow;
    
    updown_counter dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .direction(direction),
        .count(count),
        .overflow(overflow),
        .underflow(underflow)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        direction = 1;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #100;
        
        direction = 0;
        #100;
        
        direction = 1;
        #80;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "Ring counter with 8 stages and one-hot output",
    "mermaid": """graph TD
    CLK[Clock] --> RING[Ring Counter 8-stage]
    RST[Reset] --> RING
    EN[Enable] --> RING
    RING --> OUT[One-Hot Output 8-bit]""",
    "verilog": """`timescale 1ns/1ps

module ring_counter_8bit(
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [7:0] ring_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        ring_out <= 8'b00000001;
    else if (enable)
        ring_out <= {ring_out[6:0], ring_out[7]};
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    wire [7:0] ring_out;
    
    ring_counter_8bit dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .ring_out(ring_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #200;
        
        enable = 0;
        #40;
        enable = 1;
        
        #150;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "Johnson counter with 4-bit shift register",
    "mermaid": """graph TD
    CLK[Clock] --> JOHNSON[Johnson Counter]
    RST[Reset] --> JOHNSON
    EN[Enable] --> JOHNSON
    JOHNSON --> OUT[Output 4-bit]""",
    "verilog": """`timescale 1ns/1ps

module johnson_counter(
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [3:0] count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 4'b0000;
    else if (enable)
        count <= {count[2:0], ~count[3]};
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    wire [3:0] count;
    
    johnson_counter dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .count(count)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #250;
        
        enable = 0;
        #30;
        enable = 1;
        
        #100;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "Presettable counter with parallel load and count enable",
    "mermaid": """graph TD
    CLK[Clock] --> COUNTER[Presettable Counter]
    RST[Reset] --> COUNTER
    LOAD[Parallel Load] --> COUNTER
    EN[Count Enable] --> COUNTER
    DATA[Data In 8-bit] --> COUNTER
    COUNTER --> COUNT[Count Out 8-bit]
    COUNTER --> TC[Terminal Count]""",
    "verilog": """`timescale 1ns/1ps

module presettable_counter(
    input wire clk,
    input wire rst_n,
    input wire load,
    input wire count_enable,
    input wire [7:0] preset_data,
    output reg [7:0] count,
    output wire terminal_count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 8'b0;
    else if (load)
        count <= preset_data;
    else if (count_enable)
        count <= count + 1'b1;
end

assign terminal_count = (count == 8'hFF) ? 1'b1 : 1'b0;

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg load;
    reg count_enable;
    reg [7:0] preset_data;
    wire [7:0] count;
    wire terminal_count;
    
    presettable_counter dut(
        .clk(clk),
        .rst_n(rst_n),
        .load(load),
        .count_enable(count_enable),
        .preset_data(preset_data),
        .count(count),
        .terminal_count(terminal_count)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        load = 0;
        count_enable = 0;
        preset_data = 8'h00;
        
        #20 rst_n = 1;
        #10 count_enable = 1;
        
        #100;
        
        load = 1;
        preset_data = 8'hF0;
        #10 load = 0;
        
        #100;
        
        count_enable = 0;
        #40;
        count_enable = 1;
        
        #80;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},
{
    "description": "4-to-1 multiplexer with 2-bit select",
    "mermaid": """graph TD
    A[Input A] --> MUX[4:1 MUX]
    B[Input B] --> MUX
    C[Input C] --> MUX
    D[Input D] --> MUX
    SEL[Select 2-bit] --> MUX
    MUX --> Y[Output Y]""",
    "verilog": """`timescale 1ns/1ps

module mux_4to1(
    input wire [3:0] data_in,
    input wire [1:0] select,
    output reg out
);

always @(*) begin
    case(select)
        2'b00: out = data_in[0];
        2'b01: out = data_in[1];
        2'b10: out = data_in[2];
        2'b11: out = data_in[3];
        default: out = 1'b0;
    endcase
end

endmodule

module testbench;
    reg [3:0] data_in;
    reg [1:0] select;
    wire out;
    
    mux_4to1 dut(
        .data_in(data_in),
        .select(select),
        .out(out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_in = 4'b1010;
        select = 2'b00;
        
        #10 select = 2'b00;
        #10 select = 2'b01;
        #10 select = 2'b10;
        #10 select = 2'b11;
        
        data_in = 4'b0110;
        #10 select = 2'b00;
        #10 select = 2'b01;
        #10 select = 2'b10;
        #10 select = 2'b11;
        
        #20;
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "8-to-1 multiplexer with 3-bit select and enable",
    "mermaid": """graph TD
    D0[Data 0] --> MUX[8:1 MUX]
    D1[Data 1] --> MUX
    D2[Data 2] --> MUX
    D3[Data 3] --> MUX
    D4[Data 4] --> MUX
    D5[Data 5] --> MUX
    D6[Data 6] --> MUX
    D7[Data 7] --> MUX
    SEL[Select 3-bit] --> MUX
    EN[Enable] --> MUX
    MUX --> OUT[Output]""",
    "verilog": """`timescale 1ns/1ps

module mux_8to1(
    input wire [7:0] data_in,
    input wire [2:0] select,
    input wire enable,
    output reg out
);

always @(*) begin
    if (enable) begin
        case(select)
            3'b000: out = data_in[0];
            3'b001: out = data_in[1];
            3'b010: out = data_in[2];
            3'b011: out = data_in[3];
            3'b100: out = data_in[4];
            3'b101: out = data_in[5];
            3'b110: out = data_in[6];
            3'b111: out = data_in[7];
            default: out = 1'b0;
        endcase
    end
    else begin
        out = 1'b0;
    end
end

endmodule

module testbench;
    reg [7:0] data_in;
    reg [2:0] select;
    reg enable;
    wire out;
    
    mux_8to1 dut(
        .data_in(data_in),
        .select(select),
        .enable(enable),
        .out(out)
    );
    
    integer i;
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_in = 8'b10101100;
        enable = 0;
        select = 3'b000;
        
        #20 enable = 1;
        
        for (i = 0; i < 8; i = i + 1) begin
            select = i;
            #15;
        end
        
        enable = 0;
        #20;
        
        enable = 1;
        data_in = 8'b01010011;
        
        for (i = 0; i < 8; i = i + 1) begin
            select = i;
            #15;
        end
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "1-to-4 demultiplexer with 2-bit select and enable",
    "mermaid": """graph TD
    IN[Data Input] --> DEMUX[1:4 DEMUX]
    SEL[Select 2-bit] --> DEMUX
    EN[Enable] --> DEMUX
    DEMUX --> Y0[Output 0]
    DEMUX --> Y1[Output 1]
    DEMUX --> Y2[Output 2]
    DEMUX --> Y3[Output 3]""",
    "verilog": """`timescale 1ns/1ps

module demux_1to4(
    input wire data_in,
    input wire [1:0] select,
    input wire enable,
    output reg [3:0] data_out
);

always @(*) begin
    data_out = 4'b0000;
    if (enable) begin
        case(select)
            2'b00: data_out[0] = data_in;
            2'b01: data_out[1] = data_in;
            2'b10: data_out[2] = data_in;
            2'b11: data_out[3] = data_in;
        endcase
    end
end

endmodule

module testbench;
    reg data_in;
    reg [1:0] select;
    reg enable;
    wire [3:0] data_out;
    
    demux_1to4 dut(
        .data_in(data_in),
        .select(select),
        .enable(enable),
        .data_out(data_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_in = 0;
        select = 2'b00;
        enable = 0;
        
        #20 enable = 1;
        #10 data_in = 1;
        
        #10 select = 2'b00;
        #10 select = 2'b01;
        #10 select = 2'b10;
        #10 select = 2'b11;
        
        #10 data_in = 0;
        #10 select = 2'b00;
        #10 select = 2'b01;
        
        #10 enable = 0;
        #10 data_in = 1;
        #10 select = 2'b11;
        
        #20;
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "2-to-1 multiplexer with 8-bit bus width",
    "mermaid": """graph TD
    A[Input A 8-bit] --> MUX[2:1 MUX 8-bit]
    B[Input B 8-bit] --> MUX
    SEL[Select] --> MUX
    MUX --> Y[Output 8-bit]""",
    "verilog": """`timescale 1ns/1ps

module mux_2to1_8bit(
    input wire [7:0] data_a,
    input wire [7:0] data_b,
    input wire select,
    output wire [7:0] data_out
);

assign data_out = select ? data_b : data_a;

endmodule

module testbench;
    reg [7:0] data_a;
    reg [7:0] data_b;
    reg select;
    wire [7:0] data_out;
    
    mux_2to1_8bit dut(
        .data_a(data_a),
        .data_b(data_b),
        .select(select),
        .data_out(data_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_a = 8'hAA;
        data_b = 8'h55;
        select = 0;
        
        #20 select = 0;
        #20 select = 1;
        #20 select = 0;
        
        data_a = 8'hF0;
        data_b = 8'h0F;
        #20 select = 1;
        #20 select = 0;
        #20 select = 1;
        
        #20;
        
        $finish;
    end
    
endmodule"""
},
{
    "description": "16-to-1 multiplexer with 4-bit select and tri-state output",
    "mermaid": """graph TD
    D[Data Inputs 16-bit] --> MUX[16:1 MUX]
    SEL[Select 4-bit] --> MUX
    EN[Output Enable] --> MUX
    MUX --> OUT[Tri-state Output]""",
    "verilog": """`timescale 1ns/1ps

module mux_16to1(
    input wire [15:0] data_in,
    input wire [3:0] select,
    input wire output_enable,
    output reg out
);

reg selected_data;

always @(*) begin
    case(select)
        4'd0:  selected_data = data_in[0];
        4'd1:  selected_data = data_in[1];
        4'd2:  selected_data = data_in[2];
        4'd3:  selected_data = data_in[3];
        4'd4:  selected_data = data_in[4];
        4'd5:  selected_data = data_in[5];
        4'd6:  selected_data = data_in[6];
        4'd7:  selected_data = data_in[7];
        4'd8:  selected_data = data_in[8];
        4'd9:  selected_data = data_in[9];
        4'd10: selected_data = data_in[10];
        4'd11: selected_data = data_in[11];
        4'd12: selected_data = data_in[12];
        4'd13: selected_data = data_in[13];
        4'd14: selected_data = data_in[14];
        4'd15: selected_data = data_in[15];
        default: selected_data = 1'b0;
    endcase
    
    if (output_enable)
        out = selected_data;
    else
        out = 1'b0;
end

endmodule

module testbench;
    reg [15:0] data_in;
    reg [3:0] select;
    reg output_enable;
    wire out;
    
    mux_16to1 dut(
        .data_in(data_in),
        .select(select),
        .output_enable(output_enable),
        .out(out)
    );
    
    integer i;
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_in = 16'b1010101010101010;
        output_enable = 1;
        select = 4'd0;
        
        for (i = 0; i < 16; i = i + 1) begin
            select = i;
            #10;
        end
        
        output_enable = 0;
        #30;
        
        output_enable = 1;
        data_in = 16'hF0F0;
        
        for (i = 0; i < 16; i = i + 1) begin
            select = i;
            #10;
        end
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "1-to-8 demultiplexer with active low enable",
    "mermaid": """graph TD
    IN[Data Input] --> DEMUX[1:8 DEMUX]
    SEL[Select 3-bit] --> DEMUX
    EN[Enable Active Low] --> DEMUX
    DEMUX --> Y0[Output 0]
    DEMUX --> Y1[Output 1]
    DEMUX --> Y2[Output 2]
    DEMUX --> Y3[Output 3]
    DEMUX --> Y4[Output 4]
    DEMUX --> Y5[Output 5]
    DEMUX --> Y6[Output 6]
    DEMUX --> Y7[Output 7]""",
    "verilog": """`timescale 1ns/1ps

module demux_1to8(
    input wire data_in,
    input wire [2:0] select,
    input wire enable_n,
    output reg [7:0] data_out
);

always @(*) begin
    data_out = 8'b00000000;
    if (!enable_n) begin
        case(select)
            3'd0: data_out[0] = data_in;
            3'd1: data_out[1] = data_in;
            3'd2: data_out[2] = data_in;
            3'd3: data_out[3] = data_in;
            3'd4: data_out[4] = data_in;
            3'd5: data_out[5] = data_in;
            3'd6: data_out[6] = data_in;
            3'd7: data_out[7] = data_in;
        endcase
    end
end

endmodule

module testbench;
    reg data_in;
    reg [2:0] select;
    reg enable_n;
    wire [7:0] data_out;
    
    demux_1to8 dut(
        .data_in(data_in),
        .select(select),
        .enable_n(enable_n),
        .data_out(data_out)
    );
    
    integer i;
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_in = 0;
        select = 3'd0;
        enable_n = 1;
        
        #20 enable_n = 0;
        #10 data_in = 1;
        
        for (i = 0; i < 8; i = i + 1) begin
            select = i;
            #15;
        end
        
        data_in = 0;
        #20;
        
        enable_n = 1;
        data_in = 1;
        select = 3'd5;
        #20;
        
        enable_n = 0;
        #20;
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "4-to-1 multiplexer with 4-bit bus width and priority encoding",
    "mermaid": """graph TD
    A[Input A 4-bit] --> MUX[4:1 MUX Priority]
    B[Input B 4-bit] --> MUX
    C[Input C 4-bit] --> MUX
    D[Input D 4-bit] --> MUX
    SEL[Select 2-bit] --> MUX
    VALID[Valid Flags 4-bit] --> MUX
    MUX --> OUT[Output 4-bit]
    MUX --> ERR[Error Flag]""",
    "verilog": """`timescale 1ns/1ps

module mux_4to1_4bit_priority(
    input wire [3:0] data_a,
    input wire [3:0] data_b,
    input wire [3:0] data_c,
    input wire [3:0] data_d,
    input wire [1:0] select,
    input wire [3:0] valid,
    output reg [3:0] data_out,
    output reg error
);

always @(*) begin
    error = 1'b0;
    case(select)
        2'b00: begin
            if (valid[0]) data_out = data_a;
            else begin
                data_out = 4'b0000;
                error = 1'b1;
            end
        end
        2'b01: begin
            if (valid[1]) data_out = data_b;
            else begin
                data_out = 4'b0000;
                error = 1'b1;
            end
        end
        2'b10: begin
            if (valid[2]) data_out = data_c;
            else begin
                data_out = 4'b0000;
                error = 1'b1;
            end
        end
        2'b11: begin
            if (valid[3]) data_out = data_d;
            else begin
                data_out = 4'b0000;
                error = 1'b1;
            end
        end
    endcase
end

endmodule

module testbench;
    reg [3:0] data_a, data_b, data_c, data_d;
    reg [1:0] select;
    reg [3:0] valid;
    wire [3:0] data_out;
    wire error;
    
    mux_4to1_4bit_priority dut(
        .data_a(data_a),
        .data_b(data_b),
        .data_c(data_c),
        .data_d(data_d),
        .select(select),
        .valid(valid),
        .data_out(data_out),
        .error(error)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_a = 4'hA;
        data_b = 4'hB;
        data_c = 4'hC;
        data_d = 4'hD;
        valid = 4'b1111;
        select = 2'b00;
        
        #10 select = 2'b00;
        #10 select = 2'b01;
        #10 select = 2'b10;
        #10 select = 2'b11;
        
        valid = 4'b1010;
        #10 select = 2'b00;
        #10 select = 2'b01;
        #10 select = 2'b10;
        #10 select = 2'b11;
        
        valid = 4'b0000;
        #10 select = 2'b00;
        
        #20;
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "Dual 4-to-1 multiplexer with common select and individual enables",
    "mermaid": """graph TD
    A0[MUX A Inputs 4-bit] --> MUXA[MUX A]
    B0[MUX B Inputs 4-bit] --> MUXB[MUX B]
    SEL[Common Select 2-bit] --> MUXA
    SEL --> MUXB
    ENA[Enable A] --> MUXA
    ENB[Enable B] --> MUXB
    MUXA --> OUTA[Output A]
    MUXB --> OUTB[Output B]""",
    "verilog": """`timescale 1ns/1ps

module dual_mux_4to1(
    input wire [3:0] data_a_in,
    input wire [3:0] data_b_in,
    input wire [1:0] select,
    input wire enable_a,
    input wire enable_b,
    output reg out_a,
    output reg out_b
);

always @(*) begin
    if (enable_a) begin
        case(select)
            2'b00: out_a = data_a_in[0];
            2'b01: out_a = data_a_in[1];
            2'b10: out_a = data_a_in[2];
            2'b11: out_a = data_a_in[3];
        endcase
    end
    else begin
        out_a = 1'b0;
    end
    
    if (enable_b) begin
        case(select)
            2'b00: out_b = data_b_in[0];
            2'b01: out_b = data_b_in[1];
            2'b10: out_b = data_b_in[2];
            2'b11: out_b = data_b_in[3];
        endcase
    end
    else begin
        out_b = 1'b0;
    end
end

endmodule

module testbench;
    reg [3:0] data_a_in;
    reg [3:0] data_b_in;
    reg [1:0] select;
    reg enable_a;
    reg enable_b;
    wire out_a;
    wire out_b;
    
    dual_mux_4to1 dut(
        .data_a_in(data_a_in),
        .data_b_in(data_b_in),
        .select(select),
        .enable_a(enable_a),
        .enable_b(enable_b),
        .out_a(out_a),
        .out_b(out_b)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_a_in = 4'b1100;
        data_b_in = 4'b0011;
        enable_a = 1;
        enable_b = 1;
        select = 2'b00;
        
        #10 select = 2'b00;
        #10 select = 2'b01;
        #10 select = 2'b10;
        #10 select = 2'b11;
        
        enable_a = 0;
        #10 select = 2'b00;
        #10 select = 2'b01;
        
        enable_a = 1;
        enable_b = 0;
        #10 select = 2'b10;
        #10 select = 2'b11;
        
        enable_a = 1;
        enable_b = 1;
        data_a_in = 4'b0101;
        data_b_in = 4'b1010;
        #10 select = 2'b00;
        #10 select = 2'b01;
        
        #20;
        
        $finish;
    end
    
endmodule"""
},
# Add these 4 additional multiplexer/demux examples to your knowledge_base list

{
    "description": "Cascaded 4-to-1 multiplexers forming 8-to-1 with hierarchical selection",
    "mermaid": """graph TD
    D0[Data 0-3] --> MUX1[MUX 4:1 Stage 1]
    D1[Data 4-7] --> MUX2[MUX 4:1 Stage 2]
    SEL0[Select Low 2-bit] --> MUX1
    SEL0 --> MUX2
    SEL1[Select High 1-bit] --> FINAL[Final MUX 2:1]
    MUX1 --> FINAL
    MUX2 --> FINAL
    FINAL --> OUT[Output]""",
    "verilog": """`timescale 1ns/1ps

module cascaded_mux_8to1(
    input wire [7:0] data_in,
    input wire [2:0] select,
    output wire out
);

wire mux1_out, mux2_out;

assign mux1_out = (select[1:0] == 2'b00) ? data_in[0] :
                  (select[1:0] == 2'b01) ? data_in[1] :
                  (select[1:0] == 2'b10) ? data_in[2] :
                                           data_in[3];

assign mux2_out = (select[1:0] == 2'b00) ? data_in[4] :
                  (select[1:0] == 2'b01) ? data_in[5] :
                  (select[1:0] == 2'b10) ? data_in[6] :
                                           data_in[7];

assign out = select[2] ? mux2_out : mux1_out;

endmodule

module testbench;
    reg [7:0] data_in;
    reg [2:0] select;
    wire out;
    
    cascaded_mux_8to1 dut(
        .data_in(data_in),
        .select(select),
        .out(out)
    );
    
    integer i;
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        data_in = 8'b11001010;
        select = 3'd0;
        
        for (i = 0; i < 8; i = i + 1) begin
            select = i;
            #15;
        end
        
        data_in = 8'b01010101;
        
        for (i = 0; i < 8; i = i + 1) begin
            select = i;
            #15;
        end
        
        #20;
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "Bidirectional multiplexer-demultiplexer with direction control",
    "mermaid": """graph TD
    A[Port A 4-bit] --> BIDIR[Bi-directional MUX/DEMUX]
    B[Port B 4-bit] --> BIDIR
    DIR[Direction] --> BIDIR
    SEL[Select 2-bit] --> BIDIR
    BIDIR --> OUT[Output 4-bit]""",
    "verilog": """`timescale 1ns/1ps

module bidir_mux_demux(
    input wire [3:0] port_a,
    input wire [3:0] port_b,
    input wire [1:0] select,
    input wire direction,
    output reg [3:0] data_out
);

always @(*) begin
    if (direction) begin
        case(select)
            2'b00: data_out = {3'b000, port_a[0]};
            2'b01: data_out = {3'b000, port_a[1]};
            2'b10: data_out = {3'b000, port_a[2]};
            2'b11: data_out = {3'b000, port_a[3]};
        endcase
    end
    else begin
        case(select)
            2'b00: data_out[0] = port_b[0];
            2'b01: data_out[1] = port_b[1];
            2'b10: data_out[2] = port_b[2];
            2'b11: data_out[3] = port_b[3];
            default: data_out = 4'b0000;
        endcase
    end
end

endmodule

module testbench;
    reg [3:0] port_a;
    reg [3:0] port_b;
    reg [1:0] select;
    reg direction;
    wire [3:0] data_out;
    
    bidir_mux_demux dut(
        .port_a(port_a),
        .port_b(port_b),
        .select(select),
        .direction(direction),
        .data_out(data_out)
    );
    
    integer i;
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        port_a = 4'b1010;
        port_b = 4'b0101;
        direction = 1;
        select = 2'b00;
        
        for (i = 0; i < 4; i = i + 1) begin
            select = i;
            #15;
        end
        
        direction = 0;
        
        for (i = 0; i < 4; i = i + 1) begin
            select = i;
            #15;
        end
        
        direction = 1;
        port_a = 4'b1100;
        
        for (i = 0; i < 4; i = i + 1) begin
            select = i;
            #15;
        end
        
        #20;
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "3-to-8 line decoder functioning as 1-to-8 demultiplexer",
    "mermaid": """graph TD
    A[Address A] --> DEC[3:8 Decoder]
    B[Address B] --> DEC
    C[Address C] --> DEC
    EN[Enable] --> DEC
    DEC --> Y0[Output 0]
    DEC --> Y1[Output 1]
    DEC --> Y2[Output 2]
    DEC --> Y3[Output 3]
    DEC --> Y4[Output 4]
    DEC --> Y5[Output 5]
    DEC --> Y6[Output 6]
    DEC --> Y7[Output 7]""",
    "verilog": """`timescale 1ns/1ps

module decoder_3to8(
    input wire [2:0] address,
    input wire enable,
    output reg [7:0] outputs
);

always @(*) begin
    outputs = 8'b00000000;
    if (enable) begin
        case(address)
            3'd0: outputs = 8'b00000001;
            3'd1: outputs = 8'b00000010;
            3'd2: outputs = 8'b00000100;
            3'd3: outputs = 8'b00001000;
            3'd4: outputs = 8'b00010000;
            3'd5: outputs = 8'b00100000;
            3'd6: outputs = 8'b01000000;
            3'd7: outputs = 8'b10000000;
        endcase
    end
end

endmodule

module testbench;
    reg [2:0] address;
    reg enable;
    wire [7:0] outputs;
    
    decoder_3to8 dut(
        .address(address),
        .enable(enable),
        .outputs(outputs)
    );
    
    integer i;
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        address = 3'd0;
        enable = 0;
        
        #20 enable = 1;
        
        for (i = 0; i < 8; i = i + 1) begin
            address = i;
            #15;
        end
        
        enable = 0;
        #20;
        address = 3'd5;
        #20;
        
        enable = 1;
        #20;
        
        for (i = 7; i >= 0; i = i - 1) begin
            address = i;
            #15;
        end
        
        #20;
        
        $finish;
    end
    
endmodule"""
},

{
    "description": "Quad 2-to-1 multiplexer with common select for bus switching",
    "mermaid": """graph TD
    BUS_A[Bus A 4-bit] --> MUX[Quad 2:1 MUX]
    BUS_B[Bus B 4-bit] --> MUX
    SEL[Common Select] --> MUX
    MUX --> BUS_OUT[Bus Output 4-bit]""",
    "verilog": """`timescale 1ns/1ps

module quad_mux_2to1(
    input wire [3:0] bus_a,
    input wire [3:0] bus_b,
    input wire select,
    output wire [3:0] bus_out
);

assign bus_out[0] = select ? bus_b[0] : bus_a[0];
assign bus_out[1] = select ? bus_b[1] : bus_a[1];
assign bus_out[2] = select ? bus_b[2] : bus_a[2];
assign bus_out[3] = select ? bus_b[3] : bus_a[3];

endmodule

module testbench;
    reg [3:0] bus_a;
    reg [3:0] bus_b;
    reg select;
    wire [3:0] bus_out;
    
    quad_mux_2to1 dut(
        .bus_a(bus_a),
        .bus_b(bus_b),
        .select(select),
        .bus_out(bus_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        bus_a = 4'b0000;
        bus_b = 4'b1111;
        select = 0;
        
        #20 select = 0;
        #20 select = 1;
        #20 select = 0;
        
        bus_a = 4'b1010;
        bus_b = 4'b0101;
        #20 select = 0;
        #20 select = 1;
        
        bus_a = 4'b1100;
        bus_b = 4'b0011;
        #20 select = 0;
        #20 select = 1;
        
        #20 bus_a = 4'b1111;
        #20 bus_b = 4'b0000;
        #20 select = 0;
        #20 select = 1;
        
        #20;
        
        $finish;
    end
    
endmodule"""
},
# Add these 6 shift register examples to your knowledge_base list

{
    "description": "8-bit serial-in serial-out shift register with clock and reset",
    "mermaid": """graph TD
    CLK[Clock] --> SISO[SISO 8-bit]
    RST[Reset] --> SISO
    SI[Serial In] --> SISO
    SISO --> SO[Serial Out]""",
    "verilog": """`timescale 1ns/1ps

module siso_shift_register(
    input wire clk,
    input wire rst_n,
    input wire serial_in,
    output wire serial_out
);


reg [7:0] shift_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        shift_reg <= 8'b0;
    else
        shift_reg <= {shift_reg[6:0], serial_in};
end

assign serial_out = shift_reg[7];

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg serial_in;
    wire serial_out;
    
    siso_shift_register dut(
        .clk(clk),
        .rst_n(rst_n),
        .serial_in(serial_in),
        .serial_out(serial_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        serial_in = 0;
        
        #20 rst_n = 1;
        
        #10 serial_in = 1;
        #10 serial_in = 0;
        #10 serial_in = 1;
        #10 serial_in = 1;
        #10 serial_in = 0;
        #10 serial_in = 0;
        #10 serial_in = 1;
        #10 serial_in = 0;
        
        #100 serial_in = 0;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "8-bit serial-in parallel-out shift register",
    "mermaid": """graph TD
    CLK[Clock] --> SIPO[SIPO 8-bit]
    RST[Reset] --> SIPO
    SI[Serial In] --> SIPO
    SIPO --> PO[Parallel Out 8-bit]""",
    "verilog": """`timescale 1ns/1ps

module sipo_shift_register(
    input wire clk,
    input wire rst_n,
    input wire serial_in,
    output reg [7:0] parallel_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        parallel_out <= 8'b0;
    else
        parallel_out <= {parallel_out[6:0], serial_in};
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg serial_in;
    wire [7:0] parallel_out;
    
    sipo_shift_register dut(
        .clk(clk),
        .rst_n(rst_n),
        .serial_in(serial_in),
        .parallel_out(parallel_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        serial_in = 0;
        
        #20 rst_n = 1;
        
        #10 serial_in = 1;
        #10 serial_in = 1;
        #10 serial_in = 0;
        #10 serial_in = 0;
        #10 serial_in = 1;
        #10 serial_in = 0;
        #10 serial_in = 1;
        #10 serial_in = 0;
        
        #50;
        
        serial_in = 0;
        #10 serial_in = 1;
        #10 serial_in = 0;
        #10 serial_in = 1;
        
        #50;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "8-bit parallel-in serial-out shift register with load",
    "mermaid": """graph TD
    CLK[Clock] --> PISO[PISO 8-bit]
    RST[Reset] --> PISO
    LOAD[Load] --> PISO
    PI[Parallel In 8-bit] --> PISO
    PISO --> SO[Serial Out]""",
    "verilog": """`timescale 1ns/1ps

module piso_shift_register(
    input wire clk,
    input wire rst_n,
    input wire load,
    input wire [7:0] parallel_in,
    output wire serial_out
);

reg [7:0] shift_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        shift_reg <= 8'b0;
    else if (load)
        shift_reg <= parallel_in;
    else
        shift_reg <= {shift_reg[6:0], 1'b0};
end

assign serial_out = shift_reg[7];

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg load;
    reg [7:0] parallel_in;
    wire serial_out;
    
    piso_shift_register dut(
        .clk(clk),
        .rst_n(rst_n),
        .load(load),
        .parallel_in(parallel_in),
        .serial_out(serial_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        load = 0;
        parallel_in = 8'b0;
        
        #20 rst_n = 1;
        
        #10 parallel_in = 8'b10110100;
        #10 load = 1;
        #10 load = 0;
        
        #100;
        
        parallel_in = 8'b11001010;
        #10 load = 1;
        #10 load = 0;
        
        #100;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "8-bit parallel-in parallel-out shift register with shift enable",
    "mermaid": """graph TD
    CLK[Clock] --> PIPO[PIPO 8-bit]
    RST[Reset] --> PIPO
    LOAD[Load] --> PIPO
    SHIFT[Shift Enable] --> PIPO
    PI[Parallel In 8-bit] --> PIPO
    PIPO --> PO[Parallel Out 8-bit]""",
    "verilog": """`timescale 1ns/1ps

module pipo_shift_register(
    input wire clk,
    input wire rst_n,
    input wire load,
    input wire shift_enable,
    input wire [7:0] parallel_in,
    output reg [7:0] parallel_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        parallel_out <= 8'b0;
    else if (load)
        parallel_out <= parallel_in;
    else if (shift_enable)
        parallel_out <= {parallel_out[6:0], 1'b0};
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg load;
    reg shift_enable;
    reg [7:0] parallel_in;
    wire [7:0] parallel_out;
    
    pipo_shift_register dut(
        .clk(clk),
        .rst_n(rst_n),
        .load(load),
        .shift_enable(shift_enable),
        .parallel_in(parallel_in),
        .parallel_out(parallel_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        load = 0;
        shift_enable = 0;
        parallel_in = 8'b0;
        
        #20 rst_n = 1;
        
        #10 parallel_in = 8'b11110000;
        #10 load = 1;
        #10 load = 0;
        #10 shift_enable = 1;
        
        #100;
        
        shift_enable = 0;
        #30;
        
        parallel_in = 8'b10101010;
        #10 load = 1;
        #10 load = 0;
        #10 shift_enable = 1;
        
        #80;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "4-bit bidirectional shift register with left-right control",
    "mermaid": """graph TD
    CLK[Clock] --> BIDIR[Bidirectional 4-bit]
    RST[Reset] --> BIDIR
    DIR[Direction] --> BIDIR
    SI[Serial In] --> BIDIR
    BIDIR --> PO[Parallel Out 4-bit]
    BIDIR --> SO[Serial Out]""",
    "verilog": """`timescale 1ns/1ps

module bidirectional_shift_register(
    input wire clk,
    input wire rst_n,
    input wire direction,
    input wire serial_in,
    output reg [3:0] parallel_out,
    output wire serial_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        parallel_out <= 4'b0;
    else if (direction)
        parallel_out <= {parallel_out[2:0], serial_in};
    else
        parallel_out <= {serial_in, parallel_out[3:1]};
end

assign serial_out = direction ? parallel_out[3] : parallel_out[0];

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg direction;
    reg serial_in;
    wire [3:0] parallel_out;
    wire serial_out;
    
    bidirectional_shift_register dut(
        .clk(clk),
        .rst_n(rst_n),
        .direction(direction),
        .serial_in(serial_in),
        .parallel_out(parallel_out),
        .serial_out(serial_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        direction = 1;
        serial_in = 0;
        
        #20 rst_n = 1;
        
        #10 serial_in = 1;
        #10 serial_in = 0;
        #10 serial_in = 1;
        #10 serial_in = 1;
        
        #30;
        
        direction = 0;
        #10 serial_in = 0;
        #10 serial_in = 1;
        #10 serial_in = 0;
        #10 serial_in = 1;
        
        #50;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
},

{
    "description": "8-bit LFSR with Fibonacci configuration for pseudo-random generation",
    "mermaid": """graph TD
    CLK[Clock] --> LFSR[LFSR 8-bit]
    RST[Reset] --> LFSR
    EN[Enable] --> LFSR
    SEED[Seed 8-bit] --> LFSR
    LFSR --> OUT[Random Out 8-bit]""",
    "verilog": """`timescale 1ns/1ps

module lfsr_8bit(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [7:0] seed,
    output reg [7:0] lfsr_out
);

wire feedback;

assign feedback = lfsr_out[7] ^ lfsr_out[5] ^ lfsr_out[4] ^ lfsr_out[3];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        lfsr_out <= seed;
    else if (enable)
        lfsr_out <= {lfsr_out[6:0], feedback};
end

endmodule

module testbench;
    reg clk;
    reg rst_n;
    reg enable;
    reg [7:0] seed;
    wire [7:0] lfsr_out;
    
    lfsr_8bit dut(
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .seed(seed),
        .lfsr_out(lfsr_out)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        enable = 0;
        seed = 8'b10101010;
        
        #20 rst_n = 1;
        #10 enable = 1;
        
        #200;
        
        enable = 0;
        #30;
        
        seed = 8'b11001100;
        rst_n = 0;
        #20 rst_n = 1;
        #10 enable = 1;
        
        #150;
        
        $finish;
    end
    
    always #5 clk = ~clk;
    
endmodule"""
}                     
        ]
    
    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones"""
        os.makedirs("data", exist_ok=True)
        
        if os.path.exists(self.rag_data_file):
            try:
                with open(self.rag_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    # Verify knowledge base matches
                    if len(data['knowledge_base']) == len(self.knowledge_base):
                        return
            except:
                pass
        
        # Create new embeddings
        descriptions = [item['description'] for item in self.knowledge_base]
        self.embeddings = self.model.encode(descriptions)
        
        # Save for future use
        with open(self.rag_data_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'knowledge_base': self.knowledge_base
            }, f)
    
    def retrieve_similar_examples(self, query: str, top_k: int = 2) -> List[Dict]:
        """Retrieve most similar examples from knowledge base"""
        if not self.embeddings.any():
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.knowledge_base[idx],
                'similarity': similarities[idx]
            })
        
        return results
    
    def add_example(self, description: str, mermaid: str, verilog: str):
        """Add new example to knowledge base"""
        new_example = {
            "description": description,
            "mermaid": mermaid,
            "verilog": verilog
        }
        
        self.knowledge_base.append(new_example)
        
        # Re-compute embeddings
        descriptions = [item['description'] for item in self.knowledge_base]
        self.embeddings = self.model.encode(descriptions)
        
        # Save updated data
        with open(self.rag_data_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'knowledge_base': self.knowledge_base
            }, f)
    
    def enhance_prompt_with_rag(self, user_prompt: str, generation_type: str = "mermaid") -> str:
        """Enhance prompt with relevant examples from RAG"""
        similar_examples = self.retrieve_similar_examples(user_prompt, top_k=2)
        
        if not similar_examples:
            return user_prompt
        
        # Build enhanced prompt
        enhanced_prompt = f"User Request: {user_prompt}\n\n"
        enhanced_prompt += "Here are similar examples from the knowledge base:\n\n"
        
        for i, example in enumerate(similar_examples, 1):
            enhanced_prompt += f"Example {i} (similarity: {example['similarity']:.3f}):\n"
            enhanced_prompt += f"Description: {example['description']}\n"
            
            if generation_type == "mermaid":
                enhanced_prompt += f"Mermaid:\n{example['mermaid']}\n"
            elif generation_type == "verilog":
                enhanced_prompt += f"Verilog:\n{example['verilog']}\n"
            else:
                enhanced_prompt += f"Mermaid:\n{example['mermaid']}\n"
                enhanced_prompt += f"Verilog:\n{example['verilog']}\n"
            
            enhanced_prompt += "\n"
        
        enhanced_prompt += f"Based on these examples, generate a response for: {user_prompt}"
        
        return enhanced_prompt