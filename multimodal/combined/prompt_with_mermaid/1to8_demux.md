# 1-to-8 Demultiplexer

**Category**: Combinational  
**Complexity**: medium

## Original Prompt

```
Create a 1-to-8 demultiplexer with input data_in, select sel[2:0], enable en, and outputs out[7:0]. When enabled, data_in routes to selected output. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    DIN[data_in] --> DEMUX{1:8 DEMUX}
    EN[enable] --> DEMUX
    SEL0[sel[0]] --> DEMUX
    SEL1[sel[1]] --> DEMUX
    SEL2[sel[2]] --> DEMUX
    DEMUX --> O0[out[0]]
    DEMUX --> O1[out[1]]
    DEMUX --> O2[out[2]]
    DEMUX --> O3[out[3]]
    DEMUX --> O4[out[4]]
    DEMUX --> O5[out[5]]
    DEMUX --> O6[out[6]]
    DEMUX --> O7[out[7]]
    TB[[Testbench]] --> DIN
    TB --> EN
    TB --> SEL0
    TB --> SEL1
    TB --> SEL2
    O0 --> VER{Verify Routing}
    O1 --> VER
    O2 --> VER
    O3 --> VER
    O4 --> VER
    O5 --> VER
    O6 --> VER
    O7 --> VER
    VER --> PASS[Test Results]
```
