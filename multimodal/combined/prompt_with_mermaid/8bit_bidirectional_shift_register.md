# 8-bit Bidirectional Shift Register

**Category**: Sequential  
**Complexity**: medium

## Original Prompt

```
Create an 8-bit bidirectional shift register with clock, reset, load, direction (1=right, 0=left), parallel input[7:0], serial_in_left, serial_in_right, and parallel output[7:0]. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> SR[[8-bit Shift Register]]
    RST[Reset] --> SR
    LOAD[Load] --> SR
    DIR[Direction] --> MUX{Direction MUX}
    PIN[parallel_in[7:0]] --> SR
    SIL[serial_in_left] --> MUX
    SIR[serial_in_right] --> MUX
    MUX --> SR
    SR --> POUT[parallel_out[7:0]]
    
    TB[Testbench] --> CLK
    TB --> RST
    TB --> LOAD
    TB --> DIR
    TB --> PIN
    TB --> SIL
    TB --> SIR
    POUT --> VER{Verify Shift}
    VER --> PASS[Test Results]
    
    style SR fill:#f9f,stroke:#333
    style MUX fill:#bbf,stroke:#333
```
