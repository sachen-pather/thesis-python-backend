# 8-bit SIPO Shift Register

**Category**: Sequential  
**Complexity**: simple

## Original Prompt

```
Design an 8-bit Serial-In Parallel-Out (SIPO) shift register with clock, reset, serial input, and parallel output[7:0]. Include testbench shifting in pattern 10101100.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> SR[[8-bit Shift Register]]
    RST[Reset] --> SR
    SIN[Serial In] --> SR
    SR --> Q0[Q[0]]
    SR --> Q1[Q[1]]
    SR --> Q2[Q[2]]
    SR --> Q3[Q[3]]
    SR --> Q4[Q[4]]
    SR --> Q5[Q[5]]
    SR --> Q6[Q[6]]
    SR --> Q7[Q[7]]
    TB[Testbench] --> |Clock Gen|CLK
    TB --> |Reset Pulse|RST
    TB --> |Pattern: 10101100|SIN
    Q0 --> VER{Verify Shift}
    Q1 --> VER
    Q2 --> VER
    Q3 --> VER
    Q4 --> VER
    Q5 --> VER
    Q6 --> VER
    Q7 --> VER
    VER --> PASS[Test Results]
    style SR fill:#f9f,stroke:#333
```
