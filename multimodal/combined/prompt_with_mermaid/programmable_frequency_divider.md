# Programmable Frequency Divider

**Category**: Advanced Sequential  
**Complexity**: complex

## Original Prompt

```
Design a programmable frequency divider with 8-bit division ratio. Inputs: clk, rst, divisor[7:0] (divides by 2 to 256). Output: clk_out. Generate 50% duty cycle output. Include testbench with divisor values: 2, 4, 8, 16.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[clk] --> CTRL(Control Logic)
    RST[rst] --> CTRL
    DIV[divisor[7:0]] --> REG[[Divisor Register]]
    REG --> COMP{Comparator}
    
    CTRL --> CTR[[8-bit Counter]]
    CLK --> CTR
    RST --> CTR
    
    CTR --> COMP
    COMP --> |match|TOGGLE[[Toggle FF]]
    CLK --> TOGGLE
    RST --> TOGGLE
    TOGGLE --> CLKOUT[clk_out]
    
    TB[Testbench] --> DIV
    TB --> RST
    CLKOUT --> VER{Verify Frequency}
    VER --> PASS[Test Results]
    
    subgraph Test_Values
        T1[divisor=2]
        T2[divisor=4]
        T3[divisor=8]
        T4[divisor=16]
    end
```
