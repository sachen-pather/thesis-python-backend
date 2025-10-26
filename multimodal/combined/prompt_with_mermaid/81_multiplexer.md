# 8:1 Multiplexer

**Category**: Combinational  
**Complexity**: medium

## Original Prompt

```
Create an 8:1 multiplexer with input data[7:0], select sel[2:0], and output out. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    D0[data[0]] --> MUX{8:1 MUX}
    D1[data[1]] --> MUX
    D2[data[2]] --> MUX
    D3[data[3]] --> MUX
    D4[data[4]] --> MUX
    D5[data[5]] --> MUX
    D6[data[6]] --> MUX
    D7[data[7]] --> MUX
    S0[sel[0]] --> MUX
    S1[sel[1]] --> MUX
    S2[sel[2]] --> MUX
    MUX --> OUT[out]
    TB[[Testbench]]
    TB --> D0
    TB --> D1
    TB --> D2
    TB --> D3
    TB --> D4
    TB --> D5
    TB --> D6
    TB --> D7
    TB --> S0
    TB --> S1
    TB --> S2
    OUT --> TB
```
