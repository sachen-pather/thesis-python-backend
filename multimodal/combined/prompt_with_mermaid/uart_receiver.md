# UART Receiver

**Category**: Protocol  
**Complexity**: complex

## Original Prompt

```
Design a UART receiver with states: IDLE, START, DATA, STOP. Protocol: detect START(0), receive 8 data bits LSB-first, verify STOP(1). Inputs: clk, rst, rx. Outputs: rx_data[7:0], rx_valid, frame_error. Baud rate divider. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[clk] --> CTRL(FSM Control)
    RST[rst] --> CTRL
    RX[rx] --> SYNC[[Synchronizer]]
    SYNC --> CTRL
    
    CTRL --> |state|IDLE{IDLE}
    CTRL --> |state|START{START}
    CTRL --> |state|DATA{DATA}
    CTRL --> |state|STOP{STOP}
    
    BAUD_DIV[[Baud Divider]] --> CTRL
    CLK --> BAUD_DIV
    
    CTRL --> SHIFTREG[[Shift Register]]
    SYNC --> SHIFTREG
    SHIFTREG --> RXDATA[rx_data[7:0]]
    
    CTRL --> VALID[rx_valid]
    CTRL --> ERROR[frame_error]
    
    COUNTER[[Bit Counter]] --> CTRL
    CTRL --> COUNTER
    CLK --> COUNTER
    
    TB[Testbench] --> |Serial Data|RX
    RXDATA --> TB
    VALID --> TB
    ERROR --> TB
```
