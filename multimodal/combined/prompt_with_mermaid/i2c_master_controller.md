# I2C Master Controller

**Category**: Protocol  
**Complexity**: complex

## Original Prompt

```
Design an I2C master controller with start condition, stop condition, byte transmission, and ACK/NACK handling. Inputs: clk, rst, start, stop, data_in[7:0], wr_en. Outputs: scl, sda_out, sda_oe, busy, ack_received. Include testbench with write transaction.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[clk] --> CTRL(I2C Control FSM)
    RST[rst] --> CTRL
    START[start] --> CTRL
    STOP[stop] --> CTRL
    DATA[data_in[7:0]] --> SHIFT[[Shift Register]]
    WREN[wr_en] --> CTRL
    
    CTRL --> |state|IDLE{IDLE}
    CTRL --> |state|START_COND{START}
    CTRL --> |state|SEND_BYTE{SEND}
    CTRL --> |state|ACK_CHECK{ACK}
    CTRL --> |state|STOP_COND{STOP}
    
    CLKGEN[[SCL Generator]] --> SCL[scl]
    CLK --> CLKGEN
    CTRL --> CLKGEN
    
    SHIFT --> SDA_OUT[sda_out]
    CTRL --> SDA_OE[sda_oe]
    CTRL --> BUSY[busy]
    CTRL --> ACK[ack_received]
    
    TB[Testbench] --> START
    TB --> STOP
    TB --> DATA
    TB --> WREN
    SCL --> TB
    SDA_OUT --> TB
    BUSY --> TB
    ACK --> TB
```
