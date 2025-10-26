# UART Transmitter

**Category**: State Machine  
**Complexity**: complex

## Original Prompt

```
Design UART transmitter with states: IDLE, START, DATA, STOP. Protocol: START(0), 8 data bits LSB-first, STOP(1). Baud rate divider. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> CTRL(FSM Control)
    RST[Reset] --> CTRL
    TX_START[TX Start] --> CTRL
    TX_DATA[TX Data] --> SHIFTREG[[Shift Register]]

    CTRL --> |state|IDLE{IDLE}
    CTRL --> |state|START{START}
    CTRL --> |state|DATA{DATA}
    CTRL --> |state|STOP{STOP}

    BAUD_DIV[[Baud Divider]] --> CTRL
    CLK --> BAUD_DIV

    SHIFTREG --> |serial_out|MUX{Bit Select}
    START --> MUX
    STOP --> MUX
    MUX --> TX_OUT[TX Output]

    COUNTER[[Bit Counter]] --> CTRL
    CTRL --> COUNTER
    CLK --> COUNTER

    TB[Testbench] --> TX_START
    TB --> TX_DATA
    TX_OUT --> TB
```
