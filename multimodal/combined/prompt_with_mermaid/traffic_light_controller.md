# Traffic Light Controller

**Category**: State Machine  
**Complexity**: complex

## Original Prompt

```
Design a traffic light FSM with 4 states: NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW. Inputs: clk, rst, emergency. Outputs: ns_light[1:0], ew_light[1:0]. Timing: GREEN=8 cycles, YELLOW=2 cycles. Emergency makes both RED. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph TD
    CLK[clk] --> FSM[[State Register]]
    RST[rst] --> FSM
    EMG[emergency] --> FSM

    FSM --> COUNTER[[Cycle Counter]]
    CLK --> COUNTER
    RST --> COUNTER

    FSM --> DECODE(State Decoder)
    COUNTER --> NEXT(Next State Logic)
    FSM --> NEXT
    EMG --> NEXT

    NEXT --> FSM

    DECODE --> NS["ns_light[1:0]"]
    DECODE --> EW["ew_light[1:0]"]

    TB[Testbench] -.-> CLK
    TB -.-> RST
    TB -.-> EMG
    TB -.-> |verify|NS
    TB -.-> |verify|EW

    subgraph States
    S1[NS_GREEN]
    S2[NS_YELLOW]
    S3[EW_GREEN]
    S4[EW_YELLOW]
    end
```
