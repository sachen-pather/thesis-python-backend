# Mealy Sequence Detector (101)

**Category**: State Machine  
**Complexity**: medium

## Original Prompt

```
Design a Mealy FSM detecting sequence 101 in serial input with non-overlapping detection. Output high for one cycle when pattern detected. Include testbench with sequence: 1101011010.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> FSM((Mealy FSM))
    RST[Reset] --> FSM
    SIN[Serial In] --> FSM
    FSM --> OUT[Detected]
    
    TB[[Testbench: 1101011010]] --> SIN
    TB --> CLK
    TB --> RST
    OUT --> VER{Verify Detection}
    VER --> PASS[Test Results]
    
    FSM -->|S0| S0((IDLE))
    FSM -->|S1| S1((Got 1))
    FSM -->|S2| S2((Got 10))
    
    S0 -->|0/0| S0
    S0 -->|1/0| S1
    S1 -->|1/0| S1
    S1 -->|0/0| S2
    S2 -->|1/1| S0
    S2 -->|0/0| S0
    
    style FSM fill:#f9f,stroke:#333
```
