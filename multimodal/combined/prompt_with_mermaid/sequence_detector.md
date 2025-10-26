# Sequence Detector

**Category**: State Machine  
**Complexity**: medium

## Original Prompt

```
Design an FSM detecting pattern 1011 in serial input with overlapping detection. Include testbench with sequence: 10110111011.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> FSM((FSM Core))
    RST[Reset] --> FSM
    SI[Serial Input] --> FSM
    FSM --> DET[Detector Output]

    TB1[[Testbench Sequence 10110111011]] --> SI

    FSM -->|S0| S0((Idle))
    FSM -->|S1| S1((Got 1))
    FSM -->|S2| S2((Got 10))
    FSM -->|S3| S3((Got 101))
    FSM -->|S4| S4((Got 1011))

    S0 -->|1| S1
    S0 -->|0| S0
    S1 -->|0| S2
    S1 -->|1| S1
    S2 -->|1| S3
    S2 -->|0| S0
    S3 -->|1| S4
    S3 -->|0| S0
    S4 -->|1| S1
    S4 -->|0| S2
```
