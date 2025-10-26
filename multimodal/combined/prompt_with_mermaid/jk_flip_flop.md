# JK Flip-Flop

**Category**: Sequential  
**Complexity**: simple

## Original Prompt

```
Create a JK flip-flop with clock, reset, inputs j and k, and outputs q and q_bar. Include testbench verifying all four input combinations (00, 01, 10, 11).
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> JKFF[[JK Flip-Flop]]
    RST[Reset] --> JKFF
    J[J Input] --> JKFF
    K[K Input] --> JKFF
    JKFF --> Q[Q Output]
    JKFF --> QB[Q_bar Output]
    TB[Testbench] --> |Clock Gen|CLK
    TB --> |Reset Pulse|RST
    TB --> |J=0,K=0: Hold|J
    TB --> |J=0,K=1: Reset|K
    TB --> |J=1,K=0: Set|J
    TB --> |J=1,K=1: Toggle|K
    Q --> |Monitor|MON{State Verify}
    QB --> MON
    MON --> RESULT[Pass/Fail]
    style JKFF fill:#f9f,stroke:#333
    style MON fill:#ffd,stroke:#333
```
