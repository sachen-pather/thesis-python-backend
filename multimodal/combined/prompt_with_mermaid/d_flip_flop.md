# D Flip-Flop

**Category**: Sequential  
**Complexity**: simple

## Original Prompt

```
Create a D flip-flop with clock, reset, input d, and output q. Include testbench with timing verification.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock Input] --> DFF[[D Flip-Flop]]
    RST[Reset] --> DFF
    D[Data Input] --> DFF
    DFF --> Q[Q Output]
    TB[Testbench] --> |Clock Gen|CLK
    TB --> |Reset Stimulus|RST
    TB --> |Test Vector|D
    Q --> |Verify|TV{Timing Verify}
    CLK --> |Clock Edge|TV
    TV --> |Pass/Fail|RESULT[Test Results]
    style DFF fill:#f9f,stroke:#333
    style TV fill:#ffd,stroke:#333
```
