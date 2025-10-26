# 8-bit Register File

**Category**: CPU Component  
**Complexity**: complex

## Original Prompt

```
Create 8-register × 8-bit register file with dual read ports and single write port. Include testbench with simultaneous operations.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> RF[[8x8 Register File]]
    RST[Reset] --> RF
    WEN[Write Enable] --> RF
    WADDR[Write Address] --> RF
    WDATA[Write Data] --> RF
    RA1[Read Address 1] --> RF
    RA2[Read Address 2] --> RF
    RF --> RD1[Read Data 1]
    RF --> RD2[Read Data 2]
    subgraph TestBench
        TB_CLK[Test Clock] --> TB[Test Controller]
        TB_RST[Test Reset] --> TB
        TB --> WADDR
        TB --> WDATA
        TB --> WEN
        TB --> RA1
        TB --> RA2
        RD1 --> VERIFY{Verification}
        RD2 --> VERIFY
    end
```
