# SPI Master

**Category**: Protocol  
**Complexity**: complex

## Original Prompt

```
Design SPI master supporting mode 0 (CPOL=0, CPHA=0). Include SCLK generation, MOSI/MISO handling. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock Input] --> CLKGEN(Clock Generator)
    RST[Reset] --> CLKGEN
    RST --> CTRL(SPI Control FSM)
    CLK --> CTRL

    CLKGEN --> SCLK[SCLK Output]
    CLKGEN --> CTRL

    DATA_IN[TX Data] --> SHIFTER[[Shift Register]]
    CTRL --> SHIFTER
    SHIFTER --> MOSI[MOSI Output]

    MISO[MISO Input] --> RX_SHIFT[[RX Shift Register]]
    CTRL --> RX_SHIFT
    RX_SHIFT --> DATA_OUT[RX Data]

    CS_CTRL[CS Control] --> CTRL
    CTRL --> CS[CS Output]

    TB_CLK[TB Clock] -.-> TB{Testbench}
    TB_RST[TB Reset] -.-> TB
    TB -.-> DATA_IN
    TB -.-> CS_CTRL
    MOSI -.-> TB
    MISO -.-> TB
    SCLK -.-> TB
    DATA_OUT -.-> TB
```
