# PWM Generator

**Category**: Advanced Sequential  
**Complexity**: complex

## Original Prompt

```
Design 8-bit resolution PWM generator with configurable duty cycle. Include testbench with duty cycles: 0%, 25%, 50%, 75%, 100%.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLK[Clock] --> CNTRL
    RST[Reset] --> CNTRL
    DUTY[Duty Cycle Input] --> DUTY_REG[[DUTY_REG]]
    DUTY_REG --> COMP
    CNTRL(Control Logic) --> CTR
    CTR[[8-bit Counter]] --> COMP{Comparator}
    COMP --> OUT[PWM Output]
    TB_DUTY["Test Values:<br>0%<br>25%<br>50%<br>75%<br>100%"] --> DUTY_REG
    CLK --> CTR
    RST --> CTR
    CLK --> DUTY_REG
    RST --> DUTY_REG
    CLK --> COMP
```
