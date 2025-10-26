# 2-bit Comparator

**Category**: Combinational  
**Complexity**: simple

## Original Prompt

```
Design a 2-bit comparator with inputs a[1:0], b[1:0] and outputs eq, gt, lt. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    A[a1:0] --> COMP((2-bit Comparator))
    B[b1:0] --> COMP
    COMP --> EQ[eq]
    COMP --> GT[gt]
    COMP --> LT[lt]
    TB[Testbench] --> |Test Vectors| A
    TB --> |Test Vectors| B
    EQ --> |Expected 1 when a=b| TB
    GT --> |Expected 1 when a>b| TB
    LT --> |Expected 1 when a<b| TB
    CLK[Clock] --> TB
    RST[Reset] --> TB
    TB --> |Verify| RESULT{Test Results}
```
