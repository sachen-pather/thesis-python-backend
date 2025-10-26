# 4-bit Magnitude Comparator

**Category**: Combinational  
**Complexity**: simple

## Original Prompt

```
Design a 4-bit magnitude comparator with inputs a[3:0], b[3:0] and outputs equal, greater, less. Include testbench with boundary cases.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    A[a[3:0]] --> COMP((4-bit Comparator))
    B[b[3:0]] --> COMP
    COMP --> EQ[equal]
    COMP --> GT[greater]
    COMP --> LT[less]
    TB[Testbench] --> |Test Vectors| A
    TB --> |Test Vectors| B
    EQ --> |Expected 1 when a=b| CHECK{Verify}
    GT --> |Expected 1 when a>b| CHECK
    LT --> |Expected 1 when a<b| CHECK
    CHECK --> RESULT[Test Results]
    subgraph Test_Cases
        TC1[a=0000, b=0000]
        TC2[a=1111, b=1111]
        TC3[a=1010, b=0101]
        TC4[a=0011, b=1100]
    end
```
