# 3-to-8 Decoder

**Category**: Combinational  
**Complexity**: simple

## Original Prompt

```
Design a 3-to-8 decoder with inputs a[2:0], enable en, and outputs y[7:0]. When enabled, only one output is high based on input. Include testbench testing all combinations.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    A0[a[0]] --> DEC((3-to-8 Decoder))
    A1[a[1]] --> DEC
    A2[a[2]] --> DEC
    EN[enable] --> DEC
    DEC --> Y0[y[0]]
    DEC --> Y1[y[1]]
    DEC --> Y2[y[2]]
    DEC --> Y3[y[3]]
    DEC --> Y4[y[4]]
    DEC --> Y5[y[5]]
    DEC --> Y6[y[6]]
    DEC --> Y7[y[7]]
    TB[[Testbench]]
    TB --> A0
    TB --> A1
    TB --> A2
    TB --> EN
    Y0 --> TB
    Y1 --> TB
    Y2 --> TB
    Y3 --> TB
    Y4 --> TB
    Y5 --> TB
    Y6 --> TB
    Y7 --> TB
    TB --> VER{Verify One-Hot}
    VER --> PASS[Test Results]
```
