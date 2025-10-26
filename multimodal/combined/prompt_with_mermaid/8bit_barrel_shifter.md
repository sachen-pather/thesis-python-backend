# 8-bit Barrel Shifter

**Category**: Arithmetic  
**Complexity**: medium

## Original Prompt

```
Design an 8-bit barrel shifter supporting left shift, right shift, and rotate operations. Inputs: data[7:0], shift_amt[2:0], op[1:0] (00=LSL, 01=LSR, 10=ROL, 11=ROR). Output: result[7:0]. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    DATA[data[7:0]] --> SHIFT((Barrel Shifter))
    AMT[shift_amt[2:0]] --> SHIFT
    OP[op[1:0]] --> CTRL((Control Logic))
    CTRL --> SHIFT
    SHIFT --> RES[result[7:0]]
    
    TB[Testbench] --> |Test Data|DATA
    TB --> |Shift Amounts|AMT
    TB --> |Operations|OP
    RES --> VER{Verify Shift}
    VER --> PASS[Test Results]
    
    subgraph Operations
        LSL[00: Left Shift]
        LSR[01: Right Shift]
        ROL[10: Rotate Left]
        ROR[11: Rotate Right]
    end
    
    style SHIFT fill:#f9f,stroke:#333
    style CTRL fill:#bbf,stroke:#333
```
