# Full Adder

**Category**: Combinational  
**Complexity**: simple

## Original Prompt

```
Create a full adder with inputs a, b, cin and outputs sum, cout. Include testbench with all 8 cases.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    A[Input A] --> FA((Full Adder))
    B[Input B] --> FA
    CIN[Carry In] --> FA
    FA --> SUM[Sum Output]
    FA --> COUT[Carry Out]
    subgraph TestBench
        TB[Test Cases]
        T0[A=0 B=0 CIN=0 SUM=0 COUT=0]
        T1[A=0 B=0 CIN=1 SUM=1 COUT=0]
        T2[A=0 B=1 CIN=0 SUM=1 COUT=0]
        T3[A=0 B=1 CIN=1 SUM=0 COUT=1]
        T4[A=1 B=0 CIN=0 SUM=1 COUT=0]
        T5[A=1 B=0 CIN=1 SUM=0 COUT=1]
        T6[A=1 B=1 CIN=0 SUM=0 COUT=1]
        T7[A=1 B=1 CIN=1 SUM=1 COUT=1]
        TB --> T0
        TB --> T1
        TB --> T2
        TB --> T3
        TB --> T4
        TB --> T5
        TB --> T6
        TB --> T7
    end
```
