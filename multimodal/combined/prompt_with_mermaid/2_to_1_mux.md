# 2-to-1 MUX

**Category**: Combinational  
**Complexity**: simple

## Original Prompt

```
Design a 2-to-1 multiplexer with inputs a, b, select sel, and output out. Include testbench testing all combinations.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    A[Input A] --> MUX{2:1 MUX}
    B[Input B] --> MUX
    SEL[Select] --> MUX
    MUX --> OUT[Output]
    TB[[Testbench]]
    TB --"Test 1: sel=0,a=0,b=0"--> MUX
    TB --"Test 2: sel=0,a=0,b=1"--> MUX
    TB --"Test 3: sel=0,a=1,b=0"--> MUX
    TB --"Test 4: sel=0,a=1,b=1"--> MUX
    TB --"Test 5: sel=1,a=0,b=0"--> MUX
    TB --"Test 6: sel=1,a=0,b=1"--> MUX
    TB --"Test 7: sel=1,a=1,b=0"--> MUX
    TB --"Test 8: sel=1,a=1,b=1"--> MUX
```
