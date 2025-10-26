# Priority Encoder

**Category**: Combinational  
**Complexity**: medium

## Original Prompt

```
Design a 4-to-2 priority encoder with input in[3:0], output out[1:0], and valid bit. Highest bit has priority. Include testbench.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    IN0[in[0]] --> ENC((Priority Logic))
    IN1[in[1]] --> ENC
    IN2[in[2]] --> ENC
    IN3[in[3]] --> ENC
    ENC --> OUT[out[1:0]]
    ENC --> V[valid]
    TB[Testbench] -.-> IN0
    TB -.-> IN1
    TB -.-> IN2
    TB -.-> IN3
    OUT -.-> TB
    V -.-> TB
    subgraph Priority_Encoder
        ENC
    end
    subgraph Testbench_Block
        TB
    end
```
