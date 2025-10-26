# 16x8 Dual-Port RAM

**Category**: CPU Component  
**Complexity**: complex

## Original Prompt

```
Create a 16x8 dual-port RAM with independent read/write ports. Port A: clk_a, we_a, addr_a[3:0], din_a[7:0], dout_a[7:0]. Port B: clk_b, we_b, addr_b[3:0], din_b[7:0], dout_b[7:0]. Include testbench with simultaneous operations.
```

## Generated Mermaid Diagram

```mermaid
graph LR
    CLKA[clk_a] --> PORTA[[Port A Logic]]
    WEA[we_a] --> PORTA
    ADDRA[addr_a[3:0]] --> PORTA
    DINA[din_a[7:0]] --> PORTA
    PORTA --> RAM[[16x8 Memory Array]]
    
    CLKB[clk_b] --> PORTB[[Port B Logic]]
    WEB[we_b] --> PORTB
    ADDRB[addr_b[3:0]] --> PORTB
    DINB[din_b[7:0]] --> PORTB
    PORTB --> RAM
    
    RAM --> DOUTA[dout_a[7:0]]
    RAM --> DOUTB[dout_b[7:0]]
    
    subgraph TestBench
        TBA[Test Port A] --> ADDRA
        TBA --> DINA
        TBA --> WEA
        TBB[Test Port B] --> ADDRB
        TBB --> DINB
        TBB --> WEB
        DOUTA --> VER{Verify}
        DOUTB --> VER
    end
```
