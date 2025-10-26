# Extracted Verilog Circuits

This directory contains 86 Verilog circuits extracted from the test suites.

## Summary

### SIMPLE Circuits: 26

- **Combinational - Normal**: 8 circuits
- **Combinational - Buggy**: 6 circuits
- **Sequential - Normal**: 4 circuits
- **Sequential - Buggy**: 4 circuits
- **Arithmetic - Normal**: 2 circuits
- **Arithmetic - Buggy**: 2 circuits

### MEDIUM Circuits: 48

- **Combinational - Normal**: 10 circuits
- **Combinational - Buggy**: 10 circuits
- **Sequential - Normal**: 8 circuits
- **Sequential - Buggy**: 8 circuits
- **Arithmetic - Normal**: 6 circuits
- **Arithmetic - Buggy**: 6 circuits

### COMPLEX Circuits: 12

- **State Machines - Normal**: 4 circuits
- **State Machines - Buggy**: 4 circuits
- **CPU Components - Normal**: 2 circuits
- **CPU Components - Buggy**: 2 circuits

**GRAND TOTAL: 86 circuits**

## Directory Structure

```
extracted_circuits/
├── simple/          (26 circuits)
├── medium/          (48 circuits)
└── complex/         (12 circuits)
```

## Usage

Each `.v` file contains:
- The main circuit module
- A complete testbench module
- Header comments with circuit metadata

To simulate a circuit:
```bash
iverilog -o output.vvp circuit_name.v
vvp output.vvp
gtkwave dump.vcd
```
