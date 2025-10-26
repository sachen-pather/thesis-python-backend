# services/verilog_service.py - Updated with RAG support
from .unified_llm_service import UnifiedLLMService
from .rag_service import VerilogRAGService
from datetime import datetime
import re

class VerilogService:
    def __init__(self):
        self.llm_service = UnifiedLLMService()
        self.compilation_log = ""
        
        try:
            self.rag_service = VerilogRAGService()
            self.rag_enabled = True
        except Exception as e:
            print(f"RAG service initialization failed: {e}")
            self.rag_enabled = False
    
    def log_message(self, message, level="info"):
        """Add timestamped log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.compilation_log += f"[{timestamp}] {level.upper()}: {message}\n"

    def sanitize_verilog_code(self, verilog_code):
        """Fix common character encoding issues in LLM-generated Verilog"""
        correct_backtick = chr(96)
        
        wrong_chars = [
            chr(180), chr(8216), chr(8217), chr(8220), chr(8221),
            '`', '´', ''', '''
        ]
        
        sanitized_code = verilog_code
        for wrong_char in wrong_chars:
            if wrong_char in sanitized_code and 'timescale' in sanitized_code:
                sanitized_code = sanitized_code.replace(wrong_char + 'timescale', correct_backtick + 'timescale')
                sanitized_code = sanitized_code.replace(wrong_char + 'define', correct_backtick + 'define')
                sanitized_code = sanitized_code.replace(wrong_char + 'include', correct_backtick + 'include')
        
        sanitized_code = sanitized_code.encode('ascii', 'ignore').decode('ascii')
        sanitized_code = sanitized_code.replace('\r\n', '\n').replace('\r', '\n')
        sanitized_code = sanitized_code.strip()
        
        return sanitized_code

    def generate_verilog(self, mermaid_code, process_description="", model="groq", use_rag=False):
        """Generate Verilog with optional RAG"""
        
        if model == "auto":
            recommended_models = self.llm_service.get_recommended_models("verilog_generation")
            model = recommended_models[0]
        
        if use_rag and self.rag_enabled:
            # Create query for RAG from both mermaid and description
            rag_query = f"{process_description} {mermaid_code}".strip()
            if not rag_query:
                rag_query = "verilog module"
            
            enhanced_prompt = self.rag_service.enhance_prompt_with_rag(rag_query, "verilog")
            
            base_prompt = f"""
            {enhanced_prompt}
            
            Convert this Mermaid diagram to valid Verilog-2001 code that compiles with iverilog:
            {mermaid_code}
            
            Process Description: {process_description if process_description else "Simple pass-through with 1-clock delay"}
            
            CRITICAL SYNTAX REQUIREMENTS:
            1. Use ONLY standard Verilog-2001 keywords and syntax
            2. Use 'reg' for registers, 'wire' for nets
            3. All module ports must be properly declared with directions
            4. NO SystemVerilog constructs (logic, always_ff, etc.)
            5. Use proper always block sensitivity lists
            6. Avoid reserved keywords as signal names (like 'output')
            7. Use proper blocking/non-blocking assignments
            8. Follow the patterns shown in the examples above
            
            OUTPUT: Return ONLY the complete, valid Verilog code. No explanations, no markdown blocks.
            """
        else:
            # Original prompt without RAG
            base_prompt = f"""
            Convert this Mermaid diagram to valid Verilog-2001 code that compiles with iverilog:
            {mermaid_code}
            
            Process Description: {process_description if process_description else "Simple pass-through with 1-clock delay"}
            
            CRITICAL SYNTAX REQUIREMENTS:
            1. Use ONLY standard Verilog-2001 keywords and syntax
            2. Use 'reg' for registers, 'wire' for nets
            3. All module ports must be properly declared with directions
            4. NO SystemVerilog constructs (logic, always_ff, etc.)
            5. Use proper always block sensitivity lists
            6. Avoid reserved keywords as signal names (like 'output')
            7. Use proper blocking/non-blocking assignments
            
            FIXED EXAMPLE FOR 4-BIT COUNTER:
            ```
            module counter(
                input wire clk,
                input wire rst_n,
                input wire enable,
                output reg [3:0] count
            );
            
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    count <= 4'b0000;
                else if (enable)
                    count <= count + 1'b1;
            end
            
            endmodule
            
            module testbench;
                reg clk, rst_n, enable;
                wire [3:0] count;
                
                counter dut (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),
                    .count(count)
                );
                
                initial begin
                    $dumpfile("dump.vcd");
                    $dumpvars(0, testbench);
                    
                    clk = 1'b0;
                    rst_n = 1'b0;
                    enable = 1'b0;
                    
                    #20 rst_n = 1'b1;
                    #10 enable = 1'b1;
                    #100 enable = 1'b0;
                    #20 rst_n = 1'b0;
                    #20 rst_n = 1'b1;
                    #50 $finish;
                end
                
                always #5 clk = ~clk;
                
                initial begin
                    $monitor("Time=%0t clk=%b rst_n=%b enable=%b count=%d", 
                             $time, clk, rst_n, enable, count);
                end
            endmodule
            ```
            
            OUTPUT: Return ONLY the complete, valid Verilog code. No explanations, no markdown blocks, no extra text.
            """
        
        self.log_message(f"Starting Verilog generation with model: {model}, RAG: {use_rag}")
        try:
            verilog_code = self.llm_service.invoke(base_prompt, model=model)
            
            # Clean up any markdown artifacts
            if "```" in verilog_code:
                lines = verilog_code.split('\n')
                code_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or (not any(line.strip().startswith(x) for x in ['```', '#', '*', '-'])):
                        code_lines.append(line)
                
                verilog_code = '\n'.join(code_lines).strip()
            
            # Add timescale if not present
            if "`timescale" not in verilog_code and "timescale" not in verilog_code.lower():
                verilog_code = "`timescale 1ns/1ps\n\n" + verilog_code
            
            # Sanitize the code to fix backtick issues
            if "`" in verilog_code:
                verilog_code = self.sanitize_verilog_code(verilog_code)
            
            # Validate basic syntax
            validation_errors = self.validate_verilog_syntax(verilog_code)
            if validation_errors:
                self.log_message(f"Syntax validation warnings: {validation_errors}", "warning")
            
            # Attempt auto-fix if there are critical errors
            if any("CRITICAL" in str(error) for error in validation_errors):
                self.log_message("Critical errors detected, attempting auto-fix...", "warning")
                verilog_code = self.fix_verilog_syntax_errors(verilog_code)
                validation_errors = self.validate_verilog_syntax(verilog_code)
                self.log_message(f"Post-fix validation: {len(validation_errors)} issues remain")
            
            self.log_message(f"Verilog generation completed successfully with {model}")
            return verilog_code
            
        except Exception as e:
            self.log_message(f"Verilog generation failed with {model}: {str(e)}", "error")
            raise e

    def validate_verilog_syntax(self, verilog_code):
        """Verilog syntax validation"""
        errors = []
        warnings = []
        lines = verilog_code.split('\n')
        
        # Check for timescale
        has_timescale = any('`timescale' in line for line in lines)
        if not has_timescale:
            warnings.append("INFO: No `timescale directive")
        
        # Check for balanced begin/end and module/endmodule
        begin_count = sum(1 for line in lines if 'begin' in line.lower() and not line.strip().startswith('//'))
        end_count = sum(1 for line in lines if line.strip().lower() == 'end' or 'end;' in line.lower())
        
        module_count = sum(1 for line in lines if line.strip().startswith('module '))
        endmodule_count = sum(1 for line in lines if 'endmodule' in line.lower())
        
        if begin_count != end_count:
            errors.append(f"CRITICAL: Unbalanced begin/end blocks: {begin_count} begin, {end_count} end")
        
        if module_count != endmodule_count:
            errors.append(f"CRITICAL: Unbalanced module/endmodule: {module_count} module, {endmodule_count} endmodule")
        
        # Check for reserved keywords used as signal names
        reserved_keywords = ['input', 'output', 'reg', 'wire', 'module', 'endmodule', 'begin', 'end', 
                            'always', 'initial', 'if', 'else', 'case', 'default', 'for', 'while']
        
        for i, line in enumerate(lines, 1):
            line_clean = line.strip()
            if line_clean and not line_clean.startswith('//'):
                for keyword in reserved_keywords:
                    if f' {keyword}' in line or f'.{keyword}(' in line or f'[{keyword}]' in line:
                        if any(pattern in line for pattern in [f'output reg {keyword}', f'output wire {keyword}', 
                                                              f'.{keyword}({keyword})', f'wire {keyword}']):
                            errors.append(f"CRITICAL: Line {i}: Using reserved keyword '{keyword}' as signal name")
                
                if 'always @*(' in line:
                    errors.append(f"CRITICAL: Line {i}: Invalid always block syntax")
                
                if 'logic' in line.lower() or 'always_ff' in line.lower():
                    errors.append(f"CRITICAL: Line {i}: SystemVerilog syntax detected")
                
                if '= 0;' in line or '= 1;' in line:
                    warnings.append(f"WARNING: Line {i}: Consider using proper bit widths")
        
        return errors + warnings
    
    def fix_verilog_syntax_errors(self, verilog_code):
        """Automatically fix common Verilog syntax errors"""
        self.log_message("Attempting to auto-fix syntax errors...")
        
        # Fix reserved keyword 'output' used as signal name
        if 'output reg output' in verilog_code or 'output wire output' in verilog_code:
            self.log_message("Fixing: Reserved keyword 'output' used as signal name")
            verilog_code = verilog_code.replace('output reg output', 'output reg count')
            verilog_code = verilog_code.replace('output wire output', 'output wire count')
            verilog_code = verilog_code.replace('.output(output)', '.count(count)')
            verilog_code = verilog_code.replace('output <= ', 'count <= ')
            verilog_code = verilog_code.replace('output + ', 'count + ')
            verilog_code = verilog_code.replace('output=%h', 'count=%h')
            verilog_code = verilog_code.replace(', output)', ', count)')
        
        # Fix always block syntax
        verilog_code = verilog_code.replace('always @*(posedge', 'always @(posedge')
        
        # Add proper bit widths to constants
        verilog_code = verilog_code.replace(' = 0;', ' = 1\'b0;')
        verilog_code = verilog_code.replace(' = 1;', ' = 1\'b1;')
        verilog_code = verilog_code.replace('<= 0;', '<= 4\'b0000;')
        verilog_code = verilog_code.replace(' + 1;', ' + 1\'b1;')
        
        self.log_message("Auto-fix completed")
        return verilog_code
    
    def get_compilation_log(self):
        """Get the compilation log"""
        return self.compilation_log
    
    def clear_compilation_log(self):
        """Clear the compilation log"""
        self.compilation_log = ""