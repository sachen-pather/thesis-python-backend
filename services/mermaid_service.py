# services/mermaid_service.py - Updated with RAG support
from .unified_llm_service import UnifiedLLMService
#from .rag_service import VerilogRAGService
import re

class MermaidService:
    def __init__(self):
        self.llm_service = UnifiedLLMService()
        try:
            self.rag_service = VerilogRAGService()
            self.rag_enabled = True
        except Exception as e:
            print(f"RAG service initialization failed: {e}")
            self.rag_enabled = False
    
    def generate_mermaid_from_prompt(self, user_prompt: str, model: str = "groq", use_rag: bool = False):
        """Generate Mermaid diagram with optional RAG"""
        
        if use_rag and self.rag_enabled:
            # Use RAG to enhance the prompt
            enhanced_prompt = self.rag_service.enhance_prompt_with_rag(user_prompt, "mermaid")
            
            mermaid_prompt = f"""
            {enhanced_prompt}
            
            **GUIDELINES:**
            1. Use 'graph TD' (top-down) or 'graph LR' (left-right) syntax
            2. Create nodes representing hardware components/blocks
            3. Use arrows to show data flow or signal paths
            4. Follow the pattern from the examples above
            5. Keep it simple but accurate for hardware implementation
            6. Use appropriate node shapes:
               - [ ] for input/output blocks
               - ( ) for processing/logic blocks
               - {{ }} for decision/mux blocks
               - [[ ]] for memory/storage blocks
            7. Include clock and reset signals where appropriate
            8. Show control signals and data paths clearly
            9. Do not use anything that says note
            10. Avoid using the word "note" in the diagram
            
            **OUTPUT:** Return ONLY the Mermaid diagram code, no explanations or markdown blocks.
            """
        else:
            # Original prompt without RAG
            mermaid_prompt = f"""
            Convert this natural language description into a Mermaid diagram suitable for digital hardware design:
            
            USER REQUEST: "{user_prompt}"
            
            **GUIDELINES:**
            1. Use 'graph TD' (top-down) or 'graph LR' (left-right) syntax
            2. Create nodes representing hardware components/blocks
            3. Use arrows to show data flow or signal paths
            4. Include meaningful labels for inputs, outputs, and processing blocks
            5. Keep it simple but accurate for hardware implementation
            6. Use appropriate node shapes:
               - [ ] for input/output blocks
               - ( ) for processing/logic blocks
               - {{ }} for decision/mux blocks
               - [[ ]] for memory/storage blocks
            7. Include clock and reset signals where appropriate
            8. Show control signals and data paths clearly
            9. Do not use anything that says note
            10. Avoid using the word "note" in the diagram
            
            **OUTPUT:** Return ONLY the Mermaid diagram code, no explanations or markdown blocks.
            
            Example format:
            graph TD
                CLK[Clock] --> PROC[Processing Unit]
                RST[Reset] --> PROC
                A[Input Data] --> PROC
                PROC --> C[Output Register]
                CLK --> C
            """
        
        try:
            response = self.llm_service.invoke(mermaid_prompt, model=model)
            
            # Clean up any markdown artifacts
            mermaid_code = response.strip()
            if "```" in mermaid_code:
                lines = mermaid_code.split('\n')
                code_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or (not any(line.strip().startswith(x) for x in ['```', '#', '*', '-'])):
                        code_lines.append(line)
                
                mermaid_code = '\n'.join(code_lines).strip()
            
            return mermaid_code
        except Exception as e:
            return f"Error generating Mermaid diagram: {str(e)}"