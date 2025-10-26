# services/analysis_service.py
from .unified_llm_service import UnifiedLLMService
import plotly.graph_objects as go
import pandas as pd
from io import StringIO

class AnalysisService:
    def __init__(self):
        self.llm_service = UnifiedLLMService()
    
    def analyze_waveform_with_ai(self, csv_data, verilog_code="", model="groq"):
        """AI-powered waveform analysis with model selection"""
        if not csv_data or "error" in csv_data:
            return "❌ No valid waveform data to analyze"
        
        # Get recommended model for analysis if auto selected
        if model == "auto":
            recommended_models = self.llm_service.get_recommended_models("waveform_analysis")
            model = recommended_models[0]
        
        # Limit data for LLM analysis
        lines = csv_data.split('\n')
        sample_data = '\n'.join(lines[:100])  # First 100 lines
        
        # Include Verilog code context if available
        code_context = ""
        if verilog_code:
            code_lines = verilog_code.split('\n')
            design_lines = []
            in_testbench = False
            
            for line in code_lines:
                if 'module testbench' in line.lower():
                    in_testbench = True
                elif line.strip().startswith('module ') and 'testbench' not in line.lower():
                    in_testbench = False
                    
                if not in_testbench and line.strip():
                    design_lines.append(line)
                    
            if design_lines:
                code_context = f"""

    **DESIGN CODE CONTEXT:**
    ```verilog
    {chr(10).join(design_lines[:50])}
    ```
                """
        
        prompt = f"""
    Analyze this Verilog simulation waveform for bugs or anomalies.

    **WAVEFORM DATA:**
    ```
    {sample_data}
    ```
    {code_context}

    **YOUR TASK:**
    Determine if this circuit has bugs or is working correctly.

    **ANALYSIS CHECKLIST:**
    1. Are outputs stuck at constant values when they shouldn't be?
    2. Do outputs change in response to inputs as expected?
    3. For sequential circuits: Does the counter increment? Does the register update?
    4. For combinational circuits: Do outputs match the expected logic (AND/OR/XOR/etc)?
    5. Are there any signals that never change despite input changes?

    **CRITICAL: You MUST start your response with exactly one of these:**
    - "VERDICT: NORMAL" - if the circuit is working correctly with no bugs
    - "VERDICT: ANOMALOUS" - if there are bugs, stuck signals, or incorrect behavior

    Then provide your detailed analysis.

    **EXAMPLES:**

    Example 1 - Working AND gate:
    VERDICT: NORMAL
    The AND gate outputs correctly respond to all input combinations. Output is 0 when either input is 0, and 1 only when both inputs are 1. All transitions are clean and expected.

    Example 2 - Stuck counter:
    VERDICT: ANOMALOUS  
    The counter output remains at 0000 despite clock edges. Expected behavior: counter should increment on each clock edge. This indicates a bug where the counter logic is not updating the count value.

    Example 3 - Working counter:
    VERDICT: NORMAL
    The counter increments correctly from 0000 to 1111 over multiple clock cycles. Reset behavior is correct. All state transitions follow expected sequential logic.

    Now analyze the provided waveform:
    """
        
        try:
            response = self.llm_service.invoke(prompt, model=model)
            return response
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    def create_waveform_plot(self, csv_data):
        """Create interactive waveform visualization"""
        if not csv_data or "error" in csv_data:
            return None
        
        try:
            # Parse CSV data
            df = pd.read_csv(StringIO(csv_data))
            
            if df.empty:
                return None
            
            # Create plotly figure
            fig = go.Figure()
            
            # Group by signal
            for signal in df['signal'].unique():
                signal_data = df[df['signal'] == signal].sort_values('timestamp')
                
                # Convert values to numeric where possible
                try:
                    y_values = pd.to_numeric(signal_data['value'], errors='coerce').fillna(0)
                except:
                    y_values = [1 if v == '1' else 0 for v in signal_data['value']]
                
                fig.add_trace(go.Scatter(
                    x=signal_data['timestamp'],
                    y=y_values,
                    mode='lines+markers',
                    name=signal,
                    line=dict(shape='hv')  # Square wave appearance
                ))
            
            fig.update_layout(
                title="Waveform Visualization",
                xaxis_title="Time (ns)",
                yaxis_title="Signal Value",
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
        except Exception as e:
            print(f"Plot generation failed: {str(e)}")
            return None
    
    def generate_waveform_stats(self, csv_data):
        """Generate waveform statistics"""
        try:
            df = pd.read_csv(StringIO(csv_data))
            
            if df.empty:
                return {}
            
            stats = {
                'signals': df['signal'].nunique(),
                'time_points': len(df),
                'duration_ns': df['timestamp'].max() if 'timestamp' in df.columns else 0,
                'frequency_hz': 0
            }
            
            if stats['duration_ns'] > 0:
                stats['frequency_hz'] = round(1000000000 / stats['duration_ns'], 1)
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}