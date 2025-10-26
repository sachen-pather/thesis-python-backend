from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ================== REQUEST MODELS ==================

class MermaidRequest(BaseModel):
    prompt: str = Field(..., description="Natural language description of the hardware design")
    session_id: Optional[str] = Field(None, description="Session identifier")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Additional generation options")

class MermaidRenderRequest(BaseModel):
    mermaid_code: str = Field(..., description="Mermaid diagram code to render")

class MermaidValidationRequest(BaseModel):
    mermaid_code: str = Field(..., description="Mermaid diagram code to validate")

class VerilogRequest(BaseModel):
    mermaid_code: str = Field(..., description="Mermaid diagram code")
    description: Optional[str] = Field(default="", description="Additional implementation details")
    session_id: Optional[str] = Field(None, description="Session identifier")
    model_choice: Optional[str] = Field(default="llama-3.3-70b-versatile", description="LLM model to use")

class VerilogValidationRequest(BaseModel):
    verilog_code: str = Field(..., description="Verilog code to validate")

class SimulationRequest(BaseModel):
    verilog_code: str = Field(..., description="Verilog code to simulate")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timeout: Optional[int] = Field(default=60, description="Simulation timeout in seconds")

class AnalysisRequest(BaseModel):
    waveform_csv: str = Field(..., description="Waveform data in CSV format")
    verilog_code: Optional[str] = Field(default="", description="Original Verilog code for context")
    session_id: Optional[str] = Field(None, description="Session identifier")

class WaveformPlotRequest(BaseModel):
    csv_data: str = Field(..., description="Waveform CSV data")

class DownloadRequest(BaseModel):
    content: str = Field(..., description="File content to download")
    filename: Optional[str] = Field(None, description="Suggested filename")

# ================== RESPONSE MODELS ==================

class BaseResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class MermaidResponse(BaseResponse):
    mermaid_code: Optional[str] = None
    session_id: Optional[str] = None
    validation_issues: Optional[List[str]] = None

class VerilogResponse(BaseResponse):
    verilog_code: Optional[str] = None
    validation_issues: Optional[List[str]] = None
    stats: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class SimulationResponse(BaseResponse):
    simulation_output: Optional[str] = None
    waveform_csv: Optional[str] = None
    compilation_log: Optional[str] = None
    simulation_time: Optional[float] = None
    session_id: Optional[str] = None

class AnalysisResponse(BaseResponse):
    analysis_results: Optional[str] = None
    session_id: Optional[str] = None

class FileDownloadResponse(BaseResponse):
    filename: Optional[str] = None
    download_url: Optional[str] = None
    file_size: Optional[int] = None

class SystemStatusResponse(BaseResponse):
    status: Optional[Dict[str, Any]] = None

# ================== SESSION MODELS ==================

class SessionData(BaseModel):
    session_id: str
    user_prompt: Optional[str] = ""
    generated_mermaid: Optional[str] = ""
    verified_mermaid: Optional[str] = ""
    verilog_code: Optional[str] = ""
    simulation_results: Optional[str] = ""
    waveform_csv: Optional[str] = ""
    compilation_log: Optional[str] = ""
    analysis_results: Optional[str] = ""
    simulation_time: Optional[float] = 0.0
    step: Optional[int] = 1
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    last_update: Optional[datetime] = Field(default_factory=datetime.now)

class SessionResponse(BaseResponse):
    session: Optional[SessionData] = None

# ================== UTILITY MODELS ==================

class ExampleDesign(BaseModel):
    name: str
    prompt: str
    category: str
    complexity: str
    mermaid_code: Optional[str] = None

class ExamplesResponse(BaseResponse):
    examples: Optional[List[ExampleDesign]] = None

class ValidationResult(BaseModel):
    valid: bool
    issues: List[str]
    stats: Optional[Dict[str, Any]] = None

class WaveformStats(BaseModel):
    signals: int
    time_points: int
    duration_ns: float
    frequency_hz: float

class PlotData(BaseModel):
    plot_json: str
    stats: WaveformStats