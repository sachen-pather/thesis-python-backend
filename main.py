# main.py - Complete Fixed FastAPI Application
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from datetime import datetime
import tempfile
import uuid
from typing import List, Optional
from pydantic import BaseModel
import sys

# Import your services - update these imports based on your actual file structure
from services.mermaid_service import MermaidService
from services.verilog_service import VerilogService
from services.analysis_service import AnalysisService
from services.simulation_service import SimulationService
from services.session_service import SessionService

try:
    from services.rag_service import VerilogRAGService
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# If you have the new unified service, use this:
# from services.unified_llm_service import UnifiedLLMService

# If not, keep using the original services
# You can implement the unified service later

# VerilogEval services (if available)
try:
    from services.verilogeval_service import VerilogEvalService
    VERILOGEVAL_AVAILABLE = True
except ImportError:
    VERILOGEVAL_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Enhanced Verilog Generation System API",
    description="AI-Powered Hardware Design Pipeline with Multiple Model Support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for temporary files
os.makedirs("static/temp", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
mermaid_service = MermaidService()
verilog_service = VerilogService()
analysis_service = AnalysisService()
simulation_service = SimulationService()
session_service = SessionService()

if VERILOGEVAL_AVAILABLE:
    verilogeval_service = VerilogEvalService()

# Initialize unified LLM service if available
try:
    from services.unified_llm_service import UnifiedLLMService
    llm_service = UnifiedLLMService()
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False

# ================== API MODELS ==================

class MermaidRequest(BaseModel):
    prompt: str
    model: Optional[str] = "groq"  # Default to groq
    session_id: Optional[str] = None

class VerilogRequest(BaseModel):
    mermaid_code: str
    description: Optional[str] = ""
    model: Optional[str] = "groq"  # Default to groq
    use_rag: Optional[bool] = False
    session_id: Optional[str] = None

class SimulationRequest(BaseModel):
    verilog_code: str
    model: Optional[str] = "groq"  # ✅ ADD THIS LINE
    session_id: Optional[str] = None
    
class AnalysisRequest(BaseModel):
    waveform_csv: str
    verilog_code: Optional[str] = ""
    user_prompt: Optional[str] = ""
    model: Optional[str] = "groq"  # Default to groq
    session_id: Optional[str] = None

class ModelTestRequest(BaseModel):
    model: str

class BatchGenerationRequest(BaseModel):
    prompt: str
    models: List[str]
    session_id: Optional[str] = None
    
class VerilogEvalRequest(BaseModel):
    verilog_code: str
    waveform_csv: Optional[str] = ""
    user_prompt: Optional[str] = ""
    mermaid_code: Optional[str] = ""
    use_rag: Optional[bool] = False  # Add this line
    session_id: Optional[str] = None

class BenchmarkRequest(BaseModel):
    verilog_code: str
    user_prompt: Optional[str] = ""

class PassAtKRequest(BaseModel):
    verilog_codes: List[str]
    k: Optional[int] = 1
    
class RAGMermaidRequest(BaseModel):
    prompt: str
    model: Optional[str] = "claude"
    use_rag: bool = True
    session_id: Optional[str] = None

class AddRAGExampleRequest(BaseModel):
    description: str
    mermaid: str
    verilog: str

# ================== MODEL MANAGEMENT ENDPOINTS ==================

@app.get("/api/models/available")
async def get_available_models():
    """Get all available AI models"""
    try:
        if MULTI_MODEL_AVAILABLE:
            models = llm_service.get_available_models()
            
            # Add metadata for frontend display
            model_list = []
            for key, info in models.items():
                model_list.append({
                    "id": key,
                    "name": info["name"],
                    "provider": info["provider"],
                    "description": info["description"],
                    "supports_vision": info.get("supports_vision", False),
                    "context_length": info.get("context_length", 0),
                    "best_for": info.get("best_for", []),
                    "free": True  # All these models are free
                })
        else:
            # Fallback to just Groq if multi-model not available
            model_list = [{
                "id": "groq",
                "name": "Llama 3.3 70B Versatile",
                "provider": "Groq",
                "description": "Fast and reliable model via Groq",
                "supports_vision": False,
                "context_length": 32768,
                "best_for": ["general", "verilog", "analysis"],
                "free": True
            }]
        
        return {
            "success": True,
            "models": model_list,
            "total_count": len(model_list)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/models/recommendations/{task_type}")
async def get_model_recommendations(task_type: str):
    """Get recommended models for specific tasks"""
    try:
        if MULTI_MODEL_AVAILABLE:
            recommended = llm_service.get_recommended_models(task_type)
            models = llm_service.get_available_models()
            
            recommendations = []
            for model_id in recommended:
                if model_id in models:
                    model_info = models[model_id]
                    recommendations.append({
                        "id": model_id,
                        "name": model_info["name"],
                        "provider": model_info["provider"],
                        "reason": f"Optimized for {task_type.replace('_', ' ')}"
                    })
        else:
            # Fallback recommendation
            recommendations = [{
                "id": "groq",
                "name": "Llama 3.3 70B Versatile",
                "provider": "Groq",
                "reason": f"Reliable for {task_type.replace('_', ' ')}"
            }]
        
        return {
            "success": True,
            "task_type": task_type,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/models/test")
async def test_model(request: ModelTestRequest):
    """Test if a specific model is working"""
    try:
        if MULTI_MODEL_AVAILABLE:
            result = llm_service.test_model(request.model)
        else:
            # Simple test for Groq
            if request.model == "groq":
                result = {"success": True, "model": "groq", "response": "Model working"}
            else:
                result = {"success": False, "model": request.model, "error": "Model not available"}
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "model": request.model,
            "error": str(e)
        }

# ================== DESIGN ENDPOINTS ==================

@app.post("/api/design/generate-mermaid")
async def generate_mermaid(request: MermaidRequest):
    """Generate Mermaid diagram with optional model selection"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Generate mermaid code
        if hasattr(mermaid_service, 'generate_mermaid_from_prompt'):
            # If your service supports model parameter
            try:
                mermaid_code = mermaid_service.generate_mermaid_from_prompt(
                    request.prompt, 
                    model=request.model if MULTI_MODEL_AVAILABLE else None
                )
            except TypeError:
                # Fallback if model parameter not supported
                mermaid_code = mermaid_service.generate_mermaid_from_prompt(request.prompt)
        else:
            return {"success": False, "error": "Mermaid service not properly configured"}
        
        if mermaid_code.startswith("Error"):
            return {
                "success": False,
                "error": mermaid_code,
                "session_id": session_id
            }
        
        # Update session if session service is available
        if hasattr(session_service, 'update_session'):
            session_service.update_session(session_id, {
                'user_prompt': request.prompt,
                'selected_model': request.model,
                'generated_mermaid': mermaid_code,
                'verified_mermaid': mermaid_code,
                'step': 2,
                'last_update': datetime.now()
            })
        
        return {
            "success": True,
            "mermaid_code": mermaid_code,
            "model_used": request.model,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id or str(uuid.uuid4())
        }

@app.post("/api/design/generate-verilog")
async def generate_verilog(request: VerilogRequest):
    """Generate Verilog code with optional model selection"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Generate verilog code
        if hasattr(verilog_service, 'generate_verilog'):
            try:
                verilog_code = verilog_service.generate_verilog(
                    request.mermaid_code, 
                    request.description or "",
                    model=request.model if MULTI_MODEL_AVAILABLE else None,
                    use_rag=request.use_rag                    
                )
            except TypeError:
                # Fallback if model parameter not supported
                verilog_code = verilog_service.generate_verilog(
                    request.mermaid_code, 
                    request.description or ""
                )
        else:
            return {"success": False, "error": "Verilog service not properly configured"}
        
        # Validate syntax if validation method exists
        validation_issues = []
        if hasattr(verilog_service, 'validate_verilog_syntax'):
            validation_issues = verilog_service.validate_verilog_syntax(verilog_code)
        
        lines = verilog_code.split('\n')
        stats = {
            'lines': len(lines),
            'modules': verilog_code.count('module '),
            'always_blocks': verilog_code.count('always'),
            'has_timescale': '`timescale' in verilog_code,
            'has_testbench': 'testbench' in verilog_code.lower(),
            'model_used': request.model
        }
        
        # Update session if session service is available
        if hasattr(session_service, 'update_session'):
            session_service.update_session(session_id, {
                'verified_mermaid': request.mermaid_code,
                'verilog_code': verilog_code,
                'verilog_model': request.model,
                'validation_issues': validation_issues,
                'step': 3,
                'last_update': datetime.now()
            })
        
        return {
            "success": True,
            "verilog_code": verilog_code,
            "validation_issues": validation_issues,
            "stats": stats,
            "model_used": request.model,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id or str(uuid.uuid4())
        }

# ================== SIMULATION ENDPOINTS ==================

@app.post("/api/simulation/run")
async def run_simulation(request: SimulationRequest):
    """Run Verilog simulation with iverilog"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Run simulation
        if hasattr(simulation_service, 'simulate_verilog'):
            success, sim_output, csv_data, error = simulation_service.simulate_verilog(
                request.verilog_code
            )
        else:
            return {"success": False, "error": "Simulation service not properly configured"}
        
        # Get additional info if methods exist
        compilation_log = ""
        simulation_time = 0
        
        if hasattr(simulation_service, 'get_compilation_log'):
            compilation_log = simulation_service.get_compilation_log()
        
        if hasattr(simulation_service, 'get_last_simulation_time'):
            simulation_time = simulation_service.get_last_simulation_time()
        
        # Update session if session service is available
        if hasattr(session_service, 'update_session'):
            session_service.update_session(session_id, {
                'verilog_code': request.verilog_code,
                'simulation_results': sim_output,
                'waveform_csv': csv_data,
                'compilation_log': compilation_log,
                'simulation_time': simulation_time,
                'step': 4,
                'last_update': datetime.now()
            })
        
        return {
            "success": success,
            "simulation_output": sim_output,
            "waveform_csv": csv_data,
            "compilation_log": compilation_log,
            "simulation_time": simulation_time,
            "error": error,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id or str(uuid.uuid4())
        }
        
@app.post("/api/verification/vae")
async def vae_verification_only(request: dict):
    """Run VAE verification on provided Verilog code"""
    try:
        verilog_code = request.get("verilog_code", "")
        session_id = request.get("session_id", str(uuid.uuid4()))
        
        if not verilog_code:
            return {"success": False, "error": "No Verilog code provided"}
        
        # Import and run VAE verification
        from tests.vae.use_vae import verify_verilog_waveform
        
        is_anomalous, vae_error, vae_message = verify_verilog_waveform(verilog_code)
        
        return {
            "success": True,
            "vae_verification": {
                "is_anomalous": is_anomalous,
                "error": vae_error,
                "message": vae_message,
                "threshold": 0.044623,  # Your trained threshold
                "confidence": "high" if abs(vae_error - 0.044623) > 0.02 else "medium"
            },
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"VAE verification failed: {str(e)}",
            "session_id": request.get("session_id", str(uuid.uuid4()))
        }
        

@app.post("/api/simulation/run-with-verification")
async def run_simulation_with_verification(request: SimulationRequest):
    """Run Verilog simulation with both LLM and VAE verification"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Step 1: Run simulation
        if hasattr(simulation_service, 'simulate_verilog'):
            success, sim_output, csv_data, error = simulation_service.simulate_verilog(
                request.verilog_code
            )
        else:
            return {"success": False, "error": "Simulation service not properly configured"}
        
        # Get additional simulation info
        compilation_log = ""
        simulation_time = 0
        
        if hasattr(simulation_service, 'get_compilation_log'):
            compilation_log = simulation_service.get_compilation_log()
        
        if hasattr(simulation_service, 'get_last_simulation_time'):
            simulation_time = simulation_service.get_last_simulation_time()
        
        verification_results = {}
        
           # Step 2: VAE Verification (if simulation successful and CSV data available)
               # Step 2: VAE Verification (if simulation successful and CSV data available)
        if success and csv_data and len(csv_data) > 50:
            try:
                # Add the tests/vae directory to Python path
                vae_dir = os.path.join(os.path.dirname(__file__), 'tests', 'vae')
                if vae_dir not in sys.path:
                    sys.path.insert(0, vae_dir)
                
                from api_integration_vae import HybridVerificationService
                
                hybrid_verifier = HybridVerificationService()
                result = hybrid_verifier.verify_circuit_comprehensive(request.verilog_code)
                
                if result["success"] and result["vae_verification"]["available"]:
                    vae_verif = result["vae_verification"]
                    verification_results['vae_verification'] = {
                        'available': True,
                        'is_anomalous': bool(vae_verif['is_anomalous']),
                        'confidence': float(vae_verif['confidence']),
                        'message': str(vae_verif['message']),
                        'method': 'hybrid'
                    }
                else:
                    verification_results['vae_verification'] = {
                        'available': False,
                        'error': result.get('error', 'VAE initialization failed')
                    }
                    
            except Exception as vae_e:
                verification_results['vae_verification'] = {
                    'available': False,
                    'error': str(vae_e),
                    'message': 'VAE verification failed'
                }
        else:
            verification_results['vae_verification'] = {
                'available': False,
                'error': "No valid waveform data for VAE analysis",
                'message': "Simulation must produce valid CSV waveform data"
            }
        
        # Step 3: LLM Verification (if analysis service available)
        if success and csv_data and hasattr(analysis_service, 'analyze_waveform_with_ai'):
            try:
                llm_analysis = analysis_service.analyze_waveform_with_ai(
                    csv_data,
                    request.verilog_code,
                    model=request.model  # ✅ USE MODEL FROM REQUEST
                )
                verification_results['llm_verification'] = {
                    'available': True,
                    'analysis': llm_analysis,
                    'message': "LLM analysis completed"
                }
                
            except Exception as llm_e:
                verification_results['llm_verification'] = {
                    'available': False,
                    'error': f"LLM verification failed: {str(llm_e)}",
                    'message': "LLM analysis service failed"
                }
        else:
            verification_results['llm_verification'] = {
                'available': False,
                'error': "Analysis service not available or no waveform data",
                'message': "LLM verification requires analysis service and valid waveform"
            }
        
        # Step 4: Combined Assessment
        overall_assessment = {
    'simulation_success': bool(success),
    'has_waveform_data': bool(csv_data and len(csv_data) > 50),
    'vae_available': bool(verification_results['vae_verification']['available']),
    'llm_available': bool(verification_results['llm_verification']['available']),
    'overall_status': 'unknown'
}
        
        # Determine overall status
        if not success:
            overall_assessment['overall_status'] = 'simulation_failed'
        elif not overall_assessment['has_waveform_data']:
            overall_assessment['overall_status'] = 'no_waveform_data'
        elif verification_results['vae_verification']['available']:
            if verification_results['vae_verification']['is_anomalous']:
                overall_assessment['overall_status'] = 'anomalous_waveform'
            else:
                overall_assessment['overall_status'] = 'normal_waveform'
        else:
            overall_assessment['overall_status'] = 'verification_unavailable'
        
        # Update session
        if hasattr(session_service, 'update_session'):
            session_service.update_session(session_id, {
                'verilog_code': request.verilog_code,
                'simulation_results': sim_output,
                'waveform_csv': csv_data,
                'compilation_log': compilation_log,
                'simulation_time': simulation_time,
                'verification_results': verification_results,
                'overall_assessment': overall_assessment,
                'step': 4,
                'last_update': datetime.now()
            })
        
        return {
            "success": success,
            "simulation_output": sim_output,
            "waveform_csv": csv_data,
            "compilation_log": compilation_log,
            "simulation_time": simulation_time,
            "error": error,
            "verification": verification_results,
            "assessment": overall_assessment,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id or str(uuid.uuid4())
        }

# Enhanced health check with VAE status
@app.get("/api/health/verification")
async def verification_health_check():
    """Check status of all verification systems"""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "verification_systems": {}
        }
        
        # Check VAE availability
        try:
            # Add the tests/vae directory to Python path
            vae_dir = os.path.join(os.path.dirname(__file__), 'tests', 'vae')
            if vae_dir not in sys.path:
                sys.path.insert(0, vae_dir)
            
            from use_vae import load_vae_model
            model, threshold, device = load_vae_model()
            status["verification_systems"]["vae"] = {
                "available": True,
                "model_loaded": True,
                "threshold": threshold,
                "device": str(device),
                "message": "VAE verification ready"
            }
        except Exception as e:
            status["verification_systems"]["vae"] = {
                "available": False,
                "error": str(e),
                "message": "VAE verification not available"
            }
        
        # Check LLM analysis
        llm_available = hasattr(analysis_service, 'analyze_waveform_with_ai')
        status["verification_systems"]["llm"] = {
            "available": llm_available,
            "message": "LLM analysis ready" if llm_available else "LLM analysis not configured"
        }
        
        # Check simulation
        sim_available = hasattr(simulation_service, 'simulate_verilog')
        status["verification_systems"]["simulation"] = {
            "available": sim_available,
            "message": "Simulation ready" if sim_available else "Simulation service not configured"
        }
        
        return status
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ================== ANALYSIS ENDPOINTS ==================

@app.post("/api/analysis/analyze-waveform")
async def analyze_waveform(request: AnalysisRequest):
    """Analyze waveform data with optional model selection"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Analyze waveform
        if hasattr(analysis_service, 'analyze_waveform_with_ai'):
            try:
                analysis_results = analysis_service.analyze_waveform_with_ai(
                    request.waveform_csv,
                    request.verilog_code or "",
                    model=request.model if MULTI_MODEL_AVAILABLE else None
                )
            except TypeError:
                # Fallback if model parameter not supported
                analysis_results = analysis_service.analyze_waveform_with_ai(
                    request.waveform_csv,
                    request.verilog_code or ""
                )
        else:
            return {"success": False, "error": "Analysis service not properly configured"}
        
        # Update session if session service is available
        if hasattr(session_service, 'update_session'):
            session_service.update_session(session_id, {
                'analysis_results': analysis_results,
                'analysis_model': request.model,
                'last_update': datetime.now()
            })
        
        return {
            "success": True,
            "analysis_results": analysis_results,
            "model_used": request.model,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id or str(uuid.uuid4())
        }

# ================== BATCH GENERATION (if multi-model available) ==================

@app.post("/api/design/batch-generation")
async def batch_generation(request: BatchGenerationRequest):
    """Generate designs using multiple models for comparison"""
    if not MULTI_MODEL_AVAILABLE:
        return {
            "success": False,
            "error": "Multi-model support not available",
            "session_id": request.session_id or str(uuid.uuid4())
        }
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Validate all requested models
        available_models = llm_service.get_available_models()
        invalid_models = [m for m in request.models if m not in available_models]
        if invalid_models:
            return {
                "success": False,
                "error": f"Invalid models: {invalid_models}",
                "session_id": session_id
            }
        
        results = {}
        
        for model in request.models:
            try:
                # Generate Mermaid diagram
                mermaid_code = mermaid_service.generate_mermaid_from_prompt(
                    request.prompt, 
                    model=model
                )
                
                # Generate Verilog code
                verilog_code = verilog_service.generate_verilog(
                    mermaid_code, 
                    request.prompt,
                    model=model
                )
                
                # Basic validation
                validation_issues = []
                if hasattr(verilog_service, 'validate_verilog_syntax'):
                    validation_issues = verilog_service.validate_verilog_syntax(verilog_code)
                
                results[model] = {
                    "success": True,
                    "mermaid_code": mermaid_code,
                    "verilog_code": verilog_code,
                    "validation_issues": validation_issues,
                    "stats": {
                        "lines": len(verilog_code.split('\n')),
                        "modules": verilog_code.count('module '),
                        "critical_issues": len([e for e in validation_issues if "CRITICAL" in str(e)])
                    }
                }
                
            except Exception as e:
                results[model] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Update session if session service is available
        if hasattr(session_service, 'update_session'):
            session_service.update_session(session_id, {
                'batch_results': results,
                'batch_prompt': request.prompt,
                'batch_models': request.models,
                'last_update': datetime.now()
            })
        
        return {
            "success": True,
            "results": results,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id or str(uuid.uuid4())
        }

# ================== SYSTEM STATUS ==================

@app.get("/api/health")
async def health_check():
    """Enhanced health check with model availability"""
    try:
        # Check available models
        model_count = 0
        working_models = []
        
        if MULTI_MODEL_AVAILABLE:
            models = llm_service.get_available_models()
            model_count = len(models)
            
            # Test a few key models
            test_models = ["groq", "qwen-2.5-coder-32b", "deepseek-r1-0528"]
            for model in test_models:
                if model in models:
                    test_result = llm_service.test_model(model)
                    if test_result.get("success"):
                        working_models.append(model)
        else:
            model_count = 1
            working_models = ["groq"]
        
        # Check simulation availability
        iverilog_available = False
        if hasattr(simulation_service, 'check_iverilog_available'):
            iverilog_available = simulation_service.check_iverilog_available()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "groq_api": bool(os.getenv("GROQ_API_KEY")),
                "openrouter_api": bool(os.getenv("OPENROUTER_API_KEY")),
                "multi_model": MULTI_MODEL_AVAILABLE,
                "iverilog": iverilog_available,
                "verilogeval": VERILOGEVAL_AVAILABLE
            },
            "models": {
                "total_available": model_count,
                "working_models": working_models,
                "multi_model_support": MULTI_MODEL_AVAILABLE
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/system/status")
async def get_system_status():
    """Get detailed system status"""
    try:
        status_info = {
            "multi_model_support": MULTI_MODEL_AVAILABLE,
            "verilogeval_support": VERILOGEVAL_AVAILABLE,
            "total_models": 1,  # Default to Groq
        }
        
        if MULTI_MODEL_AVAILABLE:
            models = llm_service.get_available_models()
            providers = list(set([info["provider"] for info in models.values()]))
            
            # Test each provider
            provider_status = {}
            for provider in providers:
                provider_models = [k for k, v in models.items() if v["provider"] == provider]
                if provider_models:
                    test_model = provider_models[0]
                    test_result = llm_service.test_model(test_model)
                    provider_status[provider] = {
                        "available": test_result.get("success", False),
                        "models_count": len(provider_models),
                        "test_model": test_model
                    }
            
            status_info.update({
                "providers": provider_status,
                "total_models": len(models),
            })
        
        # Check simulation
        iverilog_info = {"available": False, "version": "unknown"}
        if hasattr(simulation_service, 'check_iverilog_available'):
            iverilog_info["available"] = simulation_service.check_iverilog_available()
            if hasattr(simulation_service, 'get_iverilog_version'):
                iverilog_info["version"] = simulation_service.get_iverilog_version()
        
        status_info["iverilog"] = iverilog_info
        
        # VerilogEval info
        if VERILOGEVAL_AVAILABLE:
            status_info["verilogeval"] = {
                "available": True,
                "features": ["comprehensive_evaluation", "benchmark_comparison", "quality_metrics"]
            }
        
        return {
            "success": True,
            "status": status_info
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# Add these endpoints to your main.py file after the existing endpoints

# ================== VERILOGEVAL ENDPOINTS ==================

@app.get("/api/verilogeval/standards")
async def get_verilogeval_standards():
    """Get VerilogEval standards and compliance rules"""
    if not VERILOGEVAL_AVAILABLE:
        raise HTTPException(status_code=404, detail="VerilogEval service not available")
    
    try:
        # Return standard VerilogEval compliance patterns
        standards = {
            "syntax_rules": [
                "Proper module...endmodule structure",
                "Correct sensitivity lists in always blocks",
                "Appropriate use of blocking vs non-blocking assignments",
                "Proper clock and reset patterns",
                "Timescale directive present"
            ],
            "functional_tests": [
                "Code compiles without errors",
                "Simulation executes successfully", 
                "Reset behavior verification",
                "Clock edge behavior verification",
                "Waveform generation"
            ],
            "design_quality": [
                "Modular design structure",
                "Meaningful signal names",
                "Adequate code comments",
                "Design pattern recognition",
                "Alignment with specifications"
            ]
        }
        
        return {"success": True, "standards": standards}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/verilogeval/patterns/{category}")
async def get_design_patterns(category: str):
    """Get design patterns for specific categories"""
    if not VERILOGEVAL_AVAILABLE:
        raise HTTPException(status_code=404, detail="VerilogEval service not available")
    
    try:
        patterns = {
            "combinational": [
                {
                    "name": "Multiplexer",
                    "pattern": r"case\s*\(\s*\w+\s*\)",
                    "description": "Case statement for mux logic"
                },
                {
                    "name": "Decoder", 
                    "pattern": r"assign\s+\w+\s*=\s*\(",
                    "description": "Assign statements for decoder outputs"
                }
            ],
            "sequential": [
                {
                    "name": "Counter",
                    "pattern": r"\w+\s*<=\s*\w+\s*\+\s*1",
                    "description": "Increment pattern for counters"
                },
                {
                    "name": "Register",
                    "pattern": r"always\s*@\s*\(\s*posedge\s+\w+\s*\)",
                    "description": "Clocked always block"
                }
            ],
            "advanced": [
                {
                    "name": "State Machine",
                    "pattern": r"state\s*<=\s*\w+",
                    "description": "State transition pattern"
                }
            ]
        }
        
        return {
            "success": True, 
            "patterns": patterns.get(category, []),
            "category": category
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/comprehensive")
async def comprehensive_evaluation(request: dict):
    """Run comprehensive VerilogEval assessment"""
    if not VERILOGEVAL_AVAILABLE:
        raise HTTPException(status_code=404, detail="VerilogEval service not available")
    
    try:
        verilog_code = request.get("verilog_code", "")
        waveform_csv = request.get("waveform_csv", "")
        user_prompt = request.get("user_prompt", "")
        mermaid_code = request.get("mermaid_code", "")
        session_id = request.get("session_id", str(uuid.uuid4()))
        
        if not verilog_code:
            return {"success": False, "error": "No Verilog code provided"}
        
        # Run comprehensive evaluation
        eval_result = verilogeval_service.evaluate_verilog_comprehensive(
            verilog_code, user_prompt, mermaid_code
        )
        
        # Get benchmark comparison
        benchmark_result = verilogeval_service.get_benchmark_comparison(
            verilog_code, user_prompt
        )
        
        # Calculate overall score
        overall_score = (
            eval_result.syntax_score + 
            eval_result.functional_score + 
            eval_result.benchmark_score
        )
        
        # Generate comprehensive report
        comprehensive_report = f"""VerilogEval Comprehensive Assessment

SCORES:
- Syntax Score: {eval_result.syntax_score}/30 ({eval_result.syntax_score/30*100:.1f}%)
- Functional Score: {eval_result.functional_score}/40 ({eval_result.functional_score/40*100:.1f}%)
- Design Quality Score: {eval_result.benchmark_score}/30 ({eval_result.benchmark_score/30*100:.1f}%)
- Overall Score: {overall_score}/100 ({overall_score:.1f}%)

COMPLEXITY: {eval_result.complexity_rating}

TEST RESULTS:
- Tests Passed: {eval_result.passed_tests}/{eval_result.total_tests}
- Pass Rate: {eval_result.passed_tests/max(eval_result.total_tests,1)*100:.1f}%

ISSUES FOUND:
{chr(10).join(f"- {issue}" for issue in eval_result.issues)}

RECOMMENDATIONS:
{chr(10).join(f"- {rec}" for rec in eval_result.recommendations)}

BENCHMARK COMPARISON:
- Problem Type: {benchmark_result.get('problem_type', 'Unknown')}
- Pattern Match Score: {benchmark_result.get('pattern_match_score', 0):.1f}%
- Similar Problems: {', '.join(benchmark_result.get('similar_hdlbits_problems', []))}
"""
        
        evaluation_data = {
            "verilogeval_result": {
                "functional_score": eval_result.functional_score,
                "syntax_score": eval_result.syntax_score,
                "benchmark_score": eval_result.benchmark_score,
                "passed_tests": eval_result.passed_tests,
                "total_tests": eval_result.total_tests,
                "issues": eval_result.issues,
                "complexity_rating": eval_result.complexity_rating
            },
            "benchmark_comparison": benchmark_result,
            "overall_score": overall_score,
            "comprehensive_report": comprehensive_report,
            "recommendations": eval_result.recommendations,
            "session_id": session_id
        }
        
        # Update session if available
        if hasattr(session_service, 'update_session'):
            session_service.update_session(session_id, {
                'evaluation_results': evaluation_data,
                'evaluation_timestamp': datetime.now(),
                'step': 5
            })
        
        return {
            "success": True,
            "evaluation": evaluation_data
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/evaluation/benchmark")
async def benchmark_comparison(request: dict):
    """Get benchmark comparison for Verilog code"""
    if not VERILOGEVAL_AVAILABLE:
        raise HTTPException(status_code=404, detail="VerilogEval service not available")
    
    try:
        verilog_code = request.get("verilog_code", "")
        user_prompt = request.get("user_prompt", "")
        
        if not verilog_code:
            return {"success": False, "error": "No Verilog code provided"}
        
        benchmark_result = verilogeval_service.get_benchmark_comparison(
            verilog_code, user_prompt
        )
        
        return {"success": True, "benchmark": benchmark_result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/evaluation/pass-at-k")
async def calculate_pass_at_k(request: dict):
    """Calculate pass@k metric for multiple Verilog implementations"""
    if not VERILOGEVAL_AVAILABLE:
        raise HTTPException(status_code=404, detail="VerilogEval service not available")
    
    try:
        verilog_codes = request.get("verilog_codes", [])
        k = request.get("k", 1)
        
        if not verilog_codes:
            return {"success": False, "error": "No Verilog codes provided"}
        
        pass_at_k_score = verilogeval_service.generate_pass_at_k_metric(verilog_codes, k)
        
        return {
            "success": True,
            "pass_at_k": pass_at_k_score,
            "k": k,
            "total_codes": len(verilog_codes)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/evaluation/examples/{category}")
async def get_evaluation_examples(category: str):
    """Get example problems for evaluation categories"""
    if not VERILOGEVAL_AVAILABLE:
        raise HTTPException(status_code=404, detail="VerilogEval service not available")
    
    try:
        examples = {
            "combinational": [
                {"name": "2:1 Multiplexer", "difficulty": "Basic"},
                {"name": "4:1 Multiplexer", "difficulty": "Basic"},
                {"name": "3:8 Decoder", "difficulty": "Intermediate"},
                {"name": "8-bit Adder", "difficulty": "Intermediate"}
            ],
            "sequential": [
                {"name": "4-bit Counter", "difficulty": "Basic"},
                {"name": "Shift Register", "difficulty": "Basic"},
                {"name": "Traffic Light FSM", "difficulty": "Intermediate"},
                {"name": "UART Receiver", "difficulty": "Advanced"}
            ],
            "advanced": [
                {"name": "Simple ALU", "difficulty": "Advanced"},
                {"name": "SPI Controller", "difficulty": "Advanced"},
                {"name": "Cache Controller", "difficulty": "Expert"}
            ]
        }
        
        return {
            "success": True,
            "examples": examples.get(category, []),
            "category": category
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/evaluation/metrics/{session_id}")
async def get_evaluation_metrics(session_id: str):
    """Get evaluation metrics for a specific session"""
    if not VERILOGEVAL_AVAILABLE:
        raise HTTPException(status_code=404, detail="VerilogEval service not available")
    
    try:
        if hasattr(session_service, 'get_session'):
            session_data = session_service.get_session(session_id)
            if session_data and 'evaluation_results' in session_data:
                return {
                    "success": True,
                    "metrics": session_data['evaluation_results']
                }
        
        return {"success": False, "error": "No evaluation metrics found for session"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    
# ================== RAG ENDPOINTS ==================

@app.post("/api/rag/generate-mermaid")
async def generate_mermaid_with_rag(request: RAGMermaidRequest):
    """Generate Mermaid diagram using RAG"""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=404, detail="RAG service not available")
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        mermaid_code = mermaid_service.generate_mermaid_from_prompt(
            request.prompt, 
            model=request.model,
            use_rag=request.use_rag
        )
        
        return {
            "success": True,
            "mermaid_code": mermaid_code,
            "model_used": request.model,
            "rag_enhanced": request.use_rag,
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": request.session_id or str(uuid.uuid4())
        }

@app.get("/api/rag/status")
async def get_rag_status():
    """Get RAG system status"""
    if not RAG_AVAILABLE:
        return {"available": False}
    
    try:
        kb_size = len(mermaid_service.rag_service.knowledge_base) if hasattr(mermaid_service, 'rag_service') else 0
        return {
            "available": True,
            "knowledge_base_size": kb_size,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

# ================== SESSION ENDPOINTS ==================

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session data"""
    try:
        if hasattr(session_service, 'get_session'):
            session_data = session_service.get_session(session_id)
            if session_data:
                return {"success": True, "session": session_data}
            else:
                return {"success": False, "error": "Session not found"}
        else:
            return {"success": False, "error": "Session service not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Enhanced Verilog Generation System v2.0...")
    print(f"🤖 Multi-Model Support: {'Available' if MULTI_MODEL_AVAILABLE else 'Not Available (Groq only)'}")
    print(f"🔬 VerilogEval Support: {'Available' if VERILOGEVAL_AVAILABLE else 'Not Available'}")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/health")
    print("🎯 Available Models: http://localhost:8000/api/models/available")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )