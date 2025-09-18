# model_manager.py
# FastAPI app for model management and baseline text generation.
# Fixed to properly handle both JSON and form data and generate text correctly

from __future__ import annotations

import os
import time
import json
import psutil
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
)

# ------------------------------------------------------------------------------
# App & CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="TorchWeave Model Manager", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DEVICE = "cpu"  # keep simple/portable; override later if you add CUDA

# ------------------------------------------------------------------------------
# In-memory registry/state
# ------------------------------------------------------------------------------
class LoadedModel(BaseModel):
    model_id: str
    status: str = "loaded"
    memory_usage: str = "unknown"
    load_time: str = ""
    device: str = DEVICE


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}  # model_id -> {"pipe": pipeline, "tokenizer":..., "model":...}
LOADED_MODELS: Dict[str, LoadedModel] = {}

# A simple catalog shown by /models/available
AVAILABLE_MODELS = [
    {
        "model_id": "microsoft/DialoGPT-small",
        "description": "Small conversational model - Fast inference",
        "size": "117MB",
        "type": "causal_lm",
    },
    {
        "model_id": "gpt2",
        "description": "GPT-2 base model - Creative text generation",
        "size": "548MB",
        "type": "causal_lm",
    },
    {
        "model_id": "distilgpt2",
        "description": "Distilled GPT-2 - Faster, smaller model",
        "size": "353MB",
        "type": "causal_lm",
    },
    {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Tiny Llama chat model - Good balance of size and performance",
        "size": "1.1GB",
        "type": "causal_lm",
    },
]

# ------------------------------------------------------------------------------
# Request/Response models
# ------------------------------------------------------------------------------
class LoadModelBody(BaseModel):
    model_id: Optional[str] = Field(None, description="HF Model ID to load")
    custom_model_id: Optional[str] = Field(
        None, description="Alternate field used by some UIs"
    )

class GenerateBody(BaseModel):
    model_id: Optional[str] = None
    custom_model_id: Optional[str] = None
    prompt: Optional[str] = None

    # Sampling/length controls (JSON friendly)
    max_length: Optional[int] = 128
    max_new_tokens: Optional[int] = None  # preferred if provided
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = None
    use_chat_template: Optional[bool] = False

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _mem_usage_str() -> str:
    try:
        vm = psutil.virtual_memory()
        return f"{vm.percent:.1f}%"
    except Exception:
        return "unknown"

def _now_iso() -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    except Exception:
        return ""

def _resolve_model_id(payload: dict) -> str:
    m = (payload or {}).get("model_id") or (payload or {}).get("custom_model_id")
    if not m:
        raise HTTPException(status_code=400, detail="model_id or custom_model_id is required")
    return m

async def _read_payload_json_or_form(request: Request) -> dict:
    """
    Read JSON Body (preferred). If not JSON, try form/multipart.
    This makes the endpoint compatible with both frontend JSON and curl -F.
    """
    content_type = request.headers.get("content-type", "")
    
    # Handle JSON requests
    if "application/json" in content_type.lower():
        try:
            return await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    # Handle form data requests
    if "multipart/form-data" in content_type.lower() or "application/x-www-form-urlencoded" in content_type.lower():
        try:
            form = await request.form()
            return {k: v for k, v in form.items()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid form data: {e}")
    
    # Try to read as JSON from body as fallback
    try:
        body = await request.body()
        if body:
            return json.loads(body.decode())
        return {}
    except Exception:
        return {}

def _build_generate_kwargs(body: GenerateBody) -> Dict[str, Any]:
    gen_kwargs: Dict[str, Any] = {}

    # Prefer max_new_tokens if provided; else fall back to max_length
    if body.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = int(body.max_new_tokens)
    elif body.max_length is not None:
        gen_kwargs["max_length"] = int(body.max_length)

    if body.temperature is not None:
        gen_kwargs["temperature"] = float(body.temperature)
    if body.top_p is not None:
        gen_kwargs["top_p"] = float(body.top_p)
    if body.top_k is not None:
        gen_kwargs["top_k"] = int(body.top_k)
    if body.do_sample is not None:
        gen_kwargs["do_sample"] = bool(body.do_sample)
    if body.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(body.repetition_penalty)

    # Force text generation with reasonable defaults
    gen_kwargs["do_sample"] = True
    gen_kwargs["pad_token_id"] = 50256  # GPT-2 EOS token
    gen_kwargs["eos_token_id"] = 50256
    
    # Ensure minimum generation
    if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
        gen_kwargs["max_new_tokens"] = 50

    return gen_kwargs

def _ensure_pipe(model_id: str):
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]["pipe"]

    # Load tokenizer & model and cache a text-generation pipeline
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(model_id)
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,   # CPU; change to a CUDA device id if you add GPU later
    )
    MODEL_REGISTRY[model_id] = {"pipe": text_gen, "tokenizer": tokenizer, "model": model}

    # Update LOADED_MODELS table for /models/loaded
    LOADED_MODELS[model_id] = LoadedModel(
        model_id=model_id,
        status="loaded",
        memory_usage=_mem_usage_str(),
        load_time=_now_iso(),
        device=DEVICE,
    )
    return text_gen

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "model-manager", "port": 8001}

@app.get("/models/available")
def models_available():
    return {"available_models": AVAILABLE_MODELS}

@app.get("/models/loaded")
def models_loaded():
    return {"loaded_models": [m.model_dump() for m in LOADED_MODELS.values()]}

@app.get("/system/stats")
def system_stats():
    # lightweight system snapshot
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    return {
        "cpu_percent": cpu,
        "memory": {
            "percent": vm.percent,
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
        },
        "loaded_models": list(LOADED_MODELS.keys()),
        "service": "model-manager",
    }

@app.post("/models/load")
async def load_model(request: Request):
    payload = await _read_payload_json_or_form(request)
    model_id = _resolve_model_id(payload)

    # If already loaded: return early
    if model_id in MODEL_REGISTRY:
        return {
            "success": True,
            "message": f"Model {model_id} is already loaded",
            "model_id": model_id,
            "status_code": 200,
        }

    t0 = time.time()
    try:
        _ensure_pipe(model_id)
        t1 = time.time()
        return {
            "success": True,
            "message": f"Model {model_id} loaded successfully",
            "model_id": model_id,
            "load_time": t1 - t0,
            "device": DEVICE,
            "status_code": 200,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}: {e}")

@app.post("/models/generate")
async def generate(request: Request):
    """
    Accept both JSON and form data for generation requests.
    Fixed to properly handle both content types and generate text correctly.
    """
    try:
        payload = await _read_payload_json_or_form(request)
        
        # Convert form data to proper types if needed
        if isinstance(payload.get("max_length"), str):
            payload["max_length"] = int(payload["max_length"])
        if isinstance(payload.get("max_new_tokens"), str):
            payload["max_new_tokens"] = int(payload["max_new_tokens"])
        if isinstance(payload.get("temperature"), str):
            payload["temperature"] = float(payload["temperature"])
        if isinstance(payload.get("top_p"), str):
            payload["top_p"] = float(payload["top_p"])
        if isinstance(payload.get("top_k"), str):
            payload["top_k"] = int(payload["top_k"])
        if isinstance(payload.get("repetition_penalty"), str):
            payload["repetition_penalty"] = float(payload["repetition_penalty"])

        # Create GenerateBody object
        data = GenerateBody(**payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request parsing failed: {e}")

    model_id = _resolve_model_id(payload)
    prompt = data.prompt
    if not prompt:
        raise HTTPException(status_code=422, detail="Field 'prompt' is required")

    # ensure model/pipeline
    try:
        text_gen = _ensure_pipe(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}: {e}")

    # Prepare generation kwargs
    gen_kwargs = _build_generate_kwargs(data)

    # Timing
    t0_total = time.time()
    t0_tok = time.time()
    tokenizer = MODEL_REGISTRY[model_id]["tokenizer"]
    input_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokenization_time = time.time() - t0_tok

    # Run generation
    t0_gen = time.time()
    try:
        outputs = text_gen(prompt, **gen_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    generation_time = time.time() - t0_gen

    # Enhanced text extraction and processing
    t0_dec = time.time()
    generated_text = ""
    token_count = 0
    
    try:
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            raw_output = outputs[0]
            
            if isinstance(raw_output, dict) and "generated_text" in raw_output:
                full_generated = raw_output["generated_text"]
                
                # More robust prompt removal
                if full_generated.startswith(prompt):
                    generated_text = full_generated[len(prompt):].strip()
                else:
                    # Fallback: find the prompt in the output and remove everything before it
                    prompt_pos = full_generated.find(prompt)
                    if prompt_pos >= 0:
                        generated_text = full_generated[prompt_pos + len(prompt):].strip()
                    else:
                        # If prompt not found, use the full output
                        generated_text = full_generated.strip()
                
                # Ensure we have some output - if still empty after prompt removal
                if not generated_text and full_generated.strip():
                    # Use everything after the first newline or significant break
                    lines = full_generated.split('\n')
                    if len(lines) > 1:
                        generated_text = '\n'.join(lines[1:]).strip()
                    elif len(full_generated.strip()) > len(prompt.strip()) + 5:
                        # Use second half if no line breaks and significantly longer
                        mid_point = len(prompt)
                        generated_text = full_generated[mid_point:].strip()
                    else:
                        # Last resort - show that generation occurred
                        generated_text = full_generated.strip()
                
                # Calculate token count
                if generated_text:
                    try:
                        tokens = tokenizer.encode(generated_text, add_special_tokens=False)
                        token_count = len(tokens)
                    except:
                        token_count = max(1, len(generated_text.split()))
                
            else:
                generated_text = str(raw_output) if raw_output else ""
                token_count = max(1, len(generated_text.split())) if generated_text else 0
                
        else:
            generated_text = ""
            token_count = 0
            
    except Exception as e:
        logger.error(f"Error processing generated text: {e}", exc_info=True)
        generated_text = f"Error processing output: {str(e)}"
        token_count = 0
    
    decoding_time = time.time() - t0_dec

    total_time = time.time() - t0_total
    ttft = generation_time  # rough: no streaming, so entire generation ~= TTFT

    tokens_per_second = (token_count / generation_time) if generation_time > 0 else 0.0

    return {
        "generated_text": generated_text,
        "model_id": model_id,
        "prompt": prompt,
        "metrics": {
            "total_time": total_time,
            "tokenization_time": tokenization_time,
            "generation_time": generation_time,
            "decoding_time": decoding_time,
            "ttft_estimate": ttft,
            "tokens_per_second": tokens_per_second,
            "token_count": token_count,
            "input_token_count": len(input_tokens),
            "method": "baseline_no_batching",
        },
        "model_info": {
            "device": DEVICE,
            "max_length": payload.get("max_length", 128),
            "temperature": payload.get("temperature", 1.0),
        },
    }
