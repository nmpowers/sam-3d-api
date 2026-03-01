import os
import json
import uuid
import base64
import io
import sys
import subprocess
import asyncio
import traceback

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch

# --- SAM 2 IMPORTS (Running in Main Environment A) ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
# NEW / CORRECT (Matches download_models.py):
CHECKPOINT_PATH = "/app/checkpoints/sam2.1_hiera_large.pt"
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- WORKER CONFIGURATION ---
# Path to the ISOLATED Python environment for SAM 3D
WORKER_PYTHON_EXE = "/root/miniconda3/envs/sam3d-objects/bin/python"
WORKER_SCRIPT = "/app/worker_3d.py"

app = FastAPI(title="SAM 2 + SAM 3D Ticketing API")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"]
)

# Ensure necessary directories exist
for d in ["assets", "tasks", "temp"]:
    os.makedirs(d, exist_ok=True)

# --- LOAD SAM 2 MODEL GLOBALLY ---
print(f"Loading SAM 2 model from {CHECKPOINT_PATH}...")
try:
    sam2_model = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
    print("SAM 2 model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR loading SAM 2: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_b64(b64_str: str) -> Image.Image:
    try:
        if "base64," in b64_str:
            b64_str = b64_str.split(",")[1]
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

def run_inference_sync(img_rgb, x, y):
    """
    Blocking GPU Inference for SAM 2.
    """
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor = SAM2ImagePredictor(sam2_model)
        img_np = np.array(img_rgb)
        predictor.set_image(img_np)
        
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]), 
            multimask_output=True
        )
    
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode='L')

@app.post("/segment")
async def segment_image(payload: dict = Body(...)):
    try:
        img = load_b64(payload["image_b64"]).convert("RGB")
        x, y = payload["x"], payload["y"]

        loop = asyncio.get_event_loop()
        mask_img = await loop.run_in_executor(None, run_inference_sync, img, x, y)
        
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"status": "success", "mask_b64": mask_b64})
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-3d")
async def generate_3d(payload: dict = Body(...)):
    """
    Spawns the worker_3d.py process using the ISOLATED 'sam3d' environment.
    """
    task_id = str(uuid.uuid4())

    try: 
        img_path = f"temp/{task_id}_image.png"
        mask_path = f"temp/{task_id}_mask.png"

        load_b64(payload["image_b64"]).convert("RGB").save(img_path)
        load_b64(payload["mask_b64"]).convert("L").save(mask_path)

        with open(f"tasks/{task_id}.json", "w") as f:
            json.dump({"task_id": task_id, "status": "queued"}, f)

        # --- CRITICAL CHANGE: USE ISOLATED ENV ---
        # Instead of sys.executable (which is Python 3.12 / SAM 2),
        # we use the hardcoded path to the 'sam3d' environment (Python 3.10 / SAM 3D)
        cmd = [WORKER_PYTHON_EXE, WORKER_SCRIPT, task_id, img_path, mask_path]
        
        print(f"Spawning worker: {' '.join(cmd)}")
        subprocess.Popen(cmd)

        return {"task_id": task_id, "status": "queued"}
    
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/{task_id}")
async def check_status(task_id: str, format: str = "glb"):
    """
    Checks task status and serves the requested format.
    Usage: /status/{task_id}?format=glb OR /status/{task_id}?format=ply
    """
    if format not in ["glb", "ply"]:
        return JSONResponse(status_code=400, content={"error": "Invalid format requested. Use 'glb' or 'ply'."})
    
    status_file = f"tasks/{task_id}.json"
    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Task not found")
    
    try: 
        with open(status_file, "r") as f:
            data = json.load(f)
        
        if data["status"] == "completed":
            file_key = f"{format}_file" 
            target_file = data.get(file_key)
            
            if target_file and os.path.exists(target_file):
                return FileResponse(
                    target_file, 
                    media_type="application/octet-stream", 
                    filename=f"{task_id}.{format}"
                )
            else:
                return JSONResponse(content={"status": "error", "error": f"{format.upper()} file missing on server."})
            
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        return JSONResponse(content={"status": "processing"})