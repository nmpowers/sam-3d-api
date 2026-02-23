import os
import json
import uuid
import base64
import io
import sys
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch

# Import SAM 3 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT_PATH = "checkpoints/sam3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="SAM 3 + SAM 3D Ticketing API")
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

print(f"Loading SAM 3 model from {CHECKPOINT_PATH} on {DEVICE}...")
try:
    sam3_model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT_PATH,
        enable_inst_interactivity=True
    )
    sam3_model.to(DEVICE)
    sam3_processor = Sam3Processor(sam3_model)
    print("SAM 3 model loaded sucessfully!")
except TypeError as e:
    print(f"Error loading SAM 3 model: {e}. Attempting default load.")
    sam3_model = build_sam3_image_model(checkpoint=CHECKPOINT_PATH)
    sam3_model.to(DEVICE)
    sam3_processor = Sam3Processor(sam3_model)

# Initialization

def load_b64(b64_str: str) -> Image.Image:
    try:
        # Handle data URI scheme if present
        if "base64," in b64_str:
            b64_str = b64_str.split(",")[1]
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
    
def run_inference_sync(img_rgb, x, y):
    """
    Blocking GPU Inference: Runs SAM 3 inference synchronously. This is suitable for single clicks where latency is acceptable.
    """
    with torch.no_grad():
        inference_state = sam3_processor.set_image(img_rgb)
        masks, scores, logits = sam3_model.predict_inst(
            inference_state,
            point_coords=[[x, y]], 
            point_labels=[1],
            multimask_output=True
        )
    
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]

    mask_uint8 = (best_mask * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode='L')


@app.post("/segment")
async def segment_image(payload: dict = Body(...)):
    """
    Synchronous segmentation endpoint.
    Uses asyncio executor to avoid blocking the server event during GPU inference.
    """
    try:
        # Load the raw image
        img = load_b64(payload["image_b64"]).convert("RGB")
        x, y = payload["x"], payload["y"]

        # API will remain responsive for ping/status
        loop = asyncio.get_event_loop()
        mask_img = await loop.run_in_executor(None, run_inference_sync, img, x, y)
        
        # encode the mask back to base64 for response
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"status": "success", "mask_b64": mask_b64})
        
    except Exception as e:
        traceback.print_exc() # logging all for debug
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-3d")
async def generate_3d(payload: dict = Body(...)):
    """
    Asynchronous 3D generation endpoint.
    Spawns SAM 3D worker and returns a Task ID.
    """
    task_id = str(uuid.uuid4())

    try: 
        img_path = f"temp/{task_id}_image.png"
        mask_path = f"temp/{task_id}_mask.png"

        # save input for worker
        load_b64(payload["image_b64"]).convert("RGB").save(img_path)
        load_b64(payload["mask_b64"]).convert("L").save(mask_path)

        # make task status
        with open(f"tasks/{task_id}.json", "w") as f:
            json.dump({"task_id": task_id, "status": "queued"}, f)

        # spawn worker
        subprocess.Popen([sys.executable, "worker_3d.py", task_id, img_path, mask_path])

        return {"task_id": task_id, "status": "queued"}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/{task_id}")
async def check_status(task_id: str):
    """
    Client polls this endpoint to check if the 3D model is done.
    """
    status_file = f"tasks/{task_id}.json"

    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Task not found")
    
    try: 
        with open(status_file, "r") as f:
            data = json.load(f)
        
        if data["status"] == "completed":
            if os.path.exists(data["output_file"]):
                return FileResponse(
                    data["output_file"], 
                    media_type="application/octet-stream", 
                    filename=f"{task_id}.ply"
                    )
            else:
                return JSONResponse(content={"status": "error", "error": "Output file not found"})
            
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        return JSONResponse(content={"status": "processing"})