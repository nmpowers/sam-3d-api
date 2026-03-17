import sys
import os
import json
import traceback
from PIL import Image
import numpy as np
from fastapi import FastAPI, BackgroundTasks
import torch
from pydantic import BaseModel

class TaskPayload(BaseModel):
    task_id: str
    img_path: str
    mask_path: str

# need to prevent CUDA from crashing with auto-tune
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "0"

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = "/usr/local/cuda"

# Add the SAM 3D repo to path so we can import 'Inference'
SAM3D_REPO_PATH = "external/sam-3d-objects"
NOTEBOOK_PATH = os.path.join(SAM3D_REPO_PATH, "notebook")
sys.path.append(NOTEBOOK_PATH)

from inference import Inference

app = FastAPI(title="SAM 3D Internal Worker API")

pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    print("Loading SAM 3D model onto GPU 1...")
    possible_paths = [
        "/app/checkpoints/hf/checkpoints/pipeline.yaml",  # Matches download_models.py
        "/app/checkpoints/pipeline.yaml",  # Backup location
        "external/sam-3d-objects/checkpoints/pipeline.yaml" # In case of direct repo clone
    ]

    pipeline_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not pipeline_path:
        raise FileNotFoundError("Could not find pipeline.yaml in expected locations.")
    
    pipeline = Inference(pipeline_path, compile=False)
    print("SAM 3D model loaded successfully.")

def update_ticket(task_id: str, status: str, error: str = None, **kwargs):
    data = {"task_id": task_id, "status": status}
    if error: data["error"] = error
    data.update(kwargs)

    temp_file = f"tasks/{task_id}.json.tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f)
    os.replace(temp_file, f"tasks/{task_id}.json")

# Helper crop function for mask
def smart_crop(img_rgb, mask_uint8, margin=0.1):
    coords = np.argwhere(mask_uint8 > 128)
    if coords.size == 0:
        print("Smart Crop: Mask is empty, skipping crop.")
        return img_rgb, mask_uint8

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    h, w = y1 - y0, x1 - x0
    cy, cx = y0 + h // 2, x0 + w // 2
    size = max(h, w)
    
    size = int(size * (1 + margin))
    
    left, top = max(0, cx - size // 2), max(0, cy - size // 2)
    right, bottom = min(img_rgb.shape[1], cx + size // 2), min(img_rgb.shape[0], cy + size // 2)

    cropped_img = img_rgb[top:bottom, left:right]
    cropped_mask = mask_uint8[top:bottom, left:right]

    return cropped_img, cropped_mask

def run_3d_generation(task_id: str, img_path: str, mask_path: str):
    """The core generation logic, running off the global pipeline."""
    update_ticket(task_id, "processing")
    try:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path).convert("L"))
        mask = (mask_raw > 128).astype(np.uint8) * 255

        if mask.max() == 0: raise ValueError("Mask is empty. Please provide a valid segmentation mask.")
        img_rgb, mask = smart_crop(img_rgb, mask)
        if mask.max() == 0: raise ValueError("After cropping, mask is empty. Please provide a valid segmentation mask.")

        print(f"Running 3D generation for task {task_id}...")
        output = pipeline(img_rgb, mask, seed=42)

        os.makedirs("assets", exist_ok=True)
        ply_file = f"assets/{task_id}.ply"
        glb_file = f"assets/{task_id}.glb"

        output["gs"].save_ply(ply_file)
        output["glb"].export(glb_file)

        update_ticket(task_id, "completed", ply_path=ply_file, glb_path=glb_file)
    except Exception as e:
        traceback.print_exc()
        update_ticket(task_id, "failed", error=str(e))
    finally:
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(mask_path): os.remove(mask_path)

@app.post("/process-3d")
async def process_3d(payload: TaskPayload, background_tasks: BackgroundTasks):
    """Recieves the job from the main API and processess it in the background."""
    print(f"Worker recieved task: {payload.task_id}")
    background_tasks.add_task(run_3d_generation, payload.task_id, payload.img_path, payload.mask_path)
    return {"status": "accepted"}