import sys
import os
import json
import traceback
from PIL import Image

# Prevent spconv/CUDA from auto-tuning and crashing
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "0"

# --- FIX: Patch Meta's hardcoded Conda assumption ---
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = "/usr/local/cuda"

import torch
# (Assume SAM 3D repo is cloned into external/sam-3d-objects)
sys.path.append("external/sam-3d-objects/notebook")
from inference import Inference

def update_ticket(task_id: str, status: str, output_file: str = None, error: str = None):
    data = {"task_id": task_id, "status": status}
    if output_file: data["output_file"] = output_file
    if error: data["error"] = error
    with open(f"tasks/{task_id}.json", "w") as f:
        json.dump(data, f)

def run_task(task_id, img_path, mask_path):
    update_ticket(task_id, "processing")
    try:
        pipeline = Inference("checkpoints/hf/pipeline.yaml", compile=False)
        img_rgb = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 
        
        output = pipeline(img_rgb, mask, seed=42)
        
        output_file = f"assets/{task_id}.ply"
        output["gs"].save_ply(output_file)
        
        update_ticket(task_id, "completed", output_file=output_file)
    except Exception as e:
        update_ticket(task_id, "failed", error=str(e))
    finally:
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(mask_path): os.remove(mask_path)

if __name__ == "__main__":
    run_task(sys.argv[1], sys.argv[2], sys.argv[3])