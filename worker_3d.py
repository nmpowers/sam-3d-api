import sys
import os
import json
import traceback
from PIL import Image
import numpy as np
import traceback

# Prevent spconv/CUDA from auto-tuning and crashing
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "0"

# --- FIX: Patch Meta's hardcoded Conda assumption ---
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = "/usr/local/cuda"

import torch
# (Assume SAM 3D repo is cloned into external/sam-3d-objects)
sys.path.append("external/sam-3d-objects/notebook")
try:
    from inference import Inference
except ImportError as e:
    print(f"Error importing Inference from SAM 3D repo: {e}")
    print("Ensure 'external/sam-3d-objects/notebook/inference.py' exists and is accessible.")
    sys.exit(1)

def update_ticket(task_id: str, status: str, output_file: str = None, error: str = None):
    """Writes status updates to a JSON file for the API to read"""
    data = {"task_id": task_id, "status": status}
    if output_file: 
        data["output_file"] = output_file
    if error: 
        data["error"] = error

    temp_file = f"tasks/{task_id}.json.tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f)
    os.replace(temp_file, f"tasks/{task_id}.json")  # Atomic update

def run_task(task_id, img_path, mask_path):
    update_ticket(task_id, "processing")
    try:
        # Load pipeline 
        # SAM 3D has its own checkpoints directory
        pipeline_path = "checkpoints/hf/checkpoints/pipeline.yaml"
        
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline config not found: {pipeline_path}")
        
        pipeline = Inference(pipeline_path, compile=False)

        # Convert the PIL Images into NumPy math arrays
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) 
        
        # Hand the arrays to the model
        output = pipeline(img_rgb, mask, seed=42)
        
        output_file = f"assets/{task_id}.ply" # save result
        os.makedirs("assets", exist_ok=True) # make directory if it doesn't exist
        output["gs"].save_ply(output_file)
        
        update_ticket(task_id, "completed", output_file=output_file)
    except Exception as e:
        traceback.print_exc()
        update_ticket(task_id, "failed", error=str(e))
    finally:
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(mask_path): os.remove(mask_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python worker_3d.py <task_id> <img_path> <mask_path>")
        sys.exit(1)
    
    task_id, img_path, mask_path = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"Worker started for Task ID: {task_id}")
    run_task(task_id, img_path, mask_path)