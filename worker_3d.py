import sys
import os
import json
import traceback
from PIL import Image
import numpy as np

# Prevent spconv/CUDA from auto-tuning and crashing
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "0"

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = "/usr/local/cuda"

import torch

# Add the SAM 3D repo to path so we can import 'Inference'
# Since we are running from /app, this relative path works fine.
SAM3D_REPO_PATH = "external/sam-3d-objects"
NOTEBOOK_PATH = os.path.join(SAM3D_REPO_PATH, "notebook")
sys.path.append(NOTEBOOK_PATH)

try:
    from inference import Inference
except ImportError as e:
    print(f"Error importing Inference: {e}")
    print(f"Checked path: {NOTEBOOK_PATH}")
    sys.exit(1)

def update_ticket(task_id: str, status: str, output_file: str = None, error: str = None):
    data = {"task_id": task_id, "status": status}
    if output_file: data["output_file"] = output_file
    if error: data["error"] = error

    temp_file = f"tasks/{task_id}.json.tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f)
    os.replace(temp_file, f"tasks/{task_id}.json")

def run_task(task_id, img_path, mask_path):
    update_ticket(task_id, "processing")
    try:
        # --- PATH CONFIGURATION ---
        # Updated to prioritize the most likely location in the Split-Brain Dockerfile
        possible_paths = [
            "/app/checkpoints/hf/checkpoints/pipeline.yaml", # Default download structure
            "/app/checkpoints/hf/pipeline.yaml",             # Flattened download structure
            "checkpoints/hf/checkpoints/pipeline.yaml",      # Relative path
            "external/sam-3d-objects/checkpoints/pipeline.yaml" # Fallback to repo default
        ]
        
        pipeline_path = None
        for p in possible_paths:
            if os.path.exists(p):
                pipeline_path = p
                print(f"Found pipeline config at: {p}")
                break
        
        if not pipeline_path:
            raise FileNotFoundError(f"Could not find pipeline.yaml. Checked: {possible_paths}")
        
        # Load Pipeline
        # Compile=False is safer for the split environment to avoid JIT errors
        pipeline = Inference(pipeline_path, compile=False)

        # Load Images
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) 
        
        # Run 3D Generation
        print(f"Starting inference for Task {task_id}...")
        output = pipeline(img_rgb, mask, seed=42)
        
        # Save Result
        output_file = f"assets/{task_id}.ply"
        os.makedirs("assets", exist_ok=True)
        output["gs"].save_ply(output_file)
        
        print(f"Task {task_id} completed successfully.")
        update_ticket(task_id, "completed", output_file=output_file)

    except Exception as e:
        traceback.print_exc()
        update_ticket(task_id, "failed", error=str(e))
    finally:
        # Cleanup temp inputs
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(mask_path): os.remove(mask_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python worker_3d.py <task_id> <img_path> <mask_path>")
        sys.exit(1)
    
    task_id, img_path, mask_path = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"Worker started for Task ID: {task_id} using Python: {sys.executable}")
    run_task(task_id, img_path, mask_path)