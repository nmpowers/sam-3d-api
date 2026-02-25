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
SAM3D_REPO_PATH = "external/sam-3d-objects"
NOTEBOOK_PATH = os.path.join(SAM3D_REPO_PATH, "notebook")
sys.path.append(NOTEBOOK_PATH)

try:
    from inference import Inference
except ImportError as e:
    print(f"Error importing Inference: {e}")
    sys.exit(1)

def update_ticket(task_id: str, status: str, output_file: str = None, error: str = None):
    data = {"task_id": task_id, "status": status}
    if output_file: data["output_file"] = output_file
    if error: data["error"] = error

    temp_file = f"tasks/{task_id}.json.tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f)
    os.replace(temp_file, f"tasks/{task_id}.json")

def smart_crop(img_rgb, mask_uint8, margin=0.1):
    """
    Crops the image and mask to be centered around the object.
    This prevents the pipeline's internal CenterCrop from deleting the object.
    """
    # Find bounding box of the mask
    coords = np.argwhere(mask_uint8 > 128)
    if coords.size == 0:
        print("Smart Crop: Mask is empty, skipping crop.")
        return img_rgb, mask_uint8

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # Calculate center and size
    h, w = y1 - y0, x1 - x0
    cy, cx = y0 + h // 2, x0 + w // 2
    size = max(h, w)
    
    # Add margin (default 10%)
    size = int(size * (1 + margin))
    
    # Calculate crop box (square)
    half = size // 2
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(img_rgb.shape[1], cx + half)
    bottom = min(img_rgb.shape[0], cy + half)
    
    # Perform Crop
    img_crop = img_rgb[top:bottom, left:right]
    mask_crop = mask_uint8[top:bottom, left:right]
    
    print(f"Smart Crop: Original {img_rgb.shape} -> Cropped {img_crop.shape}")
    return img_crop, mask_crop

def run_task(task_id, img_path, mask_path):
    update_ticket(task_id, "processing")
    try:
        # --- PATH CONFIGURATION ---
        possible_paths = [
            "/app/checkpoints/hf/checkpoints/pipeline.yaml",
            "checkpoints/hf/checkpoints/pipeline.yaml",
            "external/sam-3d-objects/checkpoints/pipeline.yaml"
        ]
        
        pipeline_path = None
        for p in possible_paths:
            if os.path.exists(p):
                pipeline_path = p
                print(f"Found pipeline config at: {p}")
                break
        
        if not pipeline_path:
            raise FileNotFoundError(f"Could not find pipeline.yaml.")
        
        # Load Pipeline
        pipeline = Inference(pipeline_path, compile=False)

        # --- 1. ROBUST LOADING ---
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path).convert("L"))
        
        # Normalize Mask (Strict 0 or 255)
        mask = (mask_raw > 128).astype(np.uint8) * 255
        
        if mask.max() == 0:
            raise ValueError("Task Failed: Mask is empty before cropping.")

        # --- 2. SMART CROP ---
        # This function centers the object so it isn't cropped out
        img_rgb, mask = smart_crop(img_rgb, mask)
        
        # Final Safety Check
        if mask.max() == 0:
            raise ValueError("Task Failed: Mask became empty after cropping.")

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
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(mask_path): os.remove(mask_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python worker_3d.py <task_id> <img_path> <mask_path>")
        sys.exit(1)
    
    task_id, img_path, mask_path = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"Worker started for Task ID: {task_id} using Python: {sys.executable}")
    run_task(task_id, img_path, mask_path)
