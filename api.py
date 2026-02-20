import os
import json
import uuid
import base64
import io
import subprocess
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import base64 
import io
import torch
from PIL import Image

# Import SAM 3 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model globally so it doesn't reload on every web request
device = "cuda" if torch.cude.is_available() else "cpu"

sam3_model = build_sam3_image_model(
    checkpoint_path="checkpoints/sam3.pt", 
    enable_inst_interactivity=True
).to(device)

sam3_processor = Sam3Processor(sam3_model)

app = FastAPI(title="SAM 3 + SAM 3D Ticketing API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

for d in ["assets", "tasks", "temp"]:
    os.makedirs(d, exist_ok=True)

# --- SAM 2 INITIALIZATION ---
# (You will need to download the SAM 2 checkpoints to a /checkpoints folder)
# predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", "checkpoints/sam2_hiera_large.pt"))

def load_b64(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw))

@app.post("/segment")
async def segment_image(payload: dict = Body(...)):
    """Synchronous: Uses SAM 3 to generate a mask from a single clicked point."""
    try:
        # Load the raw image
        img = load_b64(payload["image_b64"]).convert("RGB")
        x, y = payload["x"], payload["y"]
        
        # Use torch.no_grad() to prevent memory leaks during rapid clicks
        with torch.no_grad():
            # 1. Load the image into the processor's state memory
            inference_state = sam3_processor.set_image(img)
            
            # 2. Predict the mask
            # point_labels=[1] tells SAM 3 this is a foreground object we want
            masks, scores, logits = sam3_model.predict_inst(
                inference_state,
                point_coords=[[x, y]], 
                point_labels=[1],
                multimask_output=True # Generates multiple size hypotheses
            )
            
        # 3. Isolate the mask with the highest confidence score
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]  # This is a 2D boolean numpy array
        
        # 4. Convert the boolean mask to a grayscale 8-bit image
        # True becomes 255 (White/Foreground), False becomes 0 (Black/Background)
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8, mode='L')
        
        # 5. Save to a byte buffer and base64 encode it
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"status": "success", "mask_b64": mask_b64})
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-3d")
async def generate_3d(payload: dict = Body(...)):
    """Asynchronous Ticketing: Spawns SAM 3D worker and returns a Task ID."""
    task_id = str(uuid.uuid4())
    
    img_path = f"temp/{task_id}_image.png"
    mask_path = f"temp/{task_id}_mask.png"
    
    load_b64(payload["image_b64"]).convert("RGB").save(img_path)
    load_b64(payload["mask_b64"]).convert("L").save(mask_path)
    
    # 1. Create the Ticket
    with open(f"tasks/{task_id}.json", "w") as f:
        json.dump({"task_id": task_id, "status": "queued"}, f)
    
    # 2. Spawn the isolated GPU worker (does not block the server!)
    subprocess.Popen(["python", "worker_3d.py", task_id, img_path, mask_path])
    
    # 3. Respond instantly
    return {"task_id": task_id, "status": "queued"}

@app.get("/status/{task_id}")
async def check_status(task_id: str):
    """Client polls this endpoint to check if the 3D model is done."""
    status_file = f"tasks/{task_id}.json"
    if not os.path.exists(status_file):
        raise HTTPException(status_code=404, detail="Task not found")
        
    with open(status_file, "r") as f:
        data = json.load(f)
        
    if data["status"] == "completed":
        return FileResponse(data["output_file"], media_type="application/octet-stream", filename=f"{task_id}.ply")
            
    return JSONResponse(content=data)