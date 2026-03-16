import subprocess
import time
import sys
import os

print("Starting Multi-GPU API System")
os.chdir("/app") 

# GPU 0 Configuration (Main API and SAM 2 model)
env_api = os.environ.copy()
env_api["CUDA_VISIBLE_DEVICES"] = "0"

# GPU 1 Configuration (SAM 3D Model)
env_worker = os.environ.copy()
env_worker["CUDA_VISIBLE_DEVICES"] = "1"
WORKER_PYTHON_EXE = "/root/miniconda3/envs/sam3d-objects/bin/python"

# Launch GPU 1 service
print("Launching internal SAM 3D worker on GPU 1 (Port 8001)...")
worker_process = subprocess.Popen(
    [WORKER_PYTHON_EXE, "-m", "uvicorn", "worker_api:app", "--host", "0.0.0.0", "--port", "8001"],
    env=env_worker
)

# Launch GPU 0 service
print("Launching main API on GPU 0 (Port 8000)...")
api_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
    env=env_api
    )

# Give the API a second to boot up before linking the tunnel
time.sleep(2)

print("\n=========================================================")
print("Starting Free Cloudflare Quick Tunnel...")
print("LOOK FOR THE LINK ENDING IN '.trycloudflare.com' BELOW")
print("=========================================================\n")

# 2. Launch the Quick Tunnel (No token required!)
cf_process = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8000"])

try:
    # Keep the script running so you can see the logs
    api_process.wait()
    worker_process.wait()
    cf_process.wait()
except KeyboardInterrupt:
    print("\nShutting down services...")
    api_process.terminate()
    worker_process.terminate()
    cf_process.terminate()