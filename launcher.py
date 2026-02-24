import subprocess
import time
import sys
import os

print("Starting SAM 3D API...")

os.chdir("/app") 

# 1. Launch FastAPI in the background
api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])

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
    cf_process.wait()
except KeyboardInterrupt:
    print("\nShutting down services...")
    api_process.terminate()
    cf_process.terminate()