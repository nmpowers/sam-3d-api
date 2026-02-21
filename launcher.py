import os
import subprocess
from dotenv import load_dotenv

# 1. Load the .env file safely
load_dotenv()

cf_token = os.getenv("CF_TUNNEL_TOKEN")

# Strip any accidental quotes or spaces the user might have saved in the .env
if cf_token:
    cf_token = cf_token.strip().strip('"').strip("'")
else:
    print("ERROR: Could not find CF_TUNNEL_TOKEN in the .env file!")
    exit(1)

print(f"Token loaded successfully! Starts with: {cf_token[:10]}...")

# 2. Launch FastAPI in the background
print("Starting SAM 3D API...")
api_process = subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])

# 3. Launch Cloudflare Tunnel
print("Starting Cloudflare Tunnel...")
cf_process = subprocess.Popen(["cloudflared", "tunnel", "run", "--token", cf_token])

try:
    # Keep the script running so you can see the logs
    api_process.wait()
    cf_process.wait()
except KeyboardInterrupt:
    print("\nShutting down services...")
    api_process.terminate()
    cf_process.terminate()