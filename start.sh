# 1. Load the tokens from your .env file
export $(grep -v '^#' .env | xargs)

# 2. Start the FastAPI server in the background (&)
echo "Starting SAM API on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# 3. Start the Cloudflare Tunnel to expose port 8000 to the internet
echo "Starting Cloudflare Tunnel..."
cloudflared tunnel run --token $CF_TUNNEL_TOKEN