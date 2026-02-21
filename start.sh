# 1. Strip any invisible Windows/Mac carriage returns from the .env file
sed -i 's/\r$//' .env

# 2. Securely load the variables using native bash sourcing
set -a
source .env
set +a

# 3. Quick verification (Prints just the first 10 characters to prove it loaded)
echo "Loaded Token successfully! Starts with: ${CF_TUNNEL_TOKEN:0:10}..."

# 4. Start the FastAPI server in the background
echo "Starting SAM API on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# 5. Start the Cloudflare Tunnel
echo "Starting Cloudflare Tunnel..."
cloudflared tunnel run --token "$CF_TUNNEL_TOKEN"