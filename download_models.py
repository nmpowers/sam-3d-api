import os
from huggingface_hub import login, hf_hub_download, snapshot_download

# --- SECURE TOKEN EXTRACTION ---
# Read the .env file line by line to find the HF_TOKEN without needing external libraries
hf_token = None
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines and comments
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                if key == "HF_TOKEN":
                    hf_token = value
                    break

# Fallback if the script is run in an environment where the variable is already exported
if not hf_token:
    hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    raise ValueError("ERROR: Could not find HF_TOKEN in the .env file!")

# 1. Authenticate Securely
print("Authenticating with Hugging Face...")
login(token=hf_token)

# Ensure the destination directories exist
os.makedirs("checkpoints/hf", exist_ok=True)

# 2. Download SAM 2.1 Large (Replaces SAM 3)
# We download both the weights (.pt) and the config (.yaml) to be safe.
print("Downloading SAM 2.1 Large...")
hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    filename="sam2.1_hiera_large.pt",
    local_dir="checkpoints"
)

hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    filename="sam2.1_hiera_l.yaml",
    local_dir="checkpoints"
)

# 3. Download SAM 3D Objects (Entire Repository)
# This handles the pipeline.yaml and other 3D-specific assets
print("Downloading SAM 3D Objects...")
snapshot_download(
    repo_id="facebook/sam-3d-objects",
    local_dir="checkpoints/hf"
)

print("All models successfully downloaded!")