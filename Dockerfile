FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Clone SAM-3D-Objects (Provides the Inference class)
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git external/sam-3d-objects

# 3. Clone SAM 3 (Provides the sam3 library)
RUN git clone https://github.com/facebookresearch/sam3.git external/sam3

# 4. Install PyTorch3D and core ML libraries
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Install SAM 3D dependencies
RUN pip install -e './external/sam-3d-objects[dev]' \
    && pip install -e './external/sam-3d-objects[p3d]' \
    && pip install -e './external/sam-3d-objects[inference]'

# Install SAM 3 dependencies
RUN pip install -e './external/sam3'

# Install FastAPI stack
RUN pip install fastapi uvicorn python-multipart pydantic

# 5. Copy your custom API scripts into the container
COPY api.py worker_3d.py ./

# 6. Start the FastAPI server with proxy headers enabled for Cloudflare
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=\"*\""]