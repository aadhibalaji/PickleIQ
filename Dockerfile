FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# runtime (not devel) is enough — we're running PyTorch, not compiling CUDA code.
# libgl1/libglib2.0-0 are opencv-python's runtime deps (libGL.so.1) — without
# them cv2 fails to import and the server never starts.
RUN apt-get update && apt-get install -y python3.12 python3-pip libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch + torchvision installed together, from the same cu126 index, pinned to a
# known-matching pair — installing torchvision separately (or unpinned via a
# transitive ultralytics dep) risks a build mismatch that breaks torchvision's
# custom torchvision::nms op registration against torch at runtime.
RUN pip3 install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --break-system-packages --index-url https://download.pytorch.org/whl/cu126

COPY backend/requirements_cuda.txt backend/requirements_cuda.txt
RUN pip3 install -r backend/requirements_cuda.txt --break-system-packages

# Keep the backend/ subdirectory intact (don't flatten into /app) so
# backend/app/*.py's relative repo-root walk (see app/paths.py) lands on
# /app here the same way it lands on the real repo root in local dev.
COPY backend/ backend/
COPY yolo11n-pose.pt backend/yolo11n-pose.pt

# Belt-and-suspenders: paths.py honors this outright, so root resolution
# doesn't silently break if the COPY layout above ever changes.
ENV REPO_ROOT=/app
WORKDIR /app/backend

EXPOSE 8420
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8420"]