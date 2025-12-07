# Option B: Use CUDA 12.4 (Works on RTX 5070 Ti via backward compatibility)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# 1. Install system dependencies often needed by DreamerV3 (ffmpeg, etc)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Dreamer requirements FIRST
# This lets Dreamer pick the older JAX version it wants (e.g. 0.4.x)
COPY third_party/dreamerv3/env/requirements.txt /tmp/dreamer_reqs.txt
RUN pip install --root-user-action=ignore -r /tmp/dreamer_reqs.txt

# 3. OVERRIDE with the specific JAX Cuda 12 wheel
# Dreamer's reqs might install CPU-only JAX by mistake. This forces GPU support.
# We pin jax to 0.4.26 or 0.4.30 which are stable for Dreamer.
RUN pip install --upgrade --root-user-action=ignore \
  "jax[cuda12_pip]==0.4.26" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --root-user-action=ignore -r requirements.txt

COPY . .

CMD ["/bin/bash"]