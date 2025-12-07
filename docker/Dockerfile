# General Docker Image for Hierarchical World Models
# Uses PyTorch base image with CUDA 12.4 support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Install system dependencies for rendering and visualization
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglew-dev \
    x11-xserver-utils \
    xvfb \
    wget \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy repository
COPY . .

# Install JAX with CUDA 12 support (for DreamerV3 compatibility)
RUN pip install --upgrade pip && \
    pip install --root-user-action=ignore "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install DreamerV3 dependencies if present
RUN if [ -f "third_party/dreamerv3/requirements.txt" ]; then \
        pip install --root-user-action=ignore -r third_party/dreamerv3/requirements.txt; \
    fi

# Install project dependencies
RUN pip install --root-user-action=ignore -r requirements.txt

# Install third-party packages in editable mode
RUN if [ -f "third_party/dreamerv3/setup.py" ]; then \
        pip install --root-user-action=ignore -e third_party/dreamerv3; \
    fi

# Patch legacy torch.nn.Buffer usage for TDMPC2 (CRITICAL for PyTorch 2.0+)
COPY docker/scripts/patch_buffer_docker.sh /tmp/patch_buffer.sh
RUN chmod +x /tmp/patch_buffer.sh && /tmp/patch_buffer.sh

# Set environment variables for GPU rendering
ENV MUJOCO_GL=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

CMD ["/bin/bash"]