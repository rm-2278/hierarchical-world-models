# Docker Setup for Hierarchical World Models

This directory contains Docker configurations for running DreamerV3, TDMPC2, and the general Hierarchical World Models (HWM) environment.

## Requirements

- Docker 20.10+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.x compatible drivers (Driver version 525.60.13 or newer)

## Available Docker Images

### 1. DreamerV3 (`Dockerfile.dreamer`)

Built for running DreamerV3 experiments with JAX and CUDA 12 support.

**Base Image:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

**Key Features:**
- JAX with CUDA 12 support (version 0.4.33)
- All DreamerV3 dependencies
- System libraries for rendering (ffmpeg, libgl1-mesa-glx, etc.)
- Configured for GPU-accelerated RL environments

**Build:**
```bash
docker build -f docker/Dockerfile.dreamer -t hwm-dreamer .
```

**Run:**
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace hwm-dreamer
```

### 2. TDMPC2 (`Dockerfile.tdmpc`)

Built for running TDMPC2 experiments with PyTorch and CUDA 12 support.

**Base Image:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

**Key Features:**
- PyTorch 2.5.1 with CUDA 12.4 support
- Automatic patching of deprecated `torch.nn.Buffer` usage
- System libraries for rendering
- TensorDict and related dependencies

**Build:**
```bash
docker build -f docker/Dockerfile.tdmpc -t hwm-tdmpc .
```

**Run:**
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace hwm-tdmpc
```

### 3. Hierarchical World Models (`Dockerfile.hwm`)

Combined environment supporting both DreamerV3 and TDMPC2.

**Base Image:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

**Key Features:**
- Both PyTorch and JAX with CUDA 12 support
- All dependencies for DreamerV3 and TDMPC2
- Ideal for comparative experiments

**Build:**
```bash
docker build -f docker/Dockerfile.hwm -t hwm-hwm .
```

**Run:**
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace hwm-hwm
```

### 4. General (`Dockerfile`)

Default Docker image for the project, same as HWM.

## Using Docker Compose

For easier management, use docker-compose:

```bash
# Build all images
docker-compose build

# Run DreamerV3 container
docker-compose run dreamer

# Run TDMPC2 container
docker-compose run tdmpc

# Run HWM container
docker-compose run hwm
```

## CUDA Version Compatibility

All Docker images are built with **CUDA 12.4** support. This is compatible with:

- NVIDIA GPUs with Compute Capability 5.0 and above
- NVIDIA Driver version 525.60.13 or newer
- CUDA 12.x runtime (backward compatible with 12.0, 12.1, 12.2, 12.3, 12.4)

**Note:** CUDA 13.0 does not exist. The latest CUDA version is 12.x. If your system shows "CUDA Version: 13.0", it likely refers to the maximum CUDA version your driver supports, which includes CUDA 12.x.

## Testing Models

### DreamerV3 Tests

```bash
# Inside the dreamer container
export RUN_GPU_SMOKE=1
pytest experiments/tests/test_dreamerv3_gpu.py -v
```

### TDMPC2 Tests

```bash
# Inside the tdmpc container
export RUN_GPU_SMOKE=1
pytest experiments/tests/test_tdmpc2_gpu.py -v
```

## Common Issues

### 1. "Failed to authorize" when pulling base image

If you encounter authentication issues with `nvcr.io/nvidia/pytorch`, the Dockerfiles use the public `pytorch/pytorch` images instead. These are equally capable for this project.

### 2. torch.nn.Buffer deprecation

The Dockerfiles automatically patch the deprecated `torch.nn.Buffer` usage in TDMPC2 code. This is done via `sed` commands during the build process and converts:
- `Buffer(...)` → `register_buffer("name", ...)`

### 3. JAX not detecting GPU

Ensure:
- NVIDIA Docker runtime is properly configured
- Container is started with `--gpus all` flag
- JAX is installed with CUDA support: `jax[cuda12]`

### 4. Out of memory errors

For development/testing, reduce batch sizes or use the debug configuration:
```bash
python main.py --configs debug
```

## Environment Variables

The Dockerfiles set the following environment variables:

- `MUJOCO_GL=egl` - Use EGL for headless rendering
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` - Prevent JAX from pre-allocating GPU memory

You can override these when running containers:
```bash
docker run --gpus all -e MUJOCO_GL=osmesa -it hwm-dreamer
```

## Building on Systems with CUDA 12.x

If your system has CUDA 12.0, 12.1, 12.2, 12.3, or 12.4 drivers, these Docker images will work correctly due to CUDA's forward compatibility. The Docker images use CUDA 12.4, which is compatible with all CUDA 12.x drivers.

## Additional Resources

- [DreamerV3 Repository](https://github.com/danijar/dreamerv3)
- [TDMPC2 Repository](https://github.com/nicklashansen/tdmpc2)
- [JAX Installation Guide](https://github.com/google/jax#installation)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
