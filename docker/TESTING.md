# Testing Guide for DreamerV3 and TDMPC2

This guide explains how to test the Docker environments and verify that DreamerV3 and TDMPC2 are correctly set up for training and inference.

## Prerequisites

- Docker 20.10+ installed
- NVIDIA Docker runtime configured
- NVIDIA GPU with CUDA 12.x compatible drivers (Driver 525.60.13+)
- At least 20GB free disk space for Docker images

## Quick Start

### 1. Verify System Requirements

```bash
# Check Docker version
docker --version

# Check NVIDIA driver
nvidia-smi

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 2. Build Docker Images

```bash
# Build all images using docker-compose
docker-compose build

# Or build individually
docker build -f docker/Dockerfile.dreamer -t hwm-dreamer .
docker build -f docker/Dockerfile.tdmpc -t hwm-tdmpc .
docker build -f docker/Dockerfile.hwm -t hwm-hwm .
```

**Note:** Building may take 30-60 minutes depending on your internet connection and system.

### 3. Run Smoke Tests

#### Test Container Creation
```bash
# Test TDMPC2 container
docker run --rm hwm-tdmpc python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test DreamerV3 container
docker run --rm hwm-dreamer python -c "import jax; print(f'JAX: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

#### Test GPU Access
```bash
# Test TDMPC2 with GPU
docker run --gpus all --rm hwm-tdmpc python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test DreamerV3 with GPU
docker run --gpus all --rm hwm-dreamer python -c "import jax; devices = jax.devices(); print(f'JAX devices: {devices}'); print(f'GPU available: {any(d.platform == \"gpu\" or d.platform == \"cuda\" for d in devices)}')"
```

## Running Tests

### TDMPC2 Tests

#### Basic Import Test
```bash
docker-compose run tdmpc pytest experiments/tests/test_tdmpc2.py -v
```

#### GPU Test (requires GPU)
```bash
docker-compose run tdmpc bash -c "export RUN_GPU_SMOKE=1 && pytest experiments/tests/test_tdmpc2_gpu.py -v"
```

### DreamerV3 Tests

#### Basic Import Test
```bash
docker-compose run dreamer pytest experiments/tests/test_dreamerv3.py -v
```

#### GPU Test (requires GPU)
```bash
docker-compose run dreamer bash -c "export RUN_GPU_SMOKE=1 && pytest experiments/tests/test_dreamerv3_gpu.py -v"
```

## Training Tests

### TDMPC2 Training Test

Create a minimal training test:

```bash
docker-compose run tdmpc python -c "
import sys
sys.path.insert(0, 'third_party/tdmpc2/tdmpc2')
from types import SimpleNamespace
import torch

# Minimal config
cfg = SimpleNamespace(
    action_dim=2,
    action_dims=[2],
    obs_shape={'state': (4,)},
    obs='state',
    task_dim=0,
    latent_dim=8,
    mlp_dim=16,
    num_bins=1,
    num_q=1,
    dropout=0.0,
    simnorm_dim=8,
    num_enc_layers=1,
    enc_dim=16,
    num_channels=16,
    log_std_min=-5.0,
    log_std_max=2.0,
    batch_size=2,
    horizon=2,
    iterations=1,
    num_samples=4,
    num_pi_trajs=0,
    num_elites=2,
    temperature=1.0,
    min_std=0.05,
    max_std=0.5,
    mpc=False,
    lr=1e-3,
    enc_lr_scale=1.0,
    episodic=False,
    multitask=False,
    tasks=[],
    discount_denom=5,
    discount_min=0.95,
    discount_max=0.995,
    episode_length=5,
    episode_lengths=[5],
    tau=0.01,
    rho=0.5,
    reward_coef=1.0,
    value_coef=1.0,
    termination_coef=1.0,
    consistency_coef=1.0,
    entropy_coef=1e-4,
    grad_clip_norm=10.0,
    compile=False,
)

# Import and create agent
from tdmpc2 import TDMPC2
agent = TDMPC2(cfg)

# Test forward pass
obs = torch.zeros(cfg.obs_shape['state'])
action = agent.act(obs, eval_mode=True)
print(f'Action shape: {action.shape}')
print('✓ TDMPC2 training test passed!')
"
```

### DreamerV3 Training Test

```bash
docker-compose run dreamer python -c "
import sys
sys.path.insert(0, 'third_party/dreamerv3')
import jax
import jax.numpy as jnp

# Simple JAX computation
x = jnp.ones((2, 2))
y = jnp.matmul(x, x)
print(f'JAX computation result: {y}')

# Check if GPU is being used
devices = jax.devices()
gpu_available = any(d.platform in ['gpu', 'cuda'] for d in devices)
print(f'GPU available for JAX: {gpu_available}')
print('✓ DreamerV3 JAX test passed!')
"
```

## Interactive Sessions

### Start Interactive TDMPC2 Session
```bash
docker-compose run tdmpc bash
# Inside container:
cd third_party/tdmpc2/tdmpc2
python train.py # (configure as needed)
```

### Start Interactive DreamerV3 Session
```bash
docker-compose run dreamer bash
# Inside container:
cd third_party/dreamerv3
python dreamerv3/main.py --logdir /tmp/logdir --configs crafter debug
```

## Troubleshooting

### Build Failures

**Issue:** Out of disk space
```bash
# Clean up Docker
docker system prune -a
docker volume prune
```

**Issue:** Network timeouts
```bash
# Use Docker build with increased timeout
DOCKER_BUILDKIT=1 docker build --network=host -f docker/Dockerfile.tdmpc -t hwm-tdmpc .
```

### Runtime Failures

**Issue:** `torch.nn.Buffer` errors
```bash
# The patches should be applied automatically during build
# If using local environment, run:
./docker/patch_buffer.sh
```

**Issue:** CUDA not available in container
```bash
# Ensure NVIDIA Docker runtime is installed
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Issue:** JAX not detecting GPU
```bash
# Check JAX CUDA plugin
docker run --gpus all --rm hwm-dreamer python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
try:
    from jax.lib import xla_bridge
    print(f'Platform: {xla_bridge.get_backend().platform}')
except Exception as e:
    print(f'Error: {e}')
"
```

### Performance Issues

**Issue:** Slow training
- Ensure GPU is being used: Check `nvidia-smi` inside container
- Reduce batch size if running out of memory
- Use the `debug` config for initial testing

## Expected Test Output

### Successful TDMPC2 Test
```
Testing TDMPC2 GPU functionality...
CUDA devices: 1
Device name: NVIDIA GeForce RTX 4090
Action shape: torch.Size([2])
✓ All TDMPC2 tests passed
```

### Successful DreamerV3 Test
```
Testing DreamerV3 JAX functionality...
JAX version: 0.4.33
JAX devices: [cuda(id=0)]
Platform: gpu
✓ All DreamerV3 tests passed
```

## Benchmarking

To verify performance is as expected:

```bash
# TDMPC2 throughput test
docker-compose run tdmpc python -c "
import time
import torch
# Your throughput test code here
"

# DreamerV3 throughput test  
docker-compose run dreamer python -c "
import time
import jax
# Your throughput test code here
"
```

## Next Steps

After successful testing:
1. Review the [main README](../README.md) for usage instructions
2. Check [docker/README.md](README.md) for detailed Docker configuration
3. Explore example configurations in `experiments/configs/`
4. Run full training experiments

## Support

If tests fail:
1. Check the error messages carefully
2. Verify all prerequisites are met
3. Review the troubleshooting section
4. Check Docker logs: `docker-compose logs`
5. Open an issue with full error output and system information
