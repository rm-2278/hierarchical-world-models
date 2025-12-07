# Quick Reference Guide

## Building Docker Images

```bash
# Build all images
docker-compose build

# Build individual images
docker-compose build dreamer    # DreamerV3
docker-compose build tdmpc      # TDMPC2
docker-compose build hwm        # Combined
```

## Running Containers

```bash
# Interactive bash session
docker-compose run dreamer bash
docker-compose run tdmpc bash

# Run a specific command
docker-compose run tdmpc python experiments/scripts/run_demo.py
```

## Testing

```bash
# Basic tests
docker-compose run tdmpc pytest experiments/tests/test_tdmpc2.py -v
docker-compose run dreamer pytest experiments/tests/test_dreamerv3.py -v

# GPU tests
docker-compose run --rm tdmpc bash -c "export RUN_GPU_SMOKE=1 && pytest experiments/tests/test_tdmpc2_gpu.py -v"
docker-compose run --rm dreamer bash -c "export RUN_GPU_SMOKE=1 && pytest experiments/tests/test_dreamerv3_gpu.py -v"
```

## Patching (if not using Docker)

```bash
# Apply patches for local development
./docker/patch_buffer.sh

# Check if patches are needed
python docker/smoke_test.py
```

## Common Commands

```bash
# Check Docker images
docker images | grep hwm

# Remove old images
docker rmi hwm-dreamer hwm-tdmpc hwm-hwm

# Clean up Docker system
docker system prune -a

# Check container logs
docker-compose logs dreamer
docker-compose logs tdmpc
```

## Environment Variables

```bash
# GPU tests
export RUN_GPU_SMOKE=1

# MuJoCo rendering
export MUJOCO_GL=egl        # Default in containers
export MUJOCO_GL=osmesa     # Alternative

# JAX memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Default in containers
```

## File Locations

- **Dockerfiles**: `docker/Dockerfile*`
- **Documentation**: `docker/README.md`, `docker/TESTING.md`
- **Tests**: `experiments/tests/`
- **Patch scripts**: `docker/patch_buffer.sh`, `docker/scripts/patch_buffer_docker.sh`
- **Third-party code**: `third_party/dreamerv3/`, `third_party/tdmpc2/`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of disk space | `docker system prune -a` |
| Build fails | Check `docker/TESTING.md` |
| CUDA not available | Verify NVIDIA Docker runtime |
| Buffer errors | Run `./docker/patch_buffer.sh` |
| JAX not detecting GPU | Check JAX CUDA plugin installation |

## Quick Links

- [Full Docker Documentation](docker/README.md)
- [Testing Guide](docker/TESTING.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Main README](README.md)

## Version Information

- **CUDA**: 12.4
- **PyTorch**: 2.5.1
- **JAX**: 0.4.33
- **Base Image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

## Getting Help

1. Check error messages carefully
2. Review [docker/TESTING.md](docker/TESTING.md) troubleshooting section
3. Verify system requirements
4. Check Docker and NVIDIA driver versions
5. Run smoke tests: `python docker/smoke_test.py`
