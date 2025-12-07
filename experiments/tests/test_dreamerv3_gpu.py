import os
import sys

import pytest

jax = pytest.importorskip("jax")


def _ensure_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(project_root, "src")
    vendor = os.path.join(project_root, "third_party", "dreamerv3")
    for p in (project_root, src_dir, vendor):
        if p not in sys.path:
            sys.path.insert(0, p)


def test_dreamerv3_gpu_available_and_matmul():
    if os.environ.get("RUN_GPU_SMOKE") != "1":
        pytest.skip("Set RUN_GPU_SMOKE=1 to run GPU smoke")

    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu" or d.platform == "cuda"]
    if not gpu_devices:
        pytest.skip("No JAX GPU device available")

    # simple GPU-backed computation
    import jax.numpy as jnp

    x = jnp.ones((2, 2))
    y = jnp.eye(2)
    z = jnp.matmul(x, y)
    assert (z == 1.0).all()
    print("dreamerv3 jax gpu matmul result:", z)
