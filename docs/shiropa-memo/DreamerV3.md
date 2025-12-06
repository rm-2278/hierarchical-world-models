# from repo root
docker build -t hwm-smoke -f docker/Dockerfile .

docker run --rm -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc "pip install --root-user-action=ignore 'jax[cpu]>=0.4.31' && pytest -q experiments/tests/test_dreamerv3.py"


# DreamerV3 JAX GPU check (ensure JAX CUDA wheel installed in the image)
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc 'RUN_GPU_SMOKE=1 pytest -q experiments/tests/test_dreamerv3_gpu.py -s'


-> Successfully passed in 1.23s