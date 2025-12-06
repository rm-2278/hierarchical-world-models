# from repo root
docker build -t hwm-smoke -f docker/Dockerfile .

docker run --rm -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc "pip install --root-user-action=ignore 'jax[cpu]>=0.4.31' && pytest -q experiments/tests/test_dreamerv3.py"