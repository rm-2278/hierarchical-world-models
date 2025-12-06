# repo rootから実行
docker build -t hwm-smoke -f docker/Dockerfile .

(3分程度かかる)

docker run --rm -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc "pytest -q experiments/tests/test_tdmpc2.py"

# TD-MPC2 GPU smoke
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc 'RUN_GPU_SMOKE=1 pytest -q experiments/tests/test_tdmpc2_gpu.py -s'

-> Successfully passed in 1.76s
