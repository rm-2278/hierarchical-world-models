<!-- # repo rootから実行
docker build -t hwm-smoke -f docker/Dockerfile .

(最初に作る場合、3分程度かかる)

docker run --rm -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc "pytest -q experiments/tests/test_tdmpc2.py"

# TD-MPC2 GPU smoke
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc 'RUN_GPU_SMOKE=1 pytest -q experiments/tests/test_tdmpc2_gpu.py -s'

-> Successfully passed in 3.07s -->


# Using docker-compose.yaml
docker compose build tdmpc
docker compose run --rm \
  -e RUN_GPU_SMOKE=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  tdmpc bash -lc "pytest -q experiments/tests/test_tdmpc2_gpu.py -s"

--> Successfully passed in 4.26s -->