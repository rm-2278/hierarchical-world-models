# repo rootから実行
docker build -t hwm-smoke -f docker/Dockerfile .

(3分程度かかる)

docker run --rm -v "$PWD":/workspace -w /workspace hwm-smoke \
  bash -lc "pytest -q experiments/tests/test_tdmpc2.py"

