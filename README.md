# Hierarchical World Model Project

## 全体像

このレポジトリでは、階層的な世界モデルの研究を進めます。
実験から実装までを行います。

## メンバー
shiropa-uk, t-yamada02, ziwoo3244

## ファイル構造

`root/`\
 ├─ `docs/`\
 │   ├─ `mid_report/`        -- 中間報告用資料\
 │   ├─ `final_report/`      -- 最終課題レポート関連\
 │   ├─ `proposals/`      -- 研究提案書・理論背景\
 │   └─ `meetings/`          -- 議事録やメモ\
 │\
 ├─ `experiments/`
 │   ├─ `configs/`         -- YAML/JSON 実験設定\
 │   ├─ `results/`         -- 実験出力（ログ・メトリクス\
 │   └─ `scripts/`         -- 実験起動スクリプト\
 │\
 ├─ `src/`\
 │   ├─ `models/`\          -- モデルの実装\
 │   ├─ `envs/`\            -- 環境ラッパー\
 │   ├─ `train/`\           -- 訓練ループ、スケジュラー\
 │   ├─ `eval/`\            -- 評価ループ\
 │   └─ `utils/`            -- log, checkpoint, seed, config loaderなど\
 │\
 ├─ `tests/`\
 │   └─ ...                -- テストコード (unit / smoke)\
 │\
 ├─ `data/`\
 │   ├─ `raw/`               -- 生データ（git管理しない）\
 │   └─ `processed/`         -- 前処理済みデータ\
 │\
 ├─ `notebooks/`\            -- 実験・分析用ノートブック\
 │\
 ├─ `.github/`\
 │   ├─ `ISSUE_TEMPLATE/ PR_TEMPLATE/`
 │   └─ `workflows/`         -- CI/CD\
 |
 ├─ `docker/`\               -- Dockerファイル、Containerセットアップ
 │\
 ├─ `README.md`              -- レポジトリの説明\
 ├─ `CONTRIBUTING.md`        -- 開発ルール\
 ├─ `LICENSE`\
 ├─ `requirements.txt`\      -- バージョン管理\
 ├─ `pyproject.toml`\
 └─ `.gitignore`             -- プッシュしないフォルダ・ファイルを定義\

## 使い方

### 環境セットアップ

#### Option 1: Docker (Recommended)

Dockerを使用することで、CUDA 12.x互換性のある環境を簡単にセットアップできます：

```bash
# DreamerV3用
docker-compose build dreamer
docker-compose run dreamer

# TDMPC2用
docker-compose build tdmpc
docker-compose run tdmpc

# 両方を含む統合環境
docker-compose build hwm
docker-compose run hwm
```

詳細は[docker/README.md](docker/README.md)を参照してください。

#### Option 2: ローカル環境

```bash
pip install -r requirements.txt

# DreamerV3を使う場合はJAXもインストール
pip install "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 必要に応じてtorch.nn.Bufferのパッチを適用
./docker/patch_buffer.sh
```

### コードの実行
```bash
    python experiments/scripts/run_demo.py --config experiments/configs/demo.yaml
```

### モデル再現

-   All configs stored in `experiments/configs/`
-   Seeds fixed in training code
-   Results stored under `experiments/results/<experiment>/`

### モデルのテスト

```bash
# TDMPC2のテスト
export RUN_GPU_SMOKE=1
pytest experiments/tests/test_tdmpc2_gpu.py -v

# DreamerV3のテスト
export RUN_GPU_SMOKE=1
pytest experiments/tests/test_dreamerv3_gpu.py -v
```

// ...existing code...

## Third-Party Code

- `third_party/dreamerv3` — sourced from @danijar/dreamerv3 (MIT License). See `third_party/dreamerv3/LICENSE`.
- `third_party/tdmpc2` — sourced from @nicklashansen/tdmpc2 (Apache-2.0). See `third_party/tdmpc2/LICENSE`.

## 技術仕様

### CUDA互換性

すべてのDockerイメージはCUDA 12.4をサポートしており、以下と互換性があります：
- NVIDIA Driver 525.60.13以降
- CUDA 12.x ランタイム（12.0、12.1、12.2、12.3、12.4と後方互換）

注意：CUDA 13.0は存在しません。最新のCUDAバージョンは12.xです。

### PyTorch 2.0+互換性

TDMPC2コードは古い`torch.nn.Buffer` APIを使用しているため、PyTorch 2.0+では動作しません。
Dockerビルド中に自動的にパッチが適用されます。ローカル環境では`./docker/patch_buffer.sh`を実行してください。