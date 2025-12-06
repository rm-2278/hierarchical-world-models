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
```bash
    pip install -r requirements.txt
```

### コードの実行
```bash
    python experiments/scripts/run_demo.py --config experiments/configs/demo.yaml
```

### モデル再現

-   All configs stored in `experiments/configs/`
-   Seeds fixed in training code
-   Results stored under `experiments/results/<experiment>/`

// ...existing code...

## Third-Party Code

- `third_party/dreamerv3` — sourced from @danijar/dreamerv3 (MIT License). See `third_party/dreamerv3/LICENSE`.
- `third_party/tdmpc2` — sourced from @nicklashansen/tdmpc2 (Apache-2.0). See `third_party/tdmpc2/LICENSE`.