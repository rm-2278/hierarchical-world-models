# 開発者用

## GitHub

### 
- `main`: 常に論文・公開可能な最終コードのみ
- `develop`: 統合ブランチ
- `feature/<name>`ブランチで開発をします。
- ブランチ保護：`main`と`develop`ブランチの2つはピアレビュー必須。

### ピアレビューのルール
- テンプレートを用意（目的、変更点、関連イシュー、期待される影響）
- 少なくとも1名のレビューを必須にする（//TODO CODEOWNERSで割当）

### Issue / タスク管理
- Issue テンプレートを用意（バグ、機能、実験、記録）
- マイルストーンを日付に合わせて作成

### 実験の再現性
- 実験は`configs/`にYAMLを残す
- seed固定、乱数管理 (numpy/torch/random)

### 実験追跡ツール
- Weights & Biases
 

- 開発が本格化したら、Issuesでタスク管理をします。

## 

- docsに報告資料を集約します
- srcに純粋なコードをまとめます。

-   Use feature branches
-   Submit PRs to `develop`
-   Follow templates under `.github/`