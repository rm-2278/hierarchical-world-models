# デモ実行方法

## ターミナルで以下を実行(dockerを使用)
dockerでイメージをビルドし、その後コンテナを起動して`/workspace`にマウントして起動
コードを変更したら毎回実行。

```bash
sudo docker build -t hwm-demo -f docker/Dockerfile .
sudo docker run --rm -it -v $PWD:/workspace hwm-demo bash
```

コンテナ内に移動してから、実行
```bash
# cd /workspace
python experiments/scripts/run_demo.py
```

## ファイル生成
- 訓練済みモデル: `experiments/results/demo_run/agent.pth`
- 再構成画像: `experiments/results/demo_run/recon_0.png` 


## 出力
```bash
root@b33560b0c32d:/workspace# python experiments/scripts/run_demo.py
Training...
100.0%
[train] epoch 1/20 recon_loss=0.012084
[train] epoch 2/20 recon_loss=0.004031
...
[train] epoch 19/20 recon_loss=0.002644
[train] epoch 20/20 recon_loss=0.002634
Saved agent -> experiments/results/demo_run/agent.pth
Evaluating...
[eval] step 0 action mean -0.0373
[eval] step 1 action mean -0.0463
[eval] step 2 action mean -0.0324
[eval] step 3 action mean -0.0344
[eval] step 4 action mean -0.0334
Saved reconstructions to /workspace/experiments/results/demo_run
Demo complete. Check experiments/results/demo_run
```

## その他

