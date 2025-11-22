# デモ実行方法

## ターミナルで以下を実行(dockerを使用)
dockerでイメージをビルドし、その後コンテナを起動して`/workspace`にマウントして起動
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
user:~/HWM/hierarchical-world-models$ sudo docker run --rm -it -v $PWD:/workspace hwm-demo bash
root@bf57af827e8f:/workspace# python experiments/scripts/run_demo.py
Training...
[train] epoch 1/3 recon_loss=0.083739
[train] epoch 2/3 recon_loss=0.083498
[train] epoch 3/3 recon_loss=0.083448
Saved agent -> experiments/results/demo_run/agent.pth
Evaluating...
[eval] step 0 action mean -0.0099
[eval] step 1 action mean -0.0095
[eval] step 2 action mean -0.0099
[eval] step 3 action mean -0.0100
[eval] step 4 action mean -0.0099
Saved reconstructions to /workspace/experiments/results/demo_run
Demo complete. Check experiments/results/demo_run
```

## その他


コードを変更したときに実行するコード。
requirements.txtなどのdependencyを変えた場合はrebuild

```bash
sudo docker build -t hwm-demo -f docker/Dockerfile .
sudo docker run --rm -it -v $PWD:/workspace hwm-demo bash
```