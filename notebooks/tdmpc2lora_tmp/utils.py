import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import json
import csv
import os

# =========================================
# = = = = Log, model save Utilities = = = =
# =========================================
class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir = cfg.get_log_dir()
        self.model_dir = cfg.get_model_dir()
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"[Logger] Logs will be saved to: {self.log_dir}")
        print(f"[Logger] Models will be saved to: {self.model_dir}")
        
        self._save_config()
        self.csv_path = self.log_dir / "metrics.csv"
        # 既にファイルが存在する場合はヘッダー書き込みをスキップするフラグ
        self._header_written = os.path.exists(self.csv_path)

    def _save_config(self):
        cfg_dict = {k: str(v) for k, v in self.cfg.__dict__.items() if not k.startswith('_')}
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=4)

    def log(self, data, category="train"):
        """データをCSVに追記保存 (Pandas不要版)"""
        data = data.copy()
        data["category"] = category
        
        # 辞書のキーをヘッダー、値を列として追記
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            
            # 初回のみヘッダーを書き込む
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            
            writer.writerow(data)

    def save_agent(self, agent, step, is_best=False):
        state = {
            "model": agent.model.state_dict(),
            "optim": agent.optim.state_dict(),
            "step": step,
            "cfg": self.cfg.__dict__
        }
        torch.save(state, self.model_dir / "last.pth")
        if is_best:
            torch.save(state, self.model_dir / "best.pth")
            print(f" [CheckPoint] New best model saved at step {step}")

# ==================================
# = = = = Training Utilities = = = =
# ==================================
class Buffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.storage = []
        self.ptr = 0
    def add(self, time_step):
        if len(self.storage) < self.cfg.buffer_size:
            self.storage.append(time_step)
        else:
            self.storage[self.ptr] = time_step
            self.ptr = (self.ptr + 1) % self.cfg.buffer_size
    def sample(self):
        idxs = np.random.randint(0, len(self.storage), size=self.cfg.batch_size)
        batch = defaultdict(list)
        for i in idxs:
            ep = self.storage[i]
            length = ep['obs'].shape[0]
            if length <= self.cfg.horizon + 1: start = 0
            else: start = np.random.randint(0, length - self.cfg.horizon - 1)
            end = start + self.cfg.horizon + 1
            for k, v in ep.items():
                if v.shape[0] >= end: batch[k].append(v[start:end])
                else: batch[k].append(v[0:self.cfg.horizon+1])
        return {k: torch.stack(v, dim=1) for k, v in batch.items()}