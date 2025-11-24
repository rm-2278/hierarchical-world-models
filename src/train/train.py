import os, sys, yaml
import torch
import numpy as np
import random
from torch import nn, optim
from torch.utils.data import DataLoader

# make imports work relative to repo root / src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.model_demo import Agent
from data.dataset_demo import CIFAR10Dataset

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train(cfg_path=None):
    cfg_path = cfg_path or os.path.join(PROJECT_ROOT, "experiments", "configs", "demo.yaml")
    cfg = load_cfg(cfg_path)
    
    # Set seeds for reproducibility
    seed = cfg.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")

    agent = Agent(device=device,
                  latent_dim=cfg.get('latent_dim', 128),
                  action_dim=cfg.get('action_dim', 4))

    dataset = CIFAR10Dataset(train=True, img_size=cfg.get('img_size', 64))
    loader = DataLoader(dataset, batch_size=cfg.get('batch_size', 64), shuffle=True, num_workers=cfg.get('num_workers', 0))

    params = list(agent.encoder.parameters()) + list(agent.decoder.parameters())
    optim_w = optim.Adam(params, lr=cfg.get('lr', 1e-3))
    mse = nn.MSELoss()

    out_dir = cfg.get('out_dir', os.path.join(PROJECT_ROOT, "experiments", "results", "demo_run"))
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(cfg.get('epochs', 10)): # Default to 10 epochs
        total = 0.0
        agent.encoder.train(); agent.decoder.train()
        for imgs, _ in loader:
            imgs_t = imgs.float().permute(0,3,1,2).to(device)
            z = agent.encoder(imgs_t)
            recon = agent.decoder(z)
            loss = mse(recon, imgs_t)
            optim_w.zero_grad(); loss.backward(); optim_w.step()
            total += loss.item() * imgs_t.size(0)
        print(f"[train] epoch {epoch+1}/{cfg.get('epochs')} recon_loss={total/len(dataset):.6f}")

    save_path = os.path.join(out_dir, "agent.pth")
    agent.save(save_path)
    print("Saved agent ->", save_path)

if __name__ == "__main__":
    train()