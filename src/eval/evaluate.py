import os, sys, numpy as np
from PIL import Image
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.model_demo import Agent

def evaluate(agent_path=None, n=5, out_dir=None):
    agent_path = agent_path or os.path.join(PROJECT_ROOT, "experiments", "results", "demo_run", "agent.pth")
    out_dir = out_dir or os.path.join(PROJECT_ROOT, "experiments", "results", "demo_run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent.load(agent_path, device=device, latent_dim=128, action_dim=4)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n):
        img = (np.random.rand(64,64,3) * 255).astype('uint8')
        action, recon = agent(img, eval=True)
        print(f"[eval] step {i} action mean {action.mean():.4f}")
        recon_img = (np.clip(recon, 0.0, 1.0) * 255).astype('uint8')
        Image.fromarray(recon_img).save(os.path.join(out_dir, f"recon_{i}.png"))
    print("Saved reconstructions to", out_dir)

if __name__ == "__main__":
    evaluate()