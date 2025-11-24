import sys, os, numpy as np
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from models.model_demo import Agent
from data.dataset_demo import DummyImageDataset

def test_agent_and_dataset_shapes():
    a = Agent(device=None)
    ds = DummyImageDataset(size=2, img_size=64)
    img, _ = ds[0]
    action, recon = a(img)
    assert hasattr(action, "shape")
    assert recon.shape == (64,64,3)
    assert (recon >= 0.0).all() and (recon <= 1.0).all()